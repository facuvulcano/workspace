import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from custom_msgs.msg import DeltaOdom


if not hasattr(np, "float"):
    np.float = float


def wrap_angle(angle: float) -> float:
    """Keep angles within [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


@dataclass
class LandmarkEstimate:
    mu: np.ndarray
    sigma: np.ndarray


@dataclass
class Particle:
    pose: np.ndarray
    weight: float
    landmarks: Dict[int, LandmarkEstimate] = field(default_factory=dict)

    def clone(self) -> "Particle":
        """Deep copy particle for resampling."""
        cloned_landmarks = {
            lm_id: LandmarkEstimate(mu=lm.mu.copy(), sigma=lm.sigma.copy())
            for lm_id, lm in self.landmarks.items()
        }
        return Particle(pose=self.pose.copy(), weight=self.weight, landmarks=cloned_landmarks)


class FastSLAMNode(Node):
    def __init__(self) -> None:
        super().__init__("fastslam")

        self.declare_parameter("num_particles", 200)
        self.declare_parameter("alphas", [0.2, 0.2, 0.001, 0.001])
        self.declare_parameter("range_std", 0.05)
        self.declare_parameter("bearing_std", 0.05)
        self.declare_parameter("resample_ratio", 0.5)
        self.declare_parameter("publish_tf", True)

        self.num_particles = int(self.get_parameter("num_particles").value)
        self.motion_noise = np.array(self.get_parameter("alphas").value, dtype=float)
        self.range_std = float(self.get_parameter("range_std").value)
        self.bearing_std = float(self.get_parameter("bearing_std").value)
        self.resample_ratio = float(self.get_parameter("resample_ratio").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.Q = np.diag([self.range_std ** 2, self.bearing_std ** 2])
        self.identity2 = np.eye(2)

        self.particles: List[Particle] = [
            Particle(pose=np.zeros(3, dtype=float), weight=1.0 / self.num_particles)
            for _ in range(self.num_particles)
        ]

        self.pose_pub = self.create_publisher(PoseStamped, "/fastslam/pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/fastslam/landmarks", 10)
        self.path_pub = self.create_publisher(Path, "/fastslam/path", 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(DeltaOdom, "/delta", self.delta_callback, 10)
        self.create_subscription(PoseArray, "/observed_landmarks", self.landmarks_callback, 10)

        self.get_logger().info(
            f"FastSLAM initialized with {self.num_particles} particles."
        )

    def delta_callback(self, msg: DeltaOdom) -> None:
        self.predict_particles(msg)
        self.publish_best_estimate()

    def landmarks_callback(self, msg: PoseArray) -> None:
        any_update = False
        for lm_id, pose in enumerate(msg.poses):
            rng = pose.position.x
            brg = pose.position.z
            if rng <= 0.0:
                continue

            measurement = np.array([rng, brg], dtype=float)
            for particle in self.particles:
                self.correct_particle(particle, lm_id, measurement)
            any_update = True

        if any_update:
            self.normalize_weights()
            self.resample_if_needed()
            self.publish_best_estimate()

    def predict_particles(self, delta: DeltaOdom) -> None:
        alpha1, alpha2, alpha3, alpha4 = self.motion_noise

        for particle in self.particles:
            rot1 = float(delta.dr1)
            rot2 = float(delta.dr2)
            trans = float(delta.dt)

            rot1_hat = rot1 + np.random.normal(0.0, math.sqrt(alpha1 * rot1 ** 2 + alpha2 * trans ** 2))
            trans_hat = trans + np.random.normal(0.0, math.sqrt(alpha3 * trans ** 2 + alpha4 * (rot1 ** 2 + rot2 ** 2)))
            rot2_hat = rot2 + np.random.normal(0.0, math.sqrt(alpha1 * rot2 ** 2 + alpha2 * trans ** 2))

            theta = particle.pose[2]
            theta += rot1_hat
            theta = wrap_angle(theta)

            particle.pose[0] += trans_hat * math.cos(theta)
            particle.pose[1] += trans_hat * math.sin(theta)

            theta += rot2_hat
            particle.pose[2] = wrap_angle(theta)

    def correct_particle(self, particle: Particle, lm_id: int, measurement: np.ndarray) -> None:
        if lm_id not in particle.landmarks:
            self.initialize_landmark(particle, lm_id, measurement)
            return

        landmark = particle.landmarks[lm_id]
        expected_z = self.expected_measurement(particle.pose, landmark.mu)
        H = self.measurement_jacobian(particle.pose, landmark.mu)

        S = H @ landmark.sigma @ H.T + self.Q
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S += np.eye(2) * 1e-6
            S_inv = np.linalg.inv(S)

        K = landmark.sigma @ H.T @ S_inv
        innovation = measurement - expected_z
        innovation[1] = wrap_angle(float(innovation[1]))

        landmark.mu = landmark.mu + K @ innovation
        landmark.sigma = (self.identity2 - K @ H) @ landmark.sigma

        particle.weight *= self.gaussian_probability(innovation, S, S_inv)

    def initialize_landmark(self, particle: Particle, lm_id: int, measurement: np.ndarray) -> None:
        rng, brg = measurement
        theta = particle.pose[2]
        global_bearing = theta + brg

        mu = np.array(
            [
                particle.pose[0] + rng * math.cos(global_bearing),
                particle.pose[1] + rng * math.sin(global_bearing),
            ],
            dtype=float,
        )

        H = self.measurement_jacobian(particle.pose, mu)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.eye(2) * 1e-3
        else:
            cov = H_inv @ self.Q @ H_inv.T

        particle.landmarks[lm_id] = LandmarkEstimate(mu=mu, sigma=cov)

    def expected_measurement(self, pose: np.ndarray, landmark_mu: np.ndarray) -> np.ndarray:
        dx = landmark_mu[0] - pose[0]
        dy = landmark_mu[1] - pose[1]
        rng = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - pose[2])
        return np.array([rng, bearing], dtype=float)

    def measurement_jacobian(self, pose: np.ndarray, landmark_mu: np.ndarray) -> np.ndarray:
        dx = landmark_mu[0] - pose[0]
        dy = landmark_mu[1] - pose[1]
        q = dx * dx + dy * dy
        if q < 1e-9:
            q = 1e-9
        rng = math.sqrt(q)

        return np.array(
            [
                [dx / rng, dy / rng],
                [-dy / q, dx / q],
            ],
            dtype=float,
        )

    def gaussian_probability(self, innovation: np.ndarray, cov: np.ndarray, cov_inv: np.ndarray) -> float:
        det = max(float(np.linalg.det(cov)), 1e-12)
        norm = 1.0 / (2.0 * math.pi * math.sqrt(det))
        exponent = -0.5 * (innovation.T @ cov_inv @ innovation)
        return max(norm * math.exp(float(exponent)), 1e-12)

    def normalize_weights(self) -> None:
        total = sum(p.weight for p in self.particles)
        if total <= 0.0:
            equal_weight = 1.0 / self.num_particles
            for particle in self.particles:
                particle.weight = equal_weight
            return

        for particle in self.particles:
            particle.weight /= total

    def effective_sample_size(self) -> float:
        weights = np.array([p.weight for p in self.particles], dtype=float)
        return 1.0 / np.sum(np.square(weights))

    def resample_if_needed(self) -> None:
        neff = self.effective_sample_size()
        if neff >= self.resample_ratio * self.num_particles:
            return

        weights = np.array([p.weight for p in self.particles], dtype=float)
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0 

        positions = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
        indexes = np.searchsorted(cumulative, positions)

        new_particles = []
        for idx in indexes:
            cloned = self.particles[int(idx)].clone()
            cloned.weight = 1.0 / self.num_particles
            new_particles.append(cloned)

        self.particles = new_particles

    def publish_best_estimate(self) -> None:
        if not self.particles:
            return

        best_particle = max(self.particles, key=lambda p: p.weight)
        stamp = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = float(best_particle.pose[0])
        pose_msg.pose.position.y = float(best_particle.pose[1])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = quaternion_from_yaw(float(best_particle.pose[2]))
        self.pose_pub.publish(pose_msg)

        pose_for_path = PoseStamped()
        pose_for_path.header = pose_msg.header
        pose_for_path.pose = pose_msg.pose
        self.path_msg.poses.append(pose_for_path)
        self.path_msg.header.stamp = stamp
        self.path_pub.publish(self.path_msg)

        if self.publish_tf:
            self.publish_tf_transform(best_particle.pose, stamp)

        self.publish_landmark_markers(best_particle, stamp)

    def publish_tf_transform(self, pose: np.ndarray, stamp) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_footprint"
        transform.transform.translation.x = float(pose[0])
        transform.transform.translation.y = float(pose[1])
        transform.transform.translation.z = 0.0
        transform.transform.rotation = quaternion_from_yaw(float(pose[2]))
        self.tf_broadcaster.sendTransform(transform)

    def publish_landmark_markers(self, best_particle: Particle, stamp) -> None:
        ma = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        for lm_id, landmark in sorted(best_particle.landmarks.items()):
            ma.markers.append(
                self.make_landmark_marker(lm_id * 2, landmark.mu[0], landmark.mu[1], stamp)
            )
            ma.markers.append(
                self.make_covariance_marker(lm_id * 2 + 1, landmark.mu[0], landmark.mu[1], landmark.sigma, stamp)
            )

        self.marker_pub.publish(ma)

    def make_landmark_marker(self, idx: int, x: float, y: float, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.id = idx
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m

    def make_covariance_marker(self, idx: int, x: float, y: float, cov: np.ndarray, stamp) -> Marker:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = math.atan2(vecs[1, 0], vecs[0, 0])
        scale_x = 30.0 * 2.0 * math.sqrt(max(vals[0], 1e-9))
        scale_y = 30.0 * 2.0 * math.sqrt(max(vals[1], 1e-9))

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.id = idx
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation = quaternion_from_yaw(angle)
        m.scale.x = scale_x
        m.scale.y = scale_y
        m.scale.z = 0.01
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 0.3
        return m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FastSLAMNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
