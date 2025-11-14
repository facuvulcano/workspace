import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from custom_msgs.msg import Belief


MEASUREMENT_NOISE = np.diag([0.05 ** 2, 0.05 ** 2])


def normalize_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


class EKFCorrectionNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_correction")

        self._mu = None
        self._covariance = None
        self._landmarks = None
        self._belief_ready = False
        self._landmarks_ready = False

        self._belief_pub = self.create_publisher(Belief, "/belief", 10)

        self.create_subscription(Belief, "/belief", self._belief_callback, 10)
        self.create_subscription(PoseArray, "/landmarks", self._landmarks_callback, 10)
        self.create_subscription(
            PoseArray, "/observed_landmarks", self._observations_callback, 10
        )

    def _belief_callback(self, msg: Belief) -> None:
        self._mu = np.array([msg.mu.x, msg.mu.y, msg.mu.theta])
        self._covariance = np.array(
            [
                [msg.covariance[0], msg.covariance[1], msg.covariance[2]],
                [msg.covariance[3], msg.covariance[4], msg.covariance[5]],
                [msg.covariance[6], msg.covariance[7], msg.covariance[8]],
            ]
        )
        self._belief_ready = True

    def _landmarks_callback(self, msg: PoseArray) -> None:
        self._landmarks = np.array(
            [(pose.position.x, pose.position.y) for pose in msg.poses], dtype=float
        )
        self._landmarks_ready = True

    def _observations_callback(self, msg: PoseArray) -> None:
        if not (self._belief_ready and self._landmarks_ready):
            return

        if self._landmarks is None or len(self._landmarks) == 0:
            return

        mu = self._mu.copy()
        covariance = self._covariance.copy()
        updated = False

        for idx, pose in enumerate(msg.poses):
            if idx >= len(self._landmarks):
                break

            measured_range = float(pose.position.x)
            measured_bearing = float(pose.position.z)

            if measured_range <= 0.0:
                continue

            measured_bearing = normalize_angle(measured_bearing)

            landmark_x, landmark_y = self._landmarks[idx]
            dx = landmark_x - mu[0]
            dy = landmark_y - mu[1]

            q = dx * dx + dy * dy
            if q < 1e-9:
                continue

            expected_range = np.sqrt(q)
            expected_bearing = normalize_angle(np.arctan2(dy, dx) - mu[2])
            z_hat = np.array([expected_range, expected_bearing])

            residual = np.array([measured_range, measured_bearing]) - z_hat
            residual[1] = normalize_angle(residual[1])

            H = np.array(
                [
                    [-dx / expected_range, -dy / expected_range, 0.0],
                    [dy / q, -dx / q, -1.0],
                ]
            )

            S = H @ covariance @ H.T + MEASUREMENT_NOISE

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self.get_logger().warning(
                    f"Innovation covariance is singular for landmark {idx}; skipping"
                )
                continue

            K = covariance @ H.T @ S_inv

            mu = mu + K @ residual
            mu[2] = normalize_angle(mu[2])

            identity = np.eye(3)
            covariance = (
                (identity - K @ H) @ covariance @ (identity - K @ H).T
                + K @ MEASUREMENT_NOISE @ K.T
            )
            updated = True

        if not updated:
            return

        self._mu = mu
        self._covariance = covariance

        detP = float(np.linalg.det(covariance))
        self.get_logger().info(
            f"EKF correction applied; det(P)={detP:.3e}, mu=({float(mu[0]):.3f}, {float(mu[1]):.3f}, {float(mu[2]):.3f})"
        )

        belief_msg = Belief()
        belief_msg.mu.x = float(mu[0])
        belief_msg.mu.y = float(mu[1])
        belief_msg.mu.theta = float(mu[2])
        belief_msg.covariance = covariance.flatten().astype(float).tolist()

        self._belief_pub.publish(belief_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EKFCorrectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
