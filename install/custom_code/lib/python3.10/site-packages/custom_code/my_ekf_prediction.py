import numpy as np

import rclpy
from rclpy.node import Node

from custom_msgs.msg import Belief, DeltaOdom


QT = np.diag([0.02, 0.02, 0.02])


def normalize_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


class EKFPredictionNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_prediction")

        self._mu = None
        self._covariance = None
        self._belief_ready = False

        self._belief_pub = self.create_publisher(Belief, "/belief", 10)

        self.create_subscription(Belief, "/belief", self._belief_callback, 10)
        self.create_subscription(DeltaOdom, "/delta", self._delta_callback, 10)

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

    def _delta_callback(self, msg: DeltaOdom) -> None:
        if not self._belief_ready:
            return

        dr1 = msg.dr1
        dr2 = msg.dr2
        dt = msg.dt

        theta = self._mu[2]
        theta_bar = theta + dr1

        predicted_mu = np.array(
            [
                self._mu[0] + dt * np.cos(theta_bar),
                self._mu[1] + dt * np.sin(theta_bar),
                normalize_angle(theta + dr1 + dr2),
            ]
        )

        gt = np.array(
            [
                [1.0, 0.0, -dt * np.sin(theta_bar)],
                [0.0, 1.0, dt * np.cos(theta_bar)],
                [0.0, 0.0, 1.0],
            ]
        )

        predicted_covariance = gt @ self._covariance @ gt.T + QT

        self._mu = predicted_mu
        self._covariance = predicted_covariance

        belief_msg = Belief()
        belief_msg.mu.x = float(self._mu[0])
        belief_msg.mu.y = float(self._mu[1])
        belief_msg.mu.theta = float(self._mu[2])
        belief_msg.covariance = self._covariance.flatten().astype(float).tolist()

        self._belief_pub.publish(belief_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EKFPredictionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
