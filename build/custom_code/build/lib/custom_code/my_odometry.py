#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from custom_code.robot_functions import RobotFunctions

class Odom3Node(Node):
    def __init__(self):
        super().__init__("odom3")

        self.declare_parameter("alpha", [1.0, 1.0, 0.1, 0.1])
        self.alpha = self.get_parameter("alpha").get_parameter_value().double_array_value

        self.subscription = self.create_subscription(
            Odometry, "/calc_odom", self.odom_callback, 10
        )
        
        # Create a publisher for PointCloud2
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "/particle_cloud", 10
        )

        self.last_odom = None  # Store previous odometry state
        self.i = 0

        self.robot = RobotFunctions()

    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "calc_odom"  # or whatever your frame is
        
        if len(points.shape) == 3:
            points = points.reshape(-1, 3)
        
        msg.height = 1
        msg.width = len(points)
        
        # Define the point fields (x, y, z)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12  # 3 floats * 4 bytes each
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Convert points to bytes
        msg.data = points.astype(np.float32).tobytes()
        
        return msg


    def odom_callback(self, data: Odometry):
        self.i += 1
        self.i = self.i % 40  # Reset every 100 messages
        if self.i != 0:
            return
        
        """Process odometry using quaternion differencing."""
        num_samples = 100
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        # Extract quaternion (w, x, y, z) format
        q_w = data.pose.pose.orientation.w
        q_x = data.pose.pose.orientation.x
        q_y = data.pose.pose.orientation.y
        q_z = data.pose.pose.orientation.z

        # Convert quaternion to rotation object
        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]  # Extract yaw

        if self.last_odom is not None:
            prev_x, prev_y, prev_theta, prev_q = self.last_odom
            xtold = np.array([prev_x, prev_y, prev_theta])
            
            # Translation difference
            dx = x - prev_x
            dy = y - prev_y
            delta_trans = np.sqrt(dx**2 + dy**2)

            if delta_trans > 1e-6:
                delta_rot1 = np.arctan2(dy, dx) - prev_theta
                delta_rot2 = theta - prev_theta - delta_rot1
            else:
                # No translation → assume in-place rotation
                delta_rot1 = 0.0
                delta_rot2 = theta - prev_theta

            # Normalize angles
            delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
            delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

            ut = np.array([delta_rot1, delta_trans, delta_rot2])

            # Apply odometry motion model
            samples = np.array([self.robot.odometry_motion_model(list(xtold).copy(), list(ut).copy(), alpha=self.alpha) for _ in range(num_samples)])
            
            # Convert samples to PointCloud2 and publish
            # Add z=0 to make it 3D (required by PointCloud2)
            samples_3d = np.hstack([samples[:, :2], np.zeros((samples.shape[0], 1))])
            pointcloud_msg = self.create_pointcloud2(samples_3d)
            self.pointcloud_pub.publish(pointcloud_msg)

        else:
            samples = np.array([[x, y, theta]])  # First frame
            # Publish initial samples
            samples_3d = np.hstack([samples[:, :2], np.zeros((samples.shape[0], 1))])
            pointcloud_msg = self.create_pointcloud2(samples_3d)
            self.pointcloud_pub.publish(pointcloud_msg)

        # Store the last odometry state
        self.last_odom = (x, y, theta, [q_x, q_y, q_z, q_w])

def main(args=None):
    rclpy.init(args=args)
    node = Odom3Node()
    
    try:
        rclpy.spin(node)  # Keep ROS 2 running
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
