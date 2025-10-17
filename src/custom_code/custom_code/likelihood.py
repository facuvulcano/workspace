import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


class LikelihoodMapPublisher(Node):
    def __init__(self):
        super().__init__('likelihood_map_publisher')
        qos = rclpy.qos.QoSProfile(depth=1)
        qos.durability = rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(OccupancyGrid, '/likelihood_map', qos)
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos
        )
    
    def map_callback(self, msg):
        prob_msg = OccupancyGrid()
        prob_msg.header = msg.header
        prob_msg.info = msg.info

        width, height = msg.info.width, msg.info.height
        if width * height != len(msg.data):
            self.get_logger().error('OccupancyGrid data size does not match width*height')
            return

        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        OCC_THRESH = 50    
        DILATE_ITERS = 1         
        SIGMA = 0.30        
        KEEP_UNKNOWN_AS_UNKNOWN = True 
        TREAT_UNKNOWN_AS_OBSTACLE = False 

        unknown_mask  = (grid == -1)
        occupied_mask = (grid >= OCC_THRESH)

        if DILATE_ITERS > 0:
            occupied_mask = binary_dilation(occupied_mask, iterations=DILATE_ITERS)

        obstacle_mask = occupied_mask | (unknown_mask if TREAT_UNKNOWN_AS_OBSTACLE else False)

        if np.any(obstacle_mask):
            distances = distance_transform_edt(~obstacle_mask).astype(np.float32) * msg.info.resolution
        else:
            max_distance = np.float32(max(width, height) * msg.info.resolution)
            distances = np.full((height, width), max_distance, dtype=np.float32)

        with np.errstate(over='ignore', under='ignore'):
            likelihood = np.exp(-0.5 * (distances / SIGMA) ** 2).astype(np.float32)

        likelihood[obstacle_mask] = 1.0

        out = np.clip(np.rint(likelihood * 100.0), 0, 100).astype(np.int8)
        if KEEP_UNKNOWN_AS_UNKNOWN:
            out[unknown_mask] = -1 

        prob_msg.data = out.reshape(-1).tolist()
        self.pub.publish(prob_msg)
        self.get_logger().info("Published likelihood map")

def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()