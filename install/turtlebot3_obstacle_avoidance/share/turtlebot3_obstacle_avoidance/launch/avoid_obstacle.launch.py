from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="turtlebot3_obstacle_avoidance",
                executable="obstacle_avoidance",
                name="obstacle_avoidance",
                output="screen",
                parameters=[
                    {
                        "linear_speed": 0.5,
                        "angular_speed": 1.0,
                        "rotation_angle_deg": 110.0,
                        "obstacle_distance": 0.5,
                        "front_sector_width_deg": 80.0,
                        "filter_zero_intensity": False,
                    }
                ],
            )
        ]
    )
