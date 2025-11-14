#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("custom_code")
    default_rviz = os.path.join(pkg_share, "rviz", "fastslam.rviz")

    num_particles_arg = DeclareLaunchArgument(
        "num_particles",
        default_value="200",
        description="Cantidad de partículas para FastSLAM",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="Archivo de configuración RViz",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Lanzar RViz junto con el nodo FastSLAM",
    )

    fastslam_node = Node(
        package="custom_code",
        executable="fastslam",
        name="fastslam",
        output="screen",
        parameters=[
            {
                "num_particles": LaunchConfiguration("num_particles"),
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    ld = LaunchDescription()
    ld.add_action(num_particles_arg)
    ld.add_action(rviz_config_arg)
    ld.add_action(use_rviz_arg)
    ld.add_action(fastslam_node)
    ld.add_action(rviz_node)
    return ld
