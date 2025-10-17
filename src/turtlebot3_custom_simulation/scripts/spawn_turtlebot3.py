#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List

import rclpy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity
from rclpy.node import Node


def _build_pose(args: argparse.Namespace) -> Pose:
    pose = Pose()
    pose.position.x = float(args.x)
    pose.position.y = float(args.y)
    pose.position.z = float(args.z)
    pose.orientation.x = float(args.qx)
    pose.orientation.y = float(args.qy)
    pose.orientation.z = float(args.qz)
    pose.orientation.w = float(args.qw)
    return pose


def _load_urdf(xacro_path: str, extra_args: List[str]) -> str:
    from subprocess import check_output, CalledProcessError
    try:
        cmd = ['xacro', xacro_path, *extra_args]
        return check_output(cmd).decode()
    except (CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f'Failed to run {" ".join(cmd)}: {exc}') from exc


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description='Spawn TurtleBot3 into Gazebo with fallback services')
    parser.add_argument('--entity', required=True, help='Entity name to use inside Gazebo')
    parser.add_argument('--xacro', required=True, help='Path to the TurtleBot3 xacro/urdf file')
    parser.add_argument('--model', required=True, help='TurtleBot3 model name (burger|waffle|waffle_pi)')
    parser.add_argument('--x', default='0.0')
    parser.add_argument('--y', default='0.0')
    parser.add_argument('--z', default='0.2')
    parser.add_argument('--qx', default='0.0')
    parser.add_argument('--qy', default='0.0')
    parser.add_argument('--qz', default='0.0')
    parser.add_argument('--qw', default='1.0')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--robot-namespace', default='')
    parser.add_argument('--xacro-arg', action='append', default=[], help='Extra KEY=VALUE pairs for xacro')
    args, ros_args = parser.parse_known_args(argv)

    xacro_path = Path(args.xacro)
    if not xacro_path.exists():
        raise FileNotFoundError(f'xacro file not found: {xacro_path}')

    urdf_xml = _load_urdf(str(xacro_path), args.xacro_arg)

    rclpy.init(args=[sys.argv[0], *ros_args])
    node = Node('spawn_turtlebot3')
    pose = _build_pose(args)
    service_candidates = ['/spawn_entity', '/gazebo/spawn_entity', '/default/spawn_entity']
    timeout = float(args.timeout)
    success = False
    for service_name in service_candidates:
        client = node.create_client(SpawnEntity, service_name)
        node.get_logger().info(f'Waiting for service {service_name} (timeout {timeout}s)')
        if not client.wait_for_service(timeout_sec=timeout):
            node.get_logger().warn(f'Service {service_name} unavailable')
            continue

        node.get_logger().info(f'Calling service {service_name}')
        request = SpawnEntity.Request()
        request.name = args.entity
        request.xml = urdf_xml
        request.robot_namespace = args.robot_namespace or args.entity
        request.initial_pose = pose
        request.reference_frame = 'world'
        future = client.call_async(request)
        if rclpy.spin_until_future_complete(node, future, timeout_sec=timeout) and future.result():
            response = future.result()
            if response.success:
                node.get_logger().info(response.status_message)
                success = True
                break
            node.get_logger().error(f'Failed via {service_name}: {response.status_message}')
        else:
            node.get_logger().error(f'Service call to {service_name} timed out')

    rclpy.shutdown()
    if not success:
        raise RuntimeError('Unable to spawn TurtleBot3; tried services: ' + ', '.join(service_candidates))


if __name__ == '__main__':
    main()
