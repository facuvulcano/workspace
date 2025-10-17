import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, Command, PythonExpression,
    TextSubstitution, EnvironmentVariable
)
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_prefix

def generate_launch_description():
    # --- Args
    model = DeclareLaunchArgument('model', default_value='burger')   # burger | waffle | waffle_pi
    x_pose = DeclareLaunchArgument('x_pose', default_value='0.0')
    y_pose = DeclareLaunchArgument('y_pose', default_value='0.0')
    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')

    # --- World & map
    pkg_sim = FindPackageShare('turtlebot3_custom_simulation')
    world = PathJoinSubstitution([pkg_sim, 'worlds', 'room.world'])
    map_file = PathJoinSubstitution([pkg_sim, 'worlds', 'map', 'map.yaml'])

    # --- Gazebo plugin/lib paths (some conda installs miss gazebo_ros exports)
    gazebo_prefix = get_package_prefix('gazebo_ros')
    gazebo_plugin_dir = os.path.join(gazebo_prefix, 'lib')
    def _prepend_unique(path, current):
        # Keep the path only once while still respecting any existing entries.
        if not current:
            return path
        if path in current.split(':'):
            return current
        return f"{path}:{current}"

    existing_plugin_path = os.environ.get('GAZEBO_PLUGIN_PATH', '').rstrip(':')
    combined_plugin_path = _prepend_unique(gazebo_plugin_dir, existing_plugin_path)
    os.environ['GAZEBO_PLUGIN_PATH'] = combined_plugin_path

    existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '').rstrip(':')
    combined_ld_path = _prepend_unique(gazebo_plugin_dir, existing_ld_path)
    os.environ['LD_LIBRARY_PATH'] = combined_ld_path

    # --- Env vars (modelo + modelos de TB3, preservando lo que ya haya)
    set_tb3_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', LaunchConfiguration('model'))
    set_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        value=[
            PathJoinSubstitution([FindPackageShare('turtlebot3_gazebo'), 'models']),
            TextSubstitution(text=':'),
            EnvironmentVariable('GAZEBO_MODEL_PATH', default_value='')
        ]
    )
    set_plugin_path = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH',
        value=combined_plugin_path
    )
    existing_system_plugin_path = os.environ.get('GAZEBO_SYSTEM_PLUGIN_PATH', '').rstrip(':')
    combined_system_plugin_path = _prepend_unique(gazebo_plugin_dir, existing_system_plugin_path)
    os.environ['GAZEBO_SYSTEM_PLUGIN_PATH'] = combined_system_plugin_path
    set_system_plugin_path = SetEnvironmentVariable(
        'GAZEBO_SYSTEM_PLUGIN_PATH',
        value=combined_system_plugin_path
    )
    set_ld_library_path = SetEnvironmentVariable(
        'LD_LIBRARY_PATH',
        value=combined_ld_path
    )

    # --- Gazebo con plugins ROS (garantiza /spawn_entity)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'verbose': 'true',
            'world': world,
            'init': 'true',
            'factory': 'true',
            'force_system': 'true'
        }.items()
    )

    # --- Publicar robot_description (xacro del TB3)
    # arma "turtlebot3_<model>.urdf" con el <model> entre comillas
    # Algunos paquetes distribuyen el URDF del TB3 con extensión .urdf.xacro,
    # pero la versión instalada vía conda solo incluye .urdf; usamos ese nombre.
    xacro_filename = PythonExpression([
        "'turtlebot3_' + '", LaunchConfiguration('model'), "' + '.urdf'"
    ])
    xacro_path = PathJoinSubstitution([
        FindPackageShare('turtlebot3_description'),
        'urdf',
        xacro_filename
    ])
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': Command(['xacro ', xacro_path])
        }],
        output='screen'
    )

    # --- Spawnear el TB3 en Gazebo (ligero delay por si Gazebo tarda en levantar)
    spawn = TimerAction(
        period=1.0,
        actions=[Node(
            package='turtlebot3_custom_simulation',
            executable='spawn_turtlebot3.py',
            arguments=[
                '--entity', 'tb3',
                '--xacro', xacro_path,
                '--model', LaunchConfiguration('model'),
                '--x', LaunchConfiguration('x_pose'),
                '--y', LaunchConfiguration('y_pose'),
                '--z', '0.2',
                '--timeout', '60'
            ],
            output='screen'
        )]
    )

    # --- TF estáticos (nombres distintos)
    tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )
    tf_map_calcodom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_to_calc_odom',
        arguments=[LaunchConfiguration('x_pose'), LaunchConfiguration('y_pose'),
                   '0', '0', '0', '0', 'map', 'calc_odom'],
        output='screen'
    )

    # --- Tu nodo y el map_server
    sim_node = Node(
        package='turtlebot3_custom_simulation',
        executable='turtlebot3_custom_simulation',
        name='turtlebot3_custom_simulation',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'odom_frame': 'calc_odom',
            'base_frame': 'calc_base_footprint',
            'joint_states_frame': 'base_footprint',
            'wheels.separation': 0.160,
            'wheels.radius': 0.033,
            'initial_pose.x': 0.0,
            'initial_pose.y': 0.0,
            'initial_pose.yaw': 0.0,
        }]
    )

    nav2_container = ComposableNodeContainer(
        name='nav2_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='nav2_map_server',
                plugin='nav2_map_server::MapServer',
                name='map_server',
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                            {'yaml_filename': map_file}]
            ),
        ],
        output='screen',
    )

    bringup_map_server = TimerAction(
        period=3.0,
        actions=[Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                        {'autostart': True},
                        {'node_names': ['map_server']}]
        )]
    )

    return LaunchDescription([
        model, x_pose, y_pose, use_sim_time,
        LogInfo(msg=[TextSubstitution(text='GAZEBO_PLUGIN_PATH='), TextSubstitution(text=combined_plugin_path)]),
        LogInfo(msg=[TextSubstitution(text='GAZEBO_SYSTEM_PLUGIN_PATH='), TextSubstitution(text=combined_system_plugin_path)]),
        LogInfo(msg=[TextSubstitution(text='LD_LIBRARY_PATH='), TextSubstitution(text=combined_ld_path)]),
        set_tb3_model, set_model_path, set_plugin_path, set_system_plugin_path, set_ld_library_path,
        gazebo, rsp, spawn,
        tf_map_odom, tf_map_calcodom,
        sim_node, nav2_container, bringup_map_server
    ])
