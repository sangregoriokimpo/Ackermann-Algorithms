#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Use simulation clock if true"
    )

    pkg_share = get_package_share_directory("ackermann_gazebo")

    urdf_path = os.path.join(pkg_share, "urdf", "sim.urdf.xacro")
    gazebo_world_path = os.path.join(pkg_share, "worlds", "empty.world")

    # Gazebo (server + client)
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("gazebo_ros"), "launch"),
                "/gazebo.launch.py",
            ]
        ),
        launch_arguments={"world": gazebo_world_path}.items(),
    )

    # Define multiple robots: name, namespace, pose
    robots = [
        {"name": "bot1", "ns": "bot1", "x": "0.0",  "y": "6.0",  "z": "0.0"},
        {"name": "bot2", "ns": "bot2", "x": "5.0",  "y": "6.0",  "z": "0.0"},
        {"name": "bot3", "ns": "bot3", "x": "-5.0", "y": "6.0",  "z": "0.0"},
    ]

    robot_groups = []

    for r in robots:
        ns = r["ns"]

        robot_description = xacro.process_file(urdf_path).toxml()

        # Robot state publisher in namespace
        rsp_node = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=ns,
            output="screen",
            parameters=[
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "robot_description": robot_description,
                }
            ],
        )

        # Spawn entity in the same namespace so it sees that robot_description
        spawn_node = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            namespace=ns,  # <-- critical
            arguments=[
                "-topic", "robot_description",
                "-entity", r["name"],
                "-x", r["x"],
                "-y", r["y"],
                "-z", r["z"],
                "-robot_namespace", ns,
            ],
            output="screen",
        )

        # Autonomous driver for this robot, also in that namespace
        auto_driver_node = Node(
            package="autonomous_driver",
            executable="fusion_gap_driverE0",
            namespace=ns,
            output="screen",
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        )

        robot_groups.append(GroupAction([rsp_node, spawn_node, auto_driver_node]))

    return LaunchDescription(
        [
            use_sim_time_arg,
            gazebo_node,
            *robot_groups,
        ]
    )
