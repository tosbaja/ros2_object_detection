import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='opencv_package',  # Replace with your package name
            executable='opencv_node',  # Replace with your executable name
            name='opencv_node'
        )
    ])
