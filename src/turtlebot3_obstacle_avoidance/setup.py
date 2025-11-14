from setuptools import setup

package_name = "turtlebot3_obstacle_avoidance"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/avoid_obstacle.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="facuvulcano",
    maintainer_email="unknown@example.com",
    description="Nodo simple de evasión de obstáculos para Turtlebot3 usando el lidar.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "obstacle_avoidance = turtlebot3_obstacle_avoidance.obstacle_avoidance_node:main",
        ],
    },
)
