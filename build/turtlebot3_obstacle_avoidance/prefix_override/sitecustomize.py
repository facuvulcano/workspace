import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/alumno1/workspace/install/turtlebot3_obstacle_avoidance'
