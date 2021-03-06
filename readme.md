# navigation_test

| NavBench currently only supports ROS Kinetic |
| --- |


NavBench is the local planner navigation testbench used in IVALab. This testbench is capable of testing controllers of different configurations in different world of randomly populated obstacles and randomly assigned start and goal position. 

There are two packages under this repository, ```nav_scripts``` and ```nav_configs```. ```nav_scripts``` are the package that initiates the testing. ```nav_configs``` contains the supporting files for ```nav_scripts```, including robot and sensor models, configuration files for controllers, Gazebo worlds, and launch files for starting simulations. More details will be covered in subsequent sections. 

## Test Worlds
Provided worlds include:
- dense: empty square world with only side walls
- full_campus_obstacle (Campus World): outdoor free space consists of large free spaces connected by narrower corridors.
- fourth_floor_world (Office World): simplified model of the fourth floor of the building containing our lab. 

To visualize the worlds, edit the launch files in ```nav_configs/launch/``` so that default value of the ```gui``` argument is true. Run one of the following to visualize the world.

```bash
$ roslaunch nav_configs gazebo_turtlebot_empty_room_20x20_world.launch
$ roslaunch nav_configs gazebo_turtlebot_campus_obstacle_world.launch
$ roslaunch nav_configs gazebo_turtlebot_fourth_floor_obstacle_world.launch
```

## Timing Demo
1. Download sample rosbag of prerecorded data from:
'''
https://drive.google.com/file/d/10glIc6hyeFE1OTMfC8wrL3jjvh4lklRz/view?usp=sharing
'''

2. Replace `[full path to]demo_rosbag_2020-05-30-17-05-50.bag` in `nav_scripts/scripts/launch/rosbag/rosbag.launch` with the full path to the downloaded rosbag

3. Run timing demo
```
rosrun nav_scripts timing_demo.py
```

