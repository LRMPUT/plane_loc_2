[//]: <> (---)
[//]: <> (header-includes:)
[//]: <> ( - \usepackage{fvextra})
[//]: <> ( - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}})
[//]: <> (---)

# PlaneLoc2 ROS

A ROS package with PlaneLoc2 - an open source project that provides a probabilistic framework for global localization using planar segments.

Prerequesties:

+ Boost  
+ Eigen  
+ PCL 1.8  
+ OpenCV >= 3.0 
+ CGAL

## Paper

If you find PlaneLoc useful in your academic work please cite the following paper:

    @article{wietrzykowski2019,
        title = {{PlaneLoc}: Probabilistic global localization in {3-D} using local planar features},
        author = {Jan Wietrzykowski and Piotr Skrzypczy\'{n}ski},
        journal = {Robotics and Autonomous Systems},
        volume = {113},
        pages = {160 - 173},
        year = {2019},
        issn = {0921-8890},
        doi = {https://doi.org/10.1016/j.robot.2019.01.008},
        url = {http://www.sciencedirect.com/science/article/pii/S0921889018303701},
        keywords = {Global localization, SLAM, Planar segments, RGB-D data},
    }


## Building:

Tested on Ubuntu 20.04 and ROS Noetic.

1. Install dependencies:
     ```bash
     sudo apt install libboost-system-dev libboost-filesystem-dev libboost-serialization-dev libopencv-dev libpcl-dev libcgal-dev ros-noetic-cv-bridge ros-noetic-sensor-msgs ros-noetic-pcl-conversions ros-noetic-tf2 ros-noetic-tf2-ros ros-noetic-tf2-msgs ros-noetic-tf2-eigen ros-noetic-image-transport libeigen3-dev
   ```
   
2. Clone libnabo package:
    ```bash
    cd ~/catkin_ws/src
    clone https://github.com/ethz-asl/libnabo.git
    cd ..
    ```

3. Clone and build PlaneLoc2:
    ```bash
    cd ~/catkin_ws/src
    clone https://github.com/LRMPUT/plane_loc_2.git
    cd ..
    catkin catkin build plane_loc_2 -DCMAKE_BUILD_TYPE=Release
    ```
   
   
## Installing bindings

```bash
cd ~/catkin_ws/src
pip install ./plane_loc
```

## Usage

To run PlaneLoc2:
+ Process dataset with [Stereo Plane R-CNN](https://github.com/LRMPUT/sprcnn) and export detection and descriptors.
+ Pack exported information into a rosbag using `pack_rosbag_planercnn.py` script.
+ Adjust paths in `MapperBagNode.cpp` to a proper rosbag path and choose between building a global map and creating a rosbag with local maps.
+ Select a value of `acc_duration` in `mapper.launch`. Use 2 seconds for local maps and infinity for a global map.
+ Build a global map and create a rosbag with local maps.
+ Adjust paths in `LocalizerBagNode.cpp` to a proper local maps rosbag.
+ Adjust a global map path and camera parameters in `settings.yml`.

## Dataset

The `SceneNet Stereo` dataset can be downloaded [here](https://putpoznanpl-my.sharepoint.com/:f:/g/personal/jan_wietrzykowski_put_poznan_pl/ErZm6If9-91JtW7BEK4pXJcBWKLwhoujwisDu_tLDjik2Q?e=vgnlnM).

The `TERRINet` dataset can be downloaded [here](https://putpoznanpl-my.sharepoint.com/:f:/g/personal/jan_wietrzykowski_put_poznan_pl/Eqj0TnSgDrlJuJu0FC-bVGEB2hbpWHC_YA_l_qs9EDgkjw?e=JYRBIT).

## Nodes

### mapper_node

Mapper node processes detected planar segments from consecutive frames and merges them into a larger local map using odometry transformation from TF2.  

#### Subscribed topics

+ _objs_topic_ (plane_loc/Serialized) - detected planar segments as a serialized _vectorObjInstance_ C++ type. There should be TF2 transformation available from _odom_ frame to frame specified in the message header at the time specified in the header. 

#### Published topics

+ map (plane_loc/Serialized) - accumulated maps as a serialized _Map_ C++ type.

#### Parameters

+ settings (string, default: _settings.yaml_) - path to the settings YAML file.
+ objs_topic (string, default: _/detector/objs_) - name of the topic with detected planar segments.
+ acc_frames (int, default: 50) - number of frames that are accumulated before publishing the map.
+ merge_map_frames (int, default: 50) - number of frames after which additional check of all planar segments is performed and those that fulfill conditions for merging are merged.


### localizer_node

Localizer node subscribes accumulated local maps and performs global localization against the map specified in the settings YAML file. It evaluates performance using TF2 transformations from _map_ frame.    

#### Subscribed topics

+ _map_topic_ (plane_loc/Serialized) - accumulated local maps as a serialized _Map_ C++ type.

#### Published topics

+ pose (geometry_msgs/TransformStamped) - computed pose of the sensor at the time specified in the header of map message. Published only if the transformation was considered as correct.

#### Parameters

+ settings (string) - path to the settings YAML file.
+ map_topic (string, default: _/mapper/map_) - name of the topic with detected planar segments.
+ score_thresh (float, default: 1.0) - minimal value of score to consider a transformation as correct.
+ score_diff_thresh (float, default: 0.0) - minimal value of score difference between the best and the second best transformation to consider the first one as correct.
+ fit_thresh (float, default: 0.07) - maximal value of fitness test to consider a transformation as correct.
+ distinct_thresh (int, default: 6) - minimal number of distinct matched planar segments to consider a transformation as correct.
+ pose_diff_thresh (float, default: 0.16) - maximal value of squared SE3 logarithm of difference between computed and ground truth pose to count the result as positive (used only for evaluation).


## Launching

The package can be launched using the provided launch file:
```bash
roslaunch plane_loc plane_loc.launch
```
Make sure you slow down rosbag play using _r_ flag, so the system can keep up with computation:
```bash
rosbag play /mnt/data/datasets/PUT_Indoor/2017_04_04_test1/data.bag --pause -r 0.02
```

## CLion configuration

CMake options:
```bash
-DCATKIN_DEVEL_PREFIX:PATH=/home/lrm/catkin_ws/devel
```
env variables:
```bash
CMAKE_PREFIX_PATH=/home/lrm/catkin_ws/devel:/opt/ros/noetic;PYTHONPATH=/home/lrm/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
```

Run configuration arguments:
```bash
__name:=mapper
```
env variables:
```bash
ROS_NAMESPACE=/;LD_LIBRARY_PATH=/home/lrm/catkin_ws/devel/lib:/opt/ros/noetic/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64;CMAKE_PREFIX_PATH=/home/lrm/catkin_ws/devel:/opt/ros/noetic;ROS_PACKAGE_PATH=/home/lrm/catkin_ws/src/plane_loc:/opt/ros/noetic/share;DISPLAY=:1
```

To enable X server in debug mode add to `~/.gdbinit`:
```bash
set environment DISPLAY :1
```