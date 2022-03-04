//
// Created by janw on 16.12.2019.
//

#include "ros/ros.h"

#include <Mapper.hpp>

using namespace std;

int main(int argc, char** argv){
    ros::init( argc, argv, "mapper" );

    Mapper mapper;
    // use this to build a global map
    mapper.runBag("/mnt/data/TERRINet/scenes/scene0000_00/scene0000_00_depth.bag",
                  "",
                  "/mnt/data/TERRINet/scenes/scene0000_00/scene0000_00.map");
    // use this to create a bag of local maps
    // mapper.runBag("/mnt/data/TERRINet/scenes/scene0000_02/scene0000_02_depth.bag",
    //               "/mnt/data/TERRINet/scenes/scene0000_02/scene0000_02_map.bag",
    //               "");

    // mapper.runBag("/mnt/data/TERRINet/scenes/scene0001_00/scene0001_00_depth.bag",
    //               "",
    //               "/mnt/data/TERRINet/scenes/scene0001_00/scene0001_00.map");
    // mapper.runBag("/mnt/data/TERRINet/scenes/scene0001_01/scene0001_01_depth.bag",
    //               "/mnt/data/TERRINet/scenes/scene0001_01/scene0001_01_map.bag",
    //               "");

    // mapper.runBag("/mnt/data/TERRINet/scenes/scene0002_01/scene0002_01_depth.bag",
    //               "",
    //               "/mnt/data/TERRINet/scenes/scene0002_01/scene0002_01.map");
    // mapper.runBag("/mnt/data/TERRINet/scenes/scene0002_02/scene0002_02_depth.bag",
    //               "/mnt/data/TERRINet/scenes/scene0002_02/scene0002_02_map.bag",
    //               "");

    return EXIT_SUCCESS;
}