//
// Created by janw on 16.12.2019.
//

#include "ros/ros.h"

#include <Localizer.hpp>

using namespace std;

int main(int argc, char** argv){
    ros::init( argc, argv, "localizer" );

    Localizer localizer;
    // localizer.runBag("/mnt/data/TERRINet/scenes/scene0000_02/scene0000_02_map.bag");
    localizer.runBag("/mnt/data/TERRINet/scenes/scene0001_01/scene0001_01_map.bag");
    // localizer.runBag("/mnt/data/TERRINet/scenes/scene0002_02/scene0002_02_map.bag");

    return EXIT_SUCCESS;
}