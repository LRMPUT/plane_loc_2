//
// Created by janw on 16.12.2019.
//

#include "ros/ros.h"

#include <Mapper.hpp>

using namespace std;

int main(int argc, char** argv){
    ros::init( argc, argv, "mapper" );

    Mapper mapper;
    mapper.run();

    return EXIT_SUCCESS;
}