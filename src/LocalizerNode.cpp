//
// Created by janw on 16.12.2019.
//

#include "ros/ros.h"

#include <Localizer.hpp>

using namespace std;

int main(int argc, char** argv){
    ros::init( argc, argv, "localizer" );

    Localizer localizer;
    localizer.run();

    return EXIT_SUCCESS;
}