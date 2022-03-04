//
// Created by janw on 17.12.2019.
//

#ifndef PLANE_LOC_MISCROS_HPP
#define PLANE_LOC_MISCROS_HPP

/**
 * Method used to parse parameter provided in the launch file
 * @param name Name of the string parameter to read
 * @param defaultValue Default value of a parameter if not found in the file
 * @return Read value of the parameter
 */
template<class T>
T readParameter(ros::NodeHandle &nh, std::string name, T defaultValue, std::string TAG = "") {
    T value;
    if (nh.getParam(name.c_str(), value))
        ROS_INFO_STREAM(TAG << name << " : " << value);
    else {
        ROS_ERROR_STREAM(TAG << "No value for " << name << " set -- default equal to " << value);
        value = defaultValue;
    }
    return value;
}

#endif //PLANE_LOC_MISCROS_HPP
