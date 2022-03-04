//
// Created by janw on 16.12.2019.
//

#ifndef PLANE_LOC_MAPPER_HPP
#define PLANE_LOC_MAPPER_HPP

#include <queue>
#include <string>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Map.hpp>

// message types
#include <plane_loc/Serialized.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>

class Mapper {
public:
    Mapper();

    void run();

    void runBag(const std::string &bagFilename,
                const std::string &outputMapBag,
                const std::string &outputMapFilename);
private:
    void objsCallback(const plane_loc::Serialized::ConstPtr &nobjsMsg);

    void process(int &curFrameIdx,
                 const plane_loc::Serialized::ConstPtr &objsMsg,
                 const tf2_ros::Buffer &curTfBuffer,
                 plane_loc::Serialized &mapSerMsg,
                 sensor_msgs::PointCloud2 &pcMsg,
                 pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                 int v1 = 0,
                 int v2 = 0);

    const std::string TAG = "[mapper] ";

    ros::NodeHandle nh;

    cv::FileStorage settings;

    std::string objsTopic;

    Eigen::Matrix3d cameraParams;

    int nrows, ncols;

    // tf2 listeners
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;

    ros::Subscriber subObjs;

    ros::Publisher pubMap;

    ros::Publisher pubPointCloud;

    std::queue<plane_loc::Serialized::ConstPtr> objsMsgs;

    // bool newObjs;

    int accFrames;

    int mergeMapFrames;

    double accDuration;

    Map map;

    Eigen::Affine3d sensorInOdomAtStart;

    std_msgs::Header headerAtStart;
};


#endif //PLANE_LOC_MAPPER_HPP
