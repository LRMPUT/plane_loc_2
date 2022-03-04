//
// Created by janw on 16.12.2019.
//

#ifndef PLANE_LOC_LOCALIZER_HPP
#define PLANE_LOC_LOCALIZER_HPP

#include <string>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Map.hpp>

// message types
#include <plane_loc/Serialized.h>

class Localizer {
public:
    Localizer();

    void run();

    void runBag(const std::string &bagFilename);
private:
    void process(const plane_loc::Serialized::ConstPtr &mapMsg,
                 const tf2_ros::Buffer &curTfBuffer,
                 pcl::visualization::PCLVisualizer::Ptr viewer,
                 int v1,
                 int v2);

    void mapCallback(const plane_loc::Serialized::ConstPtr &nmapMsg);

    void compErrors(const Eigen::Affine3d &refPose,
                    const Eigen::Affine3d &estPose,
                    double &logError,
                    double &linError,
                    double &angError);

    const std::string TAG = "[localizer] ";

    ros::NodeHandle nh;

    cv::FileStorage settings;

    // tf2 listeners
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;

    ros::Subscriber subMap;

    std::string mapTopic;

    ros::Publisher pubPose;

    plane_loc::Serialized::ConstPtr mapMsg;

    bool newMap;

    Map globalMap;

    double scoreThresh;

    double scoreDiffThresh;

    double fitThresh;

    double matchedRatioThresh;

    double distinctThresh;

    double transDiffThresh;

    double angDiffThresh;

    int corrCnt;
    int incorrCnt;
    int unkCnt;
    int allCnt;

    double sumError;
    double sumErrorLin;
    double sumErrorAng;
    double maxErrorLin;
    double maxErrorAng;
};


#endif //PLANE_LOC_LOCALIZER_HPP
