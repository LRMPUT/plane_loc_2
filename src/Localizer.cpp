//
// Created by janw on 16.12.2019.
//

// STL
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>

// ROS
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.h>
// fix of LZ4 stream: https://github.com/ethz-asl/lidar_align/issues/16
#define LZ4_stream_t LZ4_stream_t_deprecated
#define LZ4_resetStream LZ4_resetStream_deprecated
#define LZ4_createStream LZ4_createStream_deprecated
#define LZ4_freeStream LZ4_freeStream_deprecated
#define LZ4_loadDict LZ4_loadDict_deprecated
#define LZ4_compress_fast_continue LZ4_compress_fast_continue_deprecated
#define LZ4_saveDict LZ4_saveDict_deprecated
#define LZ4_streamDecode_t LZ4_streamDecode_t_deprecated
#define LZ4_compress_continue LZ4_compress_continue_deprecated
#define LZ4_compress_limitedOutput_continue LZ4_compress_limitedOutput_continue_deprecated
#define LZ4_createStreamDecode LZ4_createStreamDecode_deprecated
#define LZ4_freeStreamDecode LZ4_freeStreamDecode_deprecated
#define LZ4_setStreamDecode LZ4_setStreamDecode_deprecated
#define LZ4_decompress_safe_continue LZ4_decompress_safe_continue_deprecated
#define LZ4_decompress_fast_continue LZ4_decompress_fast_continue_deprecated
#include <rosbag/bag.h>
#undef LZ4_stream_t
#undef LZ4_resetStream
#undef LZ4_createStream
#undef LZ4_freeStream
#undef LZ4_loadDict
#undef LZ4_compress_fast_continue
#undef LZ4_saveDict
#undef LZ4_streamDecode_t
#undef LZ4_compress_continue
#undef LZ4_compress_limitedOutput_continue
#undef LZ4_createStreamDecode
#undef LZ4_freeStreamDecode
#undef LZ4_setStreamDecode
#undef LZ4_decompress_safe_continue
#undef LZ4_decompress_fast_continue
#include <rosbag/view.h>

// Eigen
#include <Eigen/Dense>

// boost
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>

// local
#include <Localizer.hpp>
#include <MiscRos.hpp>
#include <Matching.hpp>
#include <Misc.hpp>

// message types
#include <plane_loc/Serialized.h>
#include <geometry_msgs/TransformStamped.h>


Localizer::Localizer()  :
        nh("~"),
        tfListener(tfBuffer),
        newMap(false)
{
    std::string settingsFilename = readParameter<std::string>(nh, "settings", "settings.yaml", TAG);
    mapTopic = readParameter<std::string>(nh, "map_topic", "/mapper/map", TAG);
    scoreThresh = readParameter(nh, "score_thresh", 1.0, TAG);
    scoreDiffThresh = readParameter(nh, "score_diff_thresh", 0.0, TAG);
    distinctThresh = readParameter(nh, "distinct_thresh", 6, TAG);
    matchedRatioThresh = readParameter(nh, "matched_ratio_thresh", 0.75, TAG);

    transDiffThresh = readParameter(nh, "trans_diff_thresh", 0.5, TAG);
    angDiffThresh = readParameter(nh, "ang_diff_thresh", 15.0, TAG);

    settings.open(settingsFilename, cv::FileStorage::READ);

    globalMap = Map(settings);

    subMap = nh.subscribe(mapTopic, 10, &Localizer::mapCallback, this);

    pubPose = nh.advertise<geometry_msgs::TransformStamped>("pose", 10);
}

void Localizer::run() {
    static constexpr int loopFreq = 200;

    ros::Rate loopRate(loopFreq);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("mapper"));
    int v1 = 0;
    int v2 = 0;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->addCoordinateSystem();

    corrCnt = 0;
    incorrCnt = 0;
    unkCnt = 0;
    allCnt = 0;

    sumError = 0.0;
    sumErrorLin = 0.0;
    sumErrorAng = 0.0;

    while (ros::ok()) {
        if(mapMsg){
            process(mapMsg,
                    tfBuffer,
                    viewer,
                    v1, v2);

            mapMsg.reset();
        }

        ros::spinOnce();
        loopRate.sleep();
    }
}

void Localizer::runBag(const std::string &bagFilename) {
    rosbag::Bag inputBag;
    inputBag.open(bagFilename);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("mapper"));
    int v1 = 0;
    int v2 = 0;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->addCoordinateSystem();

    corrCnt = 0;
    incorrCnt = 0;
    unkCnt = 0;
    allCnt = 0;

    sumError = 0.0;
    sumErrorLin = 0.0;
    sumErrorAng = 0.0;
    maxErrorLin = 0.0;
    maxErrorAng = 0.0;

    // populate tf buffer and find camera info
    tf2_ros::Buffer tfBufferBag(ros::Duration().fromSec(60*60));
    for(rosbag::MessageInstance const m: rosbag::View(inputBag)) {
        if (m.getTopic() == "/tf_static" || m.getTopic() == "/tf") {
            bool isStatic = false;
            if (m.getTopic() == "/tf_static"){
                isStatic = true;
            }
            tf2_msgs::TFMessage::ConstPtr msg = m.instantiate<tf2_msgs::TFMessage>();
            if (msg != nullptr) {
                for (auto trans : msg->transforms) {
                    tfBufferBag.setTransform(trans, "rosbag", isStatic);
                }
            }
        }
    }

    for (rosbag::MessageInstance const m: rosbag::View(inputBag)) {
        if (m.getTopic() == mapTopic) {
            plane_loc::Serialized::ConstPtr curMapMsg = m.instantiate<plane_loc::Serialized>();
            if (curMapMsg != nullptr) {
                ROS_INFO_STREAM(TAG << "processing next map");

                process(curMapMsg,
                        tfBufferBag,
                        viewer,
                        v1, v2);
            }
        }
    }
}

void Localizer::process(const plane_loc::Serialized::ConstPtr &curMapMsg,
                        const tf2_ros::Buffer &curTfBuffer,
                        pcl::visualization::PCLVisualizer::Ptr viewer,
                        int v1,
                        int v2)
{
    ROS_INFO_STREAM(TAG << "processing next map");

    Eigen::Affine3d sensorInMap;
    Map curMap;
    {
        std::stringstream buffer;
        buffer.write((const char*)curMapMsg->data.data(), curMapMsg->data.size());
        boost::archive::binary_iarchive ia(buffer);
        ia >> curMap;
    }
    ROS_INFO_STREAM(TAG << "curMap.size() = " << curMap.size());
    {
        geometry_msgs::TransformStamped sensorInMapTS;
        try {
            sensorInMapTS = curTfBuffer.lookupTransform("map", curMapMsg->header.frame_id, curMapMsg->header.stamp);
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN_STREAM(TAG << ex.what());
            // newMap = false;
            return;
        }
        // Getting the affine transformation laser->base_link
        sensorInMap = tf2::transformToEigen(sensorInMapTS);
    }

    // if((sensorInMap.translation() - Eigen::Vector3d(-11.7541, -11.5843, 0.053426)).norm() < 0.1){
    if(true){
        bool stopFlag = false;

        vectorVector7d planesTrans;
        std::vector<double> planesTransScores;
        std::vector<double> planesMatchedRatio;
        std::vector<int> planesTransDistinct;
        std::vector<Matching::ValidTransform> transforms;
        Matching::MatchType matchType = Matching::MatchType::Unknown;

        ROS_INFO_STREAM(TAG << "sensorInMap R = \n" << sensorInMap.rotation());
        ROS_INFO_STREAM(TAG << "sensorInMap t = \n" << sensorInMap.translation().transpose());

        {
            // std::ofstream evalFile("eval.log", std::ios_base::app);
            std::ofstream evalFile;
            if (evalFile.is_open()) {
                Vector7d sensorInMapVec = Misc::toVector(sensorInMap.matrix());
                evalFile << sensorInMapVec(0) << " "
                         << sensorInMapVec(1) << " "
                         << sensorInMapVec(2) << " "
                         << sensorInMapVec(3) << " "
                         << sensorInMapVec(4) << " "
                         << sensorInMapVec(5) << " "
                         << sensorInMapVec(6) << endl;
            }
        }

        static auto processDur = std::chrono::nanoseconds(0);
        static int nProcessDur = 0;

        // pcl::visualization::PCLVisualizer::Ptr viewerMatch(new pcl::visualization::PCLVisualizer("matching"));
        // int vMatch1 = 0;
        // int vMatch2 = 0;
        // viewerMatch->createViewPort(0.0, 0.0, 0.5, 1.0, vMatch1);
        // viewerMatch->createViewPort(0.5, 0.0, 1.0, 1.0, vMatch2);
        // viewerMatch->addCoordinateSystem();

        auto startTs = std::chrono::steady_clock::now();
        // matchType = Matching::matchLocalToGlobal(settings,
        //                                       globalMap,
        //                                       curMap,
        //                                       planesTrans,
        //                                       planesTransScores,
        //                                          planesMatchedRatio,
        //                                       planesTransDistinct,
        //                                          transforms,
        //                                          viewerMatch,
        //                                          vMatch1,
        //                                          vMatch2,
        //                                          sensorInMap.matrix());
        matchType = Matching::matchLocalToGlobal(settings,
                                                 globalMap,
                                                 curMap,
                                                 planesTrans,
                                                 planesTransScores,
                                                 planesMatchedRatio,
                                                 planesTransDistinct,
                                                 transforms,
                                                 viewer,
                                                 v1,
                                                 v2,
                                                 sensorInMap.matrix());
        auto endTs = std::chrono::steady_clock::now();
        processDur += endTs - startTs;
        nProcessDur += 1;
        if (nProcessDur > 0) {
            ROS_INFO_STREAM(TAG << "Mean processing time: " << (float)processDur.count() / nProcessDur / 1.0e6f << " ms");
        }

        std::ofstream resFile("results.txt", std::ios_base::app);

        {
            vectorVector7d newPlanesTrans;
            std::vector<double> newPlanesTransScores;
            std::vector<double> newPlanesMatchedRatio;
            std::vector<int> newPlanesTransDistinct;
            for (int t = 0; t < planesTrans.size(); ++t) {
                if (planesMatchedRatio[t] >= matchedRatioThresh && planesTransDistinct[t] >= distinctThresh) {
                    newPlanesTrans.push_back(planesTrans[t]);
                    newPlanesTransScores.push_back(planesTransScores[t]);
                    newPlanesMatchedRatio.push_back(planesMatchedRatio[t]);
                    newPlanesTransDistinct.push_back(planesTransDistinct[t]);
                }
            }

            planesTrans.swap(newPlanesTrans);
            planesTransScores.swap(newPlanesTransScores);
            planesMatchedRatio.swap(newPlanesMatchedRatio);
            planesTransDistinct.swap(newPlanesTransDistinct);

            if (planesTrans.size() == 0) {
                matchType = Matching::MatchType::Unknown;
            }
        }

        bool isUnamb = true;
        if( matchType == Matching::MatchType::Ok){
            ROS_INFO_STREAM(TAG << "match ok");

            if(planesTransScores.front() < scoreThresh){
                isUnamb = false;
            }
            // if(planesTransScores.size() > 1){
            //     if(std::abs(planesTransScores[0] - planesTransScores[1]) < scoreDiffThresh){
            //         isUnamb = false;
            //     }
            // }
            if(planesMatchedRatio.front() < matchedRatioThresh){
                isUnamb = false;
            }
            if(planesTransDistinct.front() < distinctThresh){
                isUnamb = false;
            }
        }
        else{
            ROS_INFO_STREAM(TAG << "match unknown");
        }
        if(!planesTransScores.empty()) {
            ROS_INFO_STREAM(TAG << "planesTransScores.front() = " << planesTransScores.front());
            ROS_INFO_STREAM(TAG << "planesMatchedRatio.front() = " << planesMatchedRatio.front());
            ROS_INFO_STREAM(TAG << "planesTransDistinct.front() = " << planesTransDistinct.front());
        }

        if( matchType == Matching::MatchType::Ok && isUnamb){
            Eigen::Affine3d curPoseEst(Misc::toMatrix(planesTrans.front()));

            double curLogError;
            double curLinError;
            double curAngError;
            compErrors(curPoseEst, sensorInMap, curLogError, curLinError, curAngError);

            if(curLinError < transDiffThresh && curAngError * 180.0 / M_PI < angDiffThresh){
                ++corrCnt;
            }
            else{
                ++incorrCnt;

                ROS_WARN_STREAM(TAG << "incorrect!");
            }

            sumError += curLogError;
            sumErrorLin += curLinError;
            sumErrorAng += curAngError;
            maxErrorLin = max(curLinError, maxErrorLin);
            maxErrorAng = max(curAngError, maxErrorAng);

            geometry_msgs::TransformStamped poseMsg;
            tf2::eigenToTransform(curPoseEst);
            poseMsg.header = curMapMsg->header;

            pubPose.publish(poseMsg);

            ROS_INFO_STREAM(TAG << "curPoseEst = \n" << curPoseEst.matrix());
            ROS_INFO_STREAM(TAG << "cur error = " << curLogError);
            ROS_INFO_STREAM(TAG << "cur error lin = " << curLinError);
            ROS_INFO_STREAM(TAG << "cur error ang = " << curAngError * 180.0 / M_PI);

            // if (viewer) {
            //     viewer->removeAllPointClouds();
            //     viewer->removeAllShapes();
            //     viewer->removeAllCoordinateSystems();
            //
            //     globalMap.display(viewer, v1, v2, 1.0, 0.0, 0.0, false);
            //
            //     Map localMapTrans(curMap);
            //     localMapTrans.transform(planesTrans.front());
            //     localMapTrans.display(viewer, v1, v2, 0.0, 1.0, 0.0, false);
            //
            //     viewer->addCoordinateSystem();
            //     viewer->addCoordinateSystem(1.0, sensorInMap.cast<float>(), "camera");
            //     viewer->addCoordinateSystem(1.0, curPoseEst.cast<float>(), "camera_est");
            //
            //
            //     static bool init = false;
            //     if (!init) {
            //         viewer->resetStoppedFlag();
            //         viewer->initCameraParameters();
            //         viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, -1.0, 0.0);
            //         // while (!viewer->wasStopped()) {
            //         viewer->spinOnce(100);
            //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
            //         // }
            //         init = true;
            //     } else {
            //         viewer->spinOnce(100);
            //         // viewer->resetStoppedFlag();
            //         // while (!viewer->wasStopped()) {
            //         //     viewer->spinOnce(100);
            //         //     std::this_thread::sleep_for(std::chrono::milliseconds(50));
            //         // }
            //     }
            // }

            resFile << std::fixed << curMapMsg->header.stamp.toSec() << " "
                    << planesTrans.front()(0) << " "
                    << planesTrans.front()(1) << " "
                    << planesTrans.front()(2) << " "
                    << planesTrans.front()(3) << " "
                    << planesTrans.front()(4) << " "
                    << planesTrans.front()(5) << " "
                    << planesTrans.front()(6) << endl;
        }
        else{
            ++unkCnt;

            resFile << std::fixed << curMapMsg->header.stamp.toSec() << " 0 0 0 0 0 0 0" << endl;
        }

        ++allCnt;

        {
            int recCnt = corrCnt + incorrCnt;
            ROS_INFO_STREAM(TAG << "corrCnt = " << corrCnt << ", ratio = " << (float)corrCnt / (corrCnt + incorrCnt + unkCnt));
            ROS_INFO_STREAM(TAG << "incorrCnt = " << incorrCnt << ", ratio = " << (float)incorrCnt / (corrCnt + incorrCnt + unkCnt));
            ROS_INFO_STREAM(TAG << "unkCnt = " << unkCnt << ", ratio = " << (float)unkCnt / (corrCnt + incorrCnt + unkCnt));
            if(recCnt > 0){
                ROS_INFO_STREAM(TAG << "mean error = " << sumError / recCnt);
                ROS_INFO_STREAM(TAG << "mean error lin = " << sumErrorLin / recCnt);
                ROS_INFO_STREAM(TAG << "mean error ang = " << sumErrorAng * 180.0 / M_PI / recCnt);
                ROS_INFO_STREAM(TAG << "max error lin = " << maxErrorLin);
                ROS_INFO_STREAM(TAG << "max error ang = " << maxErrorAng * 180.0 / M_PI);
            }
        }
        if( matchType == Matching::MatchType::Ok && isUnamb){
            ROS_INFO_STREAM(TAG << "LOCALIZED");
        }
        else {
            ROS_INFO_STREAM(TAG << "UNKNOWN");
        }

        if (viewer) {
            // viewer->removeAllPointClouds();
            viewer->removeAllShapes();
            viewer->removeAllCoordinateSystems();

            Map localMapTrans(curMap);
            localMapTrans.transform(Misc::toVector(sensorInMap.matrix()));
            localMapTrans.display(viewer, v2, -1);

            viewer->addCoordinateSystem();
            viewer->addCoordinateSystem(1.0, sensorInMap.cast<float>(), "camera");

            static bool init = false;
            if (!init) {
                globalMap.display(viewer, v1, -1);

                viewer->resetStoppedFlag();
                viewer->initCameraParameters();
                viewer->setCameraPosition(0.0, 0.0, 6.0, 0.0, -1.0, 0.0);

                // while (!viewer->wasStopped()) {
                //     viewer->spinOnce(100);
                //     std::this_thread::sleep_for(std::chrono::milliseconds(50));
                // }

                viewer->spinOnce(100);

                init = true;
            } else {
                viewer->spinOnce(100);

                if (stopFlag) {
                    viewer->resetStoppedFlag();
                    while (!viewer->wasStopped()) {
                        viewer->spinOnce(100);
                        // std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }
            }

            localMapTrans.cleanDisplay(viewer, v2, -1);
        }
    }
}

void Localizer::mapCallback(const plane_loc::Serialized::ConstPtr &nmapMsg) {
    mapMsg = nmapMsg;
}

void Localizer::compErrors(const Eigen::Affine3d &refPose,
                           const Eigen::Affine3d &estPose,
                           double &logError,
                           double &linError,
                           double &angError)
{
    Eigen::Affine3d diff = estPose.inverse() * refPose;
    Vector6d diffLog = Misc::logMap(diff.matrix());

    logError = diffLog.norm();
    linError = diff.matrix().block<3, 1>(0, 3).norm();
    angError = diffLog.tail<3>().norm();
}
