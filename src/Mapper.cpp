//
// Created by janw on 16.12.2019.
//

// STL
#include <string>
#include <fstream>
#include <chrono>
#include <thread>

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
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// local
#include <Types.hpp>
#include <Mapper.hpp>
#include <MiscRos.hpp>
#include <Misc.hpp>

// message types
#include <plane_loc/Serialized.h>
#include <sensor_msgs/PointCloud2.h>

Mapper::Mapper()  :
        nh("~"),
        tfListener(tfBuffer)
{
    std::string settingsFilename = readParameter<std::string>(nh, "settings", "settings.yaml", TAG);
    objsTopic = readParameter<std::string>(nh, "objs_topic", "/detector/objs", TAG);
    accFrames = readParameter(nh, "acc_frames", 50, TAG);
    mergeMapFrames = readParameter(nh, "merge_map_frames", 50, TAG);
    accDuration = readParameter(nh, "acc_duration", 2.0, TAG);
    cout << "accFrames = " << accFrames << endl;
    cout << "mergeMapFrames = " << mergeMapFrames << endl;
    cout << "accDuration = " << accDuration << endl;

    settings.open(settingsFilename, cv::FileStorage::READ);
    cv::Mat cameraParamsMat;
    settings["planeSlam"]["cameraMatrix"] >> cameraParamsMat;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            cameraParams(i, j) = cameraParamsMat.at<float>(i, j);
        }
    }
    settings["planeSlam"]["nrows"] >> nrows;
    settings["planeSlam"]["ncols"] >> ncols;
    cout << "cameraParams = " << endl << cameraParams << endl;
    cout << "camera res = (" << nrows << ", " << ncols << ")" << endl;


    subObjs = nh.subscribe(objsTopic, 1000, &Mapper::objsCallback, this, ros::TransportHints().tcpNoDelay(true));

    pubMap = nh.advertise<plane_loc::Serialized>("map", 10);
    pubPointCloud = nh.advertise<sensor_msgs::PointCloud2>("point_cloud_lab", 10);
}

void Mapper::run() {
    static constexpr int loopFreq = 200;

    ros::Rate loopRate(loopFreq);
    int curFrameIdx = 0;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("mapper"));
    int v1 = 0;
    int v2 = 0;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->addCoordinateSystem();

    while (ros::ok()) {
        while(!objsMsgs.empty()){
            ROS_INFO_STREAM(TAG << "processing next frame, curFrameIdx = " << curFrameIdx);

            plane_loc::Serialized::ConstPtr curObjsMsg = objsMsgs.front();
            objsMsgs.pop();

            plane_loc::Serialized mapSerMsg;
            sensor_msgs::PointCloud2 pcMsg;

            process(curFrameIdx,
                    curObjsMsg,
                    tfBuffer,
                    mapSerMsg,
                    pcMsg,
                    viewer,
                    v1, v2);
        }

        viewer->spinOnce(20);

        ros::spinOnce();
        loopRate.sleep();
    }
}

void Mapper::runBag(const std::string &bagFilename,
                    const std::string &outputMapBag,
                    const std::string &outputMapFilename) {
    rosbag::Bag inputBag;
    inputBag.open(bagFilename);

    int curFrameIdx = 0;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("mapper"));
    int v1 = 0;
    int v2 = 0;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    // viewer->addCoordinateSystem();

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

    rosbag::Bag outputBag;
    if (!outputMapBag.empty()) {
        outputBag.open(outputMapBag, rosbag::bagmode::Write);
        ROS_INFO_STREAM(TAG << "outputBag.isOpen() = " << outputBag.isOpen());
    }

    for (rosbag::MessageInstance const m: rosbag::View(inputBag)) {
        if (m.getTopic() == "/tf_static" || m.getTopic() == "/tf") {
            if (outputBag.isOpen()) {
                outputBag.write(m.getTopic(), m.getTime(), m);
            }
        }
        if (m.getTopic() == objsTopic) {
            plane_loc::Serialized::ConstPtr curObjsMsg = m.instantiate<plane_loc::Serialized>();
            if (curObjsMsg != nullptr) {
                ROS_INFO_STREAM(TAG << "processing next frame, curFrameIdx = " << curFrameIdx);

                plane_loc::Serialized mapSerMsg;
                mapSerMsg.header.stamp = ros::Time(0.0);
                sensor_msgs::PointCloud2 pcMsg;
                pcMsg.header.stamp = ros::Time(0.0);

                process(curFrameIdx,
                        curObjsMsg,
                        tfBufferBag,
                        mapSerMsg,
                        pcMsg,
                        viewer,
                        v1, v2);

                if (outputBag.isOpen() && mapSerMsg.header.stamp.toSec() > 0.0 && pcMsg.header.stamp.toSec() > 0.0) {
                    // cout << "mapSerMsg.header.stamp = " << mapSerMsg.header.stamp.toSec() << endl;
                    // cout << "pcMsg.header.stamp = " << pcMsg.header.stamp.toSec() << endl;
                    outputBag.write(pubMap.getTopic(), mapSerMsg.header.stamp, mapSerMsg);
                    outputBag.write(pubPointCloud.getTopic(), pcMsg.header.stamp, pcMsg);
                }
            }
        }
    }

    if (!outputMapFilename.empty()) {
        ROS_INFO_STREAM(TAG << "saving map to file " << outputMapFilename);
        std::ofstream ofs(outputMapFilename.c_str());
        if (ofs.is_open()) {
            boost::archive::binary_oarchive oa(ofs);
            oa << map;
            ROS_INFO_STREAM(TAG << "map saved successfully");
        } else {
            ROS_ERROR_STREAM(TAG << "could not open file " << outputMapFilename);
        }
    }

    map.exportPointCloud("map.ply");
}

void Mapper::objsCallback(const plane_loc::Serialized::ConstPtr &nobjsMsg) {
    objsMsgs.push(nobjsMsg);
    // newObjs = true;
}

void Mapper::process(int &curFrameIdx,
                     const plane_loc::Serialized::ConstPtr &objsMsg,
                     const tf2_ros::Buffer &curTfBuffer,
                     plane_loc::Serialized &mapSerMsg,
                     sensor_msgs::PointCloud2 &pcMsg,
                     pcl::visualization::PCLVisualizer::Ptr viewer,
                     int v1,
                     int v2) {
    Eigen::Affine3d sensorInOdom;
    std::vector<ObjInstanceView::Ptr> curObjInstances;
    {
        std::stringstream buffer;
        buffer.write((const char*)objsMsg->data.data(), objsMsg->data.size());
        boost::archive::binary_iarchive ia(buffer);
        ia >> curObjInstances;
    }
    ROS_INFO_STREAM(TAG << "curObjInstances.size() = " << curObjInstances.size());
    {
        geometry_msgs::TransformStamped sensorInOdomTS;
        try {
            sensorInOdomTS = curTfBuffer.lookupTransform("odom", objsMsg->header.frame_id, objsMsg->header.stamp);
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN_STREAM(TAG << ex.what());
            // newObjs = false;
            return;
        }
        // Getting the affine transformation laser->base_link
        sensorInOdom = tf2::transformToEigen(sensorInOdomTS);
    }

    // if current frame starts accumulation
    if (curFrameIdx == 0) {
    // if (curFrameIdx  % accFrames == 0) {
        cout << endl << "starting new accumulation" << endl << endl;

        map = Map();
        // sensorInOdomAtStart = sensorInOdom;
        sensorInOdomAtStart = Eigen::Affine3d::Identity();
        headerAtStart = objsMsg->header;
    }

    if (curFrameIdx  % accFrames == 0) {
        uint64_t tsThresh = (objsMsg->header.stamp - ros::Duration(accDuration)).toNSec();
        map.removeViews(tsThresh);

        cout << endl << "merging curObjInstances" << endl << endl;

        Eigen::Affine3d sensorInStart = sensorInOdomAtStart.inverse() * sensorInOdom;

        Vector7d sensorInStartVec = Misc::toVector(sensorInStart.matrix());

        cout << "sensorInStartLog = " << sensorInStartVec.transpose() << endl;

        for (const ObjInstanceView::Ptr &curObj : curObjInstances) {
            // cout << "Before:" << endl;
            // curObj.getHull().check();
            curObj->transform(sensorInStartVec);
            // cout << "After:" << endl;
            // curObj.getHull().check();
        }

        map.mergeNewObjInstanceViews(curObjInstances,
                                     sensorInStartVec,
                                     cameraParams,
                                     nrows,
                                     ncols/*,
                                     viewer,
                                     v1,
                                     v2*/);

        if (curFrameIdx % mergeMapFrames == 0)
        {
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();

            cout << "Merging map" << endl;
            map.mergeMapObjInstances(/*viewer,
                                     v1,
                                     v2*/);
        }

        // if(viewer) {
        //     viewer->removeAllPointClouds();
        //     viewer->removeAllShapes();
        //     viewer->removeAllCoordinateSystems();
        //
        //     map.display(viewer, v1, v2);
        //
        //     viewer->addCoordinateSystem();
        //     viewer->addCoordinateSystem(1.0, sensorInStart.cast<float>(), "camera");
        //
        //
        //     static bool init = false;
        //     if (!init) {
        //         viewer->resetStoppedFlag();
        //         viewer->initCameraParameters();
        //         viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, -1.0, 0.0);
        //         while (!viewer->wasStopped()) {
        //             viewer->spinOnce(100);
        //             std::this_thread::sleep_for(std::chrono::milliseconds(50));
        //         }
        //         init = true;
        //     }
        //     else {
        //         viewer->spinOnce(100);
        //     }

        {
            Map mapPub(map);
            // mapPub.removeObjsEolThresh(6);
            mapPub.transform(Misc::toVector(sensorInOdom.inverse().matrix()));

            cout << "publishing map" << endl;
            {
                std::stringstream bufferSs;
                boost::archive::binary_oarchive oa(bufferSs);
                oa << mapPub;
                std::string buffer = bufferSs.str();

                // plane_loc::Serialized mapSerMsg;
                mapSerMsg.header = objsMsg->header;
                // mapSerMsg.header.frame_id = "odom";
                // mapSerMsg.header.seq = objsMsg->header.seq;
                mapSerMsg.type = "Map";
                mapSerMsg.data.insert(mapSerMsg.data.begin(), buffer.begin(), buffer.end());

                pubMap.publish(mapSerMsg);
            }

            // cout << "publishing point cloud" << endl;
            // {
            //     pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointCloudLab = mapPub.getLabeledColorPointCloud();
            //
            //     // sensor_msgs::PointCloud2 pcMsg;
            //     pcl::toROSMsg(*pointCloudLab, pcMsg);
            //     pcMsg.header = objsMsg->header;
            //     // pcMsg.header.frame_id = "odom";
            //     // pcMsg.header.seq = objsMsg->header.seq;
            //
            //     pubPointCloud.publish(pcMsg);
            // }

            cout << "visualizing" << endl;
            if (viewer) {
                viewer->removeAllPointClouds();
                viewer->removeAllShapes();
                viewer->removeAllCoordinateSystems();

                // mapPub.display(viewer, v1, v2);

                viewer->addCoordinateSystem();
                viewer->addCoordinateSystem(1.0, sensorInOdom.inverse().cast<float>(), "camera");
                // viewer->addCoordinateSystem(1.0, sensorInOdom.cast<float>(), "camera");


                static bool init = false;
                if (!init) {
                    viewer->resetStoppedFlag();
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, -1.0, 0.0);
                    while (!viewer->wasStopped()) {
                        viewer->spinOnce(100);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                    init = true;
                }
                else {
                    viewer->spinOnce(100);
                    // viewer->resetStoppedFlag();
                    // while (!viewer->wasStopped()) {
                    //     viewer->spinOnce(100);
                    //     std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    // }
                }
            }
            cout << "end visualizing" << endl;
        }
    }

    ++curFrameIdx;
}
