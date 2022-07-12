/*
    Copyright (c) 2017 Mobile Robots Laboratory at Poznan University of Technology:
    -Jan Wietrzykowski name.surname [at] put.poznan.pl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <iostream>
#include <chrono>
#include <thread>
#include <string>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>

#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include "Map.hpp"
#include "Misc.hpp"
// #include "PlaneSegmentation.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"
#include "Serialization.hpp"
#include "UnionFind.h"

using namespace std;

Map::Map()
    : nnsUptodate(false)
{
    settings.eolObjInstInit = 4;
    settings.eolObjInstIncr = 2;
    settings.eolObjInstDecr = 1;
    settings.eolObjInstThresh = 10;
    // settings.eolPendingInit = 4;
    // settings.eolPendingIncr = 2;
    // settings.eolPendingDecr = 1;
    // settings.eolPendingThresh = 6;
}

Map::Map(const cv::FileStorage& fs)
    : nnsUptodate(false)
{
    settings.eolObjInstInit = 4;
    settings.eolObjInstIncr = 2;
    settings.eolObjInstDecr = 1;
    settings.eolObjInstThresh = 10;
    // settings.eolPendingInit = 4;
    // settings.eolPendingIncr = 2;
    // settings.eolPendingDecr = 1;
    // settings.eolPendingThresh = 6;
    
	if((int)fs["map"]["readFromFile"]){
		// pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("map 3D Viewer"));
        //
		// int v1 = 0;
		// int v2 = 0;
		// viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		// viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		// viewer->addCoordinateSystem();

		vector<cv::String> mapFilepaths;
        fs["map"]["mapFiles"] >> mapFilepaths;

        static constexpr int idShift = 10000000;
        for(int f = 0; f < mapFilepaths.size(); ++f) {
            Map curMap;
        
            std::ifstream ifs(mapFilepaths[f].c_str());
            if (!ifs.is_open()) {
                cout << "Could not open file: " << mapFilepaths[f].c_str() << endl;
            }
            boost::archive::binary_iarchive ia(ifs);
            ia >> curMap;

            curMap.shiftIds((f + 1)*idShift);
            for (const auto &planeIdObj : curMap) {
                planeIdToObj[planeIdObj.first] = planeIdObj.second;
            }
        }
        mergeMapObjInstances();

        cout << "object instances in map: " << planeIdToObj.size() << endl;

//         if(viewer) {
//             viewer->removeAllPointClouds(v1);
//             viewer->removeAllShapes(v1);
//             viewer->removeAllPointClouds(v2);
//             viewer->removeAllShapes(v2);
//
//             pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol = getColorPointCloud();
//             viewer->addPointCloud(pcCol, "cloud_color_map", v1);
//
//             pcl::PointCloud<pcl::PointXYZL>::Ptr pcLab = getLabeledPointCloud();
//             viewer->addPointCloud(pcLab, "cloud_labeled_map", v2);
//
//             viewer->resetStoppedFlag();
//             viewer->initCameraParameters();
//             viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
//             viewer->spinOnce(100);
// //            while (!viewer->wasStopped()) {
// //                viewer->spinOnce(100);
// //                std::this_thread::sleep_for(std::chrono::milliseconds(50));
// //            }
//         }

	}
}

Map::Map(const Map &other) {
    for (const auto &planeIdObj : other.planeIdToObj) {
        planeIdToObj[planeIdObj.first] = planeIdObj.second->copy();
    }
    descs = other.descs;
    planeIdViews = other.planeIdViews;
    nnsUptodate = other.nnsUptodate;
    settings = other.settings;

    if (descs.cols() > 0) {
        nns.reset(Nabo::NNSearchD::createKDTreeLinearHeap(descs));
    }
}

void Map::transform(const Vector7d &transform) {
    for (const auto &planeIdObj : planeIdToObj) {
        planeIdObj.second->transform(transform);
    }
}

void Map::createNewObj(const ObjInstanceView::Ptr &view) {
    int newPlaneId = view->getId();
    planeIdToObj[newPlaneId] = ObjInstance::Ptr(new ObjInstance(newPlaneId, ObjInstance::ObjType::Plane));
    planeIdToObj[newPlaneId]->addView(view);

    nnsUptodate = false;
}

// void Map::addObjs(std::vector<ObjInstanceView::Ptr>::iterator beg,
//                   std::vector<ObjInstanceView::Ptr>::iterator end) {
//     for(auto it = beg; it != end; ++it){
//         addObj(*it);
//     }
// }


void Map::mergeNewObjInstanceViews(const std::vector<ObjInstanceView::Ptr> &newObjInstanceViews,
                                   const Vector7d &pose,
                                   const Eigen::Matrix3d &cameraMatrix,
                                   int rows,
                                   int cols,
                                   pcl::visualization::PCLVisualizer::Ptr viewer,
                                   int viewPort1,
                                   int viewPort2)
{
    static constexpr double shadingLevel = 1.0/8;
    
    static const int cntThreshMerge = 500;
    
    chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();

    std::unordered_map<int, std::pair<int, ObjInstanceView::ConstPtr>> planeIdToCntView = getVisibleObjs(pose,
                                                                                               cameraMatrix,
                                                                                               rows, cols/*,
                                                                                               viewer,
                                                                                               viewPort1, viewPort2*/);

    if(viewer){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        
        {
            // int pl = 0;
            for (const auto &planeIdCntView : planeIdToCntView) {
                const int &planeId = planeIdCntView.first;
                const int &cnt = planeIdCntView.second.first;
                const ObjInstanceView &mapView = *planeIdCntView.second.second;
//                cout << "adding plane " << pl << endl;

                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = mapView.getPointCloud();
                
                viewer->addPointCloud(curPl, string("plane_ba_") + to_string(planeId), viewPort1);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_ba_") + to_string(planeId),
                                                         viewPort1);
                
                Eigen::Vector3d cent = mapView.getPlaneEstimator().getCentroid();
                viewer->addText3D("id: " + to_string(planeId) +
                                          ", cnt: " + to_string(cnt) +
                                          ", eol: " + to_string(planeIdToObj.at(planeId)->getId()),
                                  pcl::PointXYZ(cent(0), cent(1), cent(2)),
                                  0.05,
                                  1.0, 1.0, 1.0,
                                  string("plane_text_ba_") + to_string(planeId),
                                  viewPort1);
            }
        }
        {
            int npl = 0;
            for (const ObjInstanceView::Ptr &newView : newObjInstanceViews) {
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPl = newView->getPointCloud();
                
                viewer->addPointCloud(curPl, string("plane_nba_") + to_string(npl), viewPort2);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         shadingLevel,
                                                         string("plane_nba_") + to_string(npl),
                                                         viewPort2);
                
                ++npl;
            }
        }
        
    }
    
    // std::vector<ObjInstanceView::Ptr> addedObjs;
    //
    // UnionFind ufSets(objInstances.size());
    // std::map<int, int> idxToId;
    // {
    //     int pl = 0;
    //     for (auto it = objInstances.begin(); it != objInstances.end(); ++it, ++pl) {
    //         idxToId[pl] = (*it)->getId();
    //     }
    // }
    int npl = 0;
    for(const ObjInstanceView::Ptr &newView : newObjInstanceViews){
//        cout << "npl = " << npl << endl;
        
        if(viewer){
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     8.0/8,
                                                     string("plane_nba_") + to_string(npl),
                                                     viewPort2);
            
            
            // newObj.getHull().display(viewer, viewPort2);
            
        }
        
        vector<int> matches;
        // int pl = 0;
        for (const auto &planeIdCntView : planeIdToCntView) {
            const int &planeId = planeIdCntView.first;
            const int &cnt = planeIdCntView.second.first;
            const ObjInstanceView::ConstPtr &mapView = planeIdCntView.second.second;
    
            if(cnt > cntThreshMerge) {
                if (viewer) {
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                             7.0/8,
                                                             string("plane_ba_") + to_string(planeId),
                                                             viewPort1);


                    // mapObj.getHull().display(viewer, viewPort1);

                }
    
                if (mapView->isMatching(*newView/*,
                                 viewer,
                                 viewPort1,
                                 viewPort2*/)) {
                    // cout << "matching" << endl;
        
                    matches.push_back(planeId);
                }
                else {
                    // cout << "not matching" << endl;
                }
    
    
                if (viewer) {
                    viewer->resetStoppedFlag();

                    static bool cameraInit = false;

                    if (!cameraInit) {
                        viewer->initCameraParameters();
                        viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                        cameraInit = true;
                    }
                    while (!viewer->wasStopped()) {
                        viewer->spinOnce(100);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }

                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                             shadingLevel,
                                                             string("plane_ba_") + to_string(planeId),
                                                             viewPort1);


                    // mapObj.getHull().cleanDisplay(viewer, viewPort1);

                }
            }
        }

        // only matches that still exists, i.e. hasn't been merged into other objects
        std::vector<int> validMatches;
        for(const int &m : matches) {
            if (planeIdToObj.count(m) > 0) {
                validMatches.push_back(m);
            }
        }
        if(validMatches.size() == 0){
            // cout << "creating new obj from view " << newView->getId() << endl;
            createNewObj(newView);
        }
        else if(validMatches.size() > 0){
            // cout << "merging " << validMatches << endl;
            // cout << "and adding new view " << newView->getId() << endl;
            for (int p = 1; p < validMatches.size(); ++p) {
                // if (viewer) {
                //     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                //                                              8.0/8,
                //                                              string("plane_ba_") + to_string(matches[p]),
                //                                              viewPort1);
                //
                //
                //     // mapObj.getHull().display(viewer, viewPort1);
                //
                // }

                merge(planeIdToObj.at(validMatches.front()), planeIdToObj.at(validMatches[p]));

                // if (viewer) {
                //     viewer->resetStoppedFlag();
                //
                //     static bool cameraInit = false;
                //
                //     if (!cameraInit) {
                //         viewer->initCameraParameters();
                //         viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                //         cameraInit = true;
                //     }
                //     while (!viewer->wasStopped()) {
                //         viewer->spinOnce(100);
                //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
                //     }
                //
                //     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                //                                              shadingLevel,
                //                                              string("plane_ba_") + to_string(matches[p]),
                //                                              viewPort1);
                //
                //
                //     // mapObj.getHull().cleanDisplay(viewer, viewPort1);
                //
                // }
            }

            planeIdToObj[validMatches.front()]->addView(newView);
            planeIdToObj[validMatches.front()]->increaseEolCnt(settings.eolObjInstIncr);
        }
        // else{
        //     set<int> matchedIds;
        //     for(auto it : matches){
        //         matchedIds.insert(it->getId());
        //     }
        //     PendingMatchKey pmatchKey{matchedIds};
        //     if(getPendingMatch(pmatchKey)){
        //         addPendingObj(newObj, matchedIds, settings.eolPendingIncr);
        //     }
        //     else{
        //         addPendingObj(newObj, matchedIds, settings.eolPendingInit);
        //     }
        //
        //     cout << "Multiple matches" << endl;
        // }
        
        if(viewer){
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                     shadingLevel,
                                                     string("plane_nba_") + to_string(npl),
                                                     viewPort2);
            
            
            // newObj.getHull().cleanDisplay(viewer, viewPort2);
            
        }
        
        ++npl;
    }


    for (const auto &planeIdCntView : planeIdToCntView) {
        const int &planeId = planeIdCntView.first;
        const int &cnt = planeIdCntView.second.first;
        const ObjInstanceView::ConstPtr &mapView = planeIdCntView.second.second;

        if (planeIdToObj.count(planeId) > 0) {
            if (cnt > 2 * cntThreshMerge && planeIdToObj.at(planeId)->getEolCnt() < settings.eolObjInstThresh) {
                planeIdToObj.at(planeId)->decreaseEolCnt(settings.eolObjInstDecr);
            }
        }
    }
    removeObjsEol();

    nnsUptodate = false;
    
    chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();
    
    static chrono::milliseconds totalTime = chrono::milliseconds::zero();
    static int totalCnt = 0;
    
    totalTime += chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    ++totalCnt;
    
    cout << "Mean mergeNewObjInstances time: " << (totalTime.count() / totalCnt) << endl;
}

void Map::mergeMapObjInstances(pcl::visualization::PCLVisualizer::Ptr viewer,
                               int viewPort1,
                               int viewPort2)
{
    chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();

    std::unordered_map<int, ObjInstanceView::ConstPtr> planeIdToView;
    for (const auto &planeIdObj : planeIdToObj) {
        ObjInstanceView::ConstPtr bestView = planeIdObj.second->getBestQualityView();
        if (bestView) {
            planeIdToView[planeIdObj.first] = bestView;
        }
    }

    if(viewer) {
        for (const auto &planeIdView : planeIdToView) {
            planeIdView.second->display(viewer, viewPort1);
        }
    }

    for(auto it = planeIdToView.begin(); it != planeIdToView.end(); ++it) {
        const int &planeId1 = it->first;
        ObjInstanceView::ConstPtr mapView1 = it->second;

        // if plane1 still exists, i.e. hasn't been merged with any other plane
        if (planeIdToObj.count(planeId1) > 0) {
            auto it2 = it;
            ++it2;
            for (; it2 != planeIdToView.end(); ++it2) {
                const int &planeId2 = it2->first;
                ObjInstanceView::ConstPtr mapView2 = it2->second;

                // if plane2 still exists, i.e. hasn't been merged with any other plane
                if (planeIdToObj.count(planeId2) > 0) {
                    if (mapView1->isMatching(*mapView2)) {
                        cout << "Merging map obj instances " << planeId1 << " and " << planeId2 << endl;
                        merge(planeIdToObj.at(planeId1), planeIdToObj.at(planeId2));
                    }
                }
            }
        }
    }

    nnsUptodate = false;

    chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();
    
    static chrono::milliseconds totalTime = chrono::milliseconds::zero();
    static int totalCnt = 0;
    
    totalTime += chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    ++totalCnt;
    
    cout << "Mean mergeMapObjInstances time: " << (totalTime.count() / totalCnt) << endl;
}

void Map::decreaseObjEol(int eolSub) {
    for(auto &planeIdObj : planeIdToObj){
        if(planeIdObj.second->getEolCnt() < settings.eolObjInstThresh){
            planeIdObj.second->decreaseEolCnt(eolSub);
        }
    }
}

void Map::removeObjsEol() {
    removeObjsEolThresh(0);
}

void Map::removeObjsEolThresh(int eolThresh) {
    for(auto it = planeIdToObj.begin(); it != planeIdToObj.end(); ) {
        if (it->second->getEolCnt() <= eolThresh) {
            it = planeIdToObj.erase(it);
        }
        else {
            ++it;
        }
    }

    nnsUptodate = false;
}

void Map::shiftIds(int startId) {
//    map<int, int> oldIdToNewId;
    std::unordered_map<int, ObjInstance::Ptr> newPlaneIdToObj;
    for(auto &planeIdObj : planeIdToObj){
        const ObjInstance::Ptr &obj = planeIdObj.second;
        const int &planeId = planeIdObj.first;

        int newPlaneId = planeId + startId;

        obj->shiftIds(startId);

        newPlaneIdToObj[newPlaneId] = obj;
    }
    planeIdToObj.swap(newPlaneIdToObj);

    nnsUptodate = false;
}

std::vector<std::vector<ObjInstanceView::ConstPtr>> Map::getKNN(const std::vector<ObjInstanceView::ConstPtr> &viewsQuery,
                                                           int k,
                                                           double maxDist)
{
    int nq = viewsQuery.size();
    std::vector<std::vector<ObjInstanceView::ConstPtr>> retKNN(nq);

    // lazy update - only when necessary
    if (!nnsUptodate) {
        buildKdtree();
    }

    if (nns && nq > 0) {
        int descLen = viewsQuery.front()->getDescriptor().size();

        Eigen::MatrixXi idxs(k, nq);
        Eigen::MatrixXd dists2(k, nq);

        Eigen::MatrixXd descsQuery(descLen, nq);
        for (int i = 0; i < viewsQuery.size(); ++i) {
            descsQuery.col(i) = viewsQuery[i]->getDescriptor();
        }

        nns->knn(descsQuery, idxs, dists2, k);

        for (int i = 0; i < viewsQuery.size(); ++i) {
            for (int ki = 0; ki < k; ki++) {
                if (sqrt(dists2(ki, i)) / viewsQuery[i]->getDescriptor().norm() < maxDist) {
                    const int &planeIdMatch = planeIdViews[idxs(ki, i)].first;
                    const int &viewIdxMatch = planeIdViews[idxs(ki, i)].second;
                    const ObjInstanceView::Ptr &viewMatch = planeIdToObj[planeIdMatch]->getViews()[viewIdxMatch];
                    retKNN[i].push_back(viewMatch);
                }
            }
        }
    }

    return retKNN;
}

void Map::removeViews(uint64_t tsThresh) {
    for(auto it = planeIdToObj.begin(); it != planeIdToObj.end(); ) {
        const ObjInstance::Ptr &obj = it->second;
        const int &planeId = it->first;

        obj->removeViews(tsThresh);
        if (obj->getViews().empty()) {
            it = planeIdToObj.erase(it);
        }
        else {
            ++it;
        }
    }
    for (const auto &planeIdObj : planeIdToObj) {
        const ObjInstance::Ptr &obj = planeIdObj.second;
        const int &planeId = planeIdObj.first;
        obj->removeViews(tsThresh);

    }
    nnsUptodate = false;
}

pcl::PointCloud<pcl::PointXYZRGBL>::Ptr Map::getLabeledColorPointCloud()
{
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcLab(new pcl::PointCloud<pcl::PointXYZRGBL>());
    for(auto &planeIdObj : planeIdToObj) {
        ObjInstanceView::ConstPtr view = planeIdObj.second->getBestQualityView();

        if (view) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = view->getPointCloudProj();
            for (int pt = 0; pt < curPc->size(); ++pt) {
                pcl::PointXYZRGBL newPt;
                newPt.x = curPc->at(pt).x;
                newPt.y = curPc->at(pt).y;
                newPt.z = curPc->at(pt).z;
                newPt.rgb = curPc->at(pt).rgb;
                newPt.label = planeIdObj.first;
                pcLab->push_back(newPt);
            }
        }
    }
    return pcLab;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr Map::getLabeledPointCloud()
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr pcLab(new pcl::PointCloud<pcl::PointXYZL>());
    for(auto &planeIdObj : planeIdToObj) {
        ObjInstanceView::ConstPtr view = planeIdObj.second->getBestQualityView();

        if (view) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = view->getPointCloudProj();
            for (int pt = 0; pt < curPc->size(); ++pt) {
                pcl::PointXYZL newPt;
                newPt.x = curPc->at(pt).x;
                newPt.y = curPc->at(pt).y;
                newPt.z = curPc->at(pt).z;
                newPt.label = planeIdObj.first;
                pcLab->push_back(newPt);
            }
        }
    }
    return pcLab;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Map::getColorPointCloud()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol(new pcl::PointCloud<pcl::PointXYZRGB>());
    for(auto &planeIdObj : planeIdToObj) {
        // for(auto &view : planeIdObj.second->getViews()) {
            ObjInstanceView::ConstPtr view = planeIdObj.second->getBestQualityView();

            if (view) {
                // pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = view->getPointCloudProj();
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = view->getPointCloud();
                pcCol->insert(pcCol->end(), curPc->begin(), curPc->end());
            }
        // }
    }
    return pcCol;
}

void Map::display(pcl::visualization::PCLVisualizer::Ptr viewer,
                  int v1,
                  int v2,
                  float r,
                  float g,
                  float b,
                  bool dispInfo)
{
    if (v1 >= 0) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol = getColorPointCloud();
        viewer->addPointCloud(pcCol, "cloud_color_map_" + std::to_string((unsigned long long) this), v1);

        if (r >= 0.0 && g >= 0.0 && b >= 0.0) {
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                     r, g, b,
                                                     "cloud_color_map_" + std::to_string((unsigned long long) this),
                                                     v1);
        }
        if (dispInfo) {
            for (auto &planeIdObj: planeIdToObj) {
                const ObjInstance::Ptr &obj = planeIdObj.second;
                const int &planeId = planeIdObj.first;
                ObjInstanceView::ConstPtr view = obj->getBestQualityView();

                if (view) {
                    Eigen::Vector3d cent = view->getPlaneEstimator().getCentroid();
                    viewer->addText3D("id: " + to_string(view->getId()) +
                                      ", eol: " + to_string(obj->getEolCnt()),
                                      pcl::PointXYZ(cent(0), cent(1), cent(2)),
                                      0.05,
                                      1.0, 1.0, 1.0,
                                      string("plane_text_ba_") + to_string(planeId),
                                      v1);
                }
            }
        }
    }

    if (v2 >= 0) {
        pcl::PointCloud<pcl::PointXYZL>::Ptr pcLab = getLabeledPointCloud();
        viewer->addPointCloud(pcLab, "cloud_labeled_map_" + std::to_string((unsigned long long) this), v2);
    }
}

void Map::cleanDisplay(pcl::visualization::PCLVisualizer::Ptr viewer,
                  int v1,
                  int v2,
                       bool dispInfo) {
    if (v1 >= 0) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol = getColorPointCloud();
        viewer->removePointCloud("cloud_color_map_" + std::to_string((unsigned long long) this), v1);

        if (dispInfo) {
            for (auto &planeIdObj: planeIdToObj) {
                const ObjInstance::Ptr &obj = planeIdObj.second;
                const int &planeId = planeIdObj.first;
                ObjInstanceView::ConstPtr view = obj->getBestQualityView();

                if (view) {
                    viewer->removeText3D(string("plane_text_ba_") + to_string(planeId),
                                         v1);
                }
            }
        }
    }

    if (v2 >= 0) {
        viewer->removePointCloud("cloud_labeled_map_" + std::to_string((unsigned long long) this), v2);
    }
}

void Map::exportPointCloud(const std::string &path) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCol = getColorPointCloud();
    pcl::PCLPointCloud2 pcCol2;
    pcl::toPCLPointCloud2(*pcCol, pcCol2);

    pcl::PLYWriter writer;
    if(writer.writeASCII(path, pcCol2)) {
        cout << "Error writing map to file: " << path << endl;
    }
    else {
        cout << "Successfully wrote map to file: " << path << endl;
    }
}

std::unordered_map<int, std::pair<int, ObjInstanceView::ConstPtr>> Map::getVisibleObjs(const Vector7d &pose,
                                                                     const Eigen::Matrix3d &cameraMatrix,
                                                                     int rows,
                                                                     int cols,
                                                                     pcl::visualization::PCLVisualizer::Ptr viewer,
                                                                     int viewPort1,
                                                                     int viewPort2) const
{
    // cout << "getting visible" << endl;

    static constexpr double shadingLevel = 1.0/8;

    chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();

    Eigen::Matrix4d poseMat = Misc::toMatrix(pose);
    Eigen::Matrix4d poseInvMat = poseMat.inverse();
    Eigen::Matrix4d poseMatt = poseMat.transpose();
    Eigen::Matrix3d R = poseMat.block<3, 3>(0, 0);
    Eigen::Vector3d t = poseMat.block<3, 1>(0, 3);

//    vectorVector2d imageCorners;
//    imageCorners.push_back((Eigen::Vector2d() << 0, 0).finished());
//    imageCorners.push_back((Eigen::Vector2d() << cols - 1, 0).finished());
//    imageCorners.push_back((Eigen::Vector2d() << cols - 1, rows - 1).finished());
//    imageCorners.push_back((Eigen::Vector2d() << 0, rows - 1).finished());

    std::unordered_map<int, std::pair<int, ObjInstanceView::ConstPtr>> planeIdToCntView;
    for (const auto &planeIdObj : planeIdToObj) {
        ObjInstanceView::ConstPtr bestView = planeIdObj.second->getBestView(poseMat);
        if (bestView) {
            planeIdToCntView[planeIdObj.first] = std::make_pair(0, bestView);
        }
    }

    if(viewer){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        for (const auto &planeIdCntView : planeIdToCntView) {
            planeIdCntView.second.second->display(viewer, viewPort1, shadingLevel);
        }
        viewer->addCoordinateSystem();
        Eigen::Affine3f trans = Eigen::Affine3f::Identity();
        trans.matrix() = poseMat.cast<float>();
        viewer->addCoordinateSystem(0.5, trans, "camera_coord");
    }

    vector<vector<vector<pair<double, int>>>> projPlanes(rows,
                                                         vector<vector<pair<double, int>>>(cols,
                                                                                           vector<pair<double, int>>()));

    cv::Mat projPoly(rows, cols, CV_8UC1);
    for (const auto &planeIdCntView : planeIdToCntView) {
        const int &planeId = planeIdCntView.first;
        const ObjInstanceView &view = *planeIdCntView.second.second;

        // cout << "id = " << view.getId() << endl;

        Eigen::Vector4d planeEqCamera = poseMatt * view.getPlaneEq();
        // cout << "planeEqCamera = " << planeEqCamera.transpose() << endl;

        if (viewer) {
            view.cleanDisplay(viewer, viewPort1);
            view.display(viewer, viewPort1);
        }

        // condition for observing the right face of the plane
        Eigen::Vector3d normal = planeEqCamera.head<3>();
        double d = -planeEqCamera(3);
//        Eigen::Vector3d zAxis;
//        zAxis << 0, 0, 1;

//        cout << "normal.dot(zAxis) = " << normal.dot(zAxis) << endl;
//         cout << "d = " << d << endl;
        if (d > 0) {
//        vectorVector3d imageCorners3d;
//        bool valid = Misc::projectImagePointsOntoPlane(imageCorners,
//                                                       imageCorners3d,
//                                                       cameraMatrix,
//                                                       planeEqCamera);

            vector<cv::Point *> polyCont;
            vector<int> polyContNpts;

            ConcaveHull hull = view.getHull().transform(Misc::toVector(poseInvMat));

            if (viewer) {
                hull.display(viewer, viewPort1);
            }
//
//            hull.transform(poseSE3Quat.inverse().toVector());

            ConcaveHull hullClip = hull.clipToCameraFrustum(cameraMatrix, rows, cols, 0.2);

            ConcaveHull hullClipMap = hullClip.transform(pose);
            if (viewer) {
                hullClipMap.display(viewer, viewPort1, 1.0, 0.0, 0.0);
            }

            const std::vector<Eigen::MatrixPt> &polygons3d = hullClip.getPolygons3d();
            // cout << "polygons3d.size() = " << polygons3d.size() << endl;
            for (const Eigen::MatrixPt &poly3d : polygons3d) {
                polyCont.push_back(new cv::Point[poly3d.cols()]);
                polyContNpts.push_back(poly3d.cols());

//                pcl::PointCloud<pcl::PointXYZRGB>::Ptr poly3dPose(new pcl::PointCloud<pcl::PointXYZRGB>());
//                // transform to camera frame
//                pcl::transformPointCloud(*poly3d, *poly3dPose, poseInvMat);

                cv::Mat pointsReproj = Misc::reprojectTo2D(poly3d, cameraMatrix);
//                for (int pt = 0; pt < poly3d->size(); ++pt) {
//                    cout << poly3d->at(pt).getVector3fMap().transpose() << endl;
//                }
//                 cout << "cameraMatrix = " << cameraMatrix << endl;
//                 cout << "pointsReproj = " << pointsReproj << endl;

                int corrPointCnt = 0;
                for (int pt = 0; pt < pointsReproj.cols; ++pt) {
                    int u = std::round(pointsReproj.at<cv::Vec3f>(pt)[0]);
                    int v = std::round(pointsReproj.at<cv::Vec3f>(pt)[1]);
                    float d = pointsReproj.at<cv::Vec3f>(pt)[2];

                    if (u >= 0 && u < cols && v >= 0 && v < rows && d > 0) {
                        ++corrPointCnt;
                    }
                    polyCont.back()[pt] = cv::Point(u, v);
                }
                // cout << "corrPointCnt = " << corrPointCnt << endl;
                if (corrPointCnt == 0) {
                    delete[] polyCont.back();
                    polyCont.erase(polyCont.end() - 1);
                    polyContNpts.erase(polyContNpts.end() - 1);
                }
            }
            if (polyCont.size() > 0) {
                projPoly.setTo(0);

                cv::fillPoly(projPoly,
                             (const cv::Point **) polyCont.data(),
                             polyContNpts.data(),
                             polyCont.size(),
                             cv::Scalar(255));

                if (viewer) {
                    cv::imshow("proj_poly", projPoly);
                }

                vectorVector2d polyImagePts;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        if (projPoly.at<uint8_t>(r, c) > 0) {
                            polyImagePts.push_back((Eigen::Vector2d() << c, r).finished());
                        }
                    }
                }
                // cout << "polyImagePts.size() = " << polyImagePts.size() << endl;
                vectorVector3d polyPlanePts;
                bool projSuccess = Misc::projectImagePointsOntoPlane(polyImagePts,
                                                                  polyPlanePts,
                                                                     cameraMatrix,
                                                                     planeEqCamera);
                if (projSuccess) {
                    for (int pt = 0; pt < polyImagePts.size(); ++pt) {
                        int x = std::round(polyImagePts[pt](0));
                        int y = std::round(polyImagePts[pt](1));
                        // depth is z coordinate
                        double d = polyPlanePts[pt](2);

                        // cout << "inserting point at (" << y << ", " << x << ") with d = " << d << endl;

                        projPlanes[y][x].push_back(make_pair(d, planeId));
                    }
                }
            }

            for (int p = 0; p < polyCont.size(); ++p) {
                delete[] polyCont[p];
            }

            if (viewer) {
                static bool cameraInit = false;

                if (!cameraInit) {
                    viewer->initCameraParameters();
                    viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                    cameraInit = true;
                }
                viewer->resetStoppedFlag();
                while (!viewer->wasStopped()) {
                    viewer->spinOnce(50);
                    cv::waitKey(50);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }

                hull.cleanDisplay(viewer, viewPort1);
                hullClipMap.cleanDisplay(viewer, viewPort1);
            }
        }
        else {
            if (viewer) {
                viewer->resetStoppedFlag();
                while (!viewer->wasStopped()) {
                    viewer->spinOnce(50);
                    cv::waitKey(50);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }
        }
        if (viewer) {
            view.cleanDisplay(viewer, viewPort1);
            view.display(viewer, viewPort1, shadingLevel);
        }
    }

    // cout << "searching nearest" << endl;
    for(int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            vector<pair<double, int>> &curPlanes = projPlanes[r][c];
            sort(curPlanes.begin(), curPlanes.end());

            if(!curPlanes.empty()){
                double minD = curPlanes.front().first;

                for(const pair<double, int> &curPair : curPlanes){
                    if(abs(minD - curPair.first) < 0.2){
                        int id = curPair.second;

                        planeIdToCntView.at(id).first += 1;
                    }
                }
            }
        }
    }
//    for(const pair<int, int> &curCnt : idToCnt){
//        cout << "curCnt " << curCnt.first << " = " << curCnt.second << endl;
//    }

    if(viewer){
        for (const auto &planeIdCntView : planeIdToCntView) {
            planeIdCntView.second.second->cleanDisplay(viewer, viewPort1);
        }
    }

    chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();

    static chrono::milliseconds totalTime = chrono::milliseconds::zero();
    static int totalCnt = 0;

    totalTime += chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    ++totalCnt;

    // cout << "Mean getVisibleObjs time: " << (totalTime.count() / totalCnt) << endl;

    return planeIdToCntView;
}

void Map::merge(const ObjInstance::Ptr &obj1, const ObjInstance::Ptr &obj2) {
    int planeId1 = obj1->getId();
    int planeId2 = obj2->getId();
    // cout << "merging " << planeId1 << " and " << planeId2 << endl;

    // add views to the first object
    obj1->merge(*obj2);
    // remove second object
    planeIdToObj.erase(planeId2);

    nnsUptodate = false;
}

void Map::buildKdtree() {
    planeIdViews.clear();
    nns.reset();
    std::vector<Eigen::VectorXd> descsVec;
    for (const auto &planeIdObj : planeIdToObj) {
        const auto &planeId = planeIdObj.first;
        const auto &obj = planeIdObj.second;
        const auto &views = obj->getViews();

        double largestArea = 0.0;
        // for (int v = 0; v < views.size(); ++v) {
        //     double curA = views[v]->getImageArea();
        //     if (largestArea < curA) {
        //         largestArea = curA;
        //     }
        // }
        // only views larger than half the size of the largest view
        for (int v = 0; v < views.size(); ++v) {
            if (largestArea < 2.0 * views[v]->getImageArea()) {
                planeIdViews.emplace_back(planeId, v);
                descsVec.push_back(views[v]->getDescriptor());
            }
        }
    }
    if (!descsVec.empty()) {
        int descLen = descsVec.front().size();

        descs = Eigen::MatrixXd(descLen, descsVec.size());
        for (int i = 0; i < descsVec.size(); ++i) {
            descs.col(i) = descsVec[i];
            // if (i == 200 || i == 556 || i == 555 || i == 202 || i == 819) {
            //     cout << "i = " << i << endl;
            //     cout << "desc = " << descsVec[i].transpose() << endl;
            //     cout << "planeId = " << planeIdViews[i].first << ", view = " << planeIdViews[i].second << endl;
            //     cout << "view planeId = " << planeIdToObj[planeIdViews[i].first]->getViews()[planeIdViews[i].second]->getId() << endl;
            // }
        }

        nns.reset(Nabo::NNSearchD::createKDTreeLinearHeap(descs));
    }

    nnsUptodate = true;
}
