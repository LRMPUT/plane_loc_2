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
#include <fstream>
#include <chrono>
#include <thread>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
// #include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <UnionFind.h>

#include "Matching.hpp"
#include "Misc.hpp"

using namespace std;

#define PRINT(x) std::cout << #x << " = " << std::endl << x << std::endl

//Matching::Matching() {
//
//}

Matching::MatchType Matching::matchLocalToGlobal(const cv::FileStorage &fs,
                                                 Map &globalMap,
                                                 Map &localMap,
                                                 vectorVector7d &bestTrans,
                                                 std::vector<double> &bestTransProbs,
                                                 std::vector<double> &bestTransMatchedRatio,
                                                 std::vector<int> &bestTransDistinct,
                                                 std::vector<Matching::ValidTransform> &retTransforms,
                                                 pcl::visualization::PCLVisualizer::Ptr viewer,
                                                 int viewPort1,
                                                 int viewPort2,
                                                 const Eigen::Matrix4d &refTrans)
{
    static constexpr int kKnn = 2;

    double planeAppThresh = (double) fs["matching"]["planeAppThresh"];

    double planeToPlaneAngThresh = (double) fs["matching"]["planeToPlaneAngThresh"];
    double planeDistThresh = (double) fs["matching"]["planeDistThresh"];
    double planeToPlaneDistThresh = (double) fs["matching"]["planeToPlaneDistThresh"];

    double scoreThresh = (double) fs["matching"]["scoreThresh"];
    double sinValsThresh = (double) fs["matching"]["sinValsThresh"];
    double planeEqDiffThresh = (double) fs["matching"]["planeEqDiffThresh"];
    double intAreaThresh = (double) fs["matching"]["intAreaThresh"];

    Eigen::Matrix3d cameraMatrix;
    int nrows, ncols;
    cv::Mat cameraMatrixMat;
    fs["planeSlam"]["cameraMatrix"] >> cameraMatrixMat;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            cameraMatrix(i, j) = cameraMatrixMat.at<float>(i, j);
        }
    }
    fs["planeSlam"]["nrows"] >> nrows;
    fs["planeSlam"]["ncols"] >> ncols;

    double shadingLevel = 1.0 / 16;

    std::vector<ObjInstanceView::ConstPtr> viewsLocal;
    std::unordered_map<int, int> idToPlaneIdLocal;
    std::unordered_map<int, int> idToIdxLocal;
    int nPlaneIdsLocal = 0;
    double areaViewsLocal = 0.0;
    for (const auto &obj : localMap) {
        const auto &bestView = obj.second->getBestQualityView();
        if (bestView) {
            viewsLocal.push_back(bestView);
            idToPlaneIdLocal[bestView->getId()] = obj.first;
            idToIdxLocal[bestView->getId()] = viewsLocal.size() - 1;

            ++nPlaneIdsLocal;
            areaViewsLocal += bestView->getHull().getTotalArea();
        }
    }

    // k = 5, 1917.405
    // k = 2, 1897.878
    // scene0000: k = 2, 90%, 1454.325
    // scene0000: k = 2, 95%, 1774.830
    std::vector<std::vector<ObjInstanceView::ConstPtr>> knns = globalMap.getKNN(viewsLocal, kKnn, 1454.325 * 1454.325);
    std::unordered_map<int, int> idToIdxGlobal;
    std::unordered_map<int, int> idToPlaneIdGlobal;
    std::vector<ObjInstanceView::ConstPtr> viewsGlobal;
    std::vector<PotMatch> potMatches;
    for (int v = 0; v < knns.size(); ++v) {
        for (int k = 0; k < knns[v].size(); ++k) {
            if (idToIdxGlobal.count(knns[v][k]->getId()) == 0) {
                viewsGlobal.push_back(knns[v][k]);
                idToIdxGlobal[knns[v][k]->getId()] = viewsGlobal.size() - 1;
            }
            potMatches.emplace_back(idToIdxGlobal.at(knns[v][k]->getId()),
                                    idToIdxLocal.at(viewsLocal[v]->getId()),
                                    (knns[v][k]->getDescriptor() - viewsLocal[v]->getDescriptor()).norm());
        }
    }
    for (const auto &obj : globalMap) {
        for (const auto &view : obj.second->getViews()) {
            idToPlaneIdGlobal[view->getId()] = obj.first;
        }
    }


    cout << "Adding sets" << endl;
    vector<vector<PotMatch> > potSets = findPotSets(potMatches,
                                                    viewsGlobal,
                                                    viewsLocal,
                                                    planeDistThresh,
                                                    planeToPlaneAngThresh,
                                                    planeToPlaneDistThresh,
                                                    viewer,
                                                    viewPort1,
                                                    viewPort2);

    chrono::high_resolution_clock::time_point endTripletTime = chrono::high_resolution_clock::now();

    cout << "potSets.size() = " << potSets.size() << endl;

    cout << "computing 3D transforms" << endl;

    std::vector<ValidTransform> transforms;

    // std::ofstream transFile("trans.log", ios_base::app);
    std::ofstream transFile;
    for (int s = 0; s < potSets.size(); ++s) {
//		cout << "s = " << s << endl;

        std::vector<ObjInstanceView::ConstPtr> curViewsGlobal;
        std::vector<ObjInstanceView::ConstPtr> curViewsLocal;
        for (int ch = 0; ch < potSets[s].size(); ++ch) {
            curViewsGlobal.push_back(viewsGlobal[potSets[s][ch].plane1]);
            curViewsLocal.push_back(viewsLocal[potSets[s][ch].plane2]);
        }

        bool fullConstrObjs = false;
        double resError;
        double svdr, svdt;
        Vector7d transformObjs = bestTransformObjs(curViewsGlobal,
                                                       curViewsLocal,
                                                       resError,
                                                       fullConstrObjs,
                                                       svdr,
                                                       svdt);

        if (transFile.is_open()) {
            transFile << potSets[s].size() << endl;
            for (int ch = 0; ch < potSets[s].size(); ++ch) {
                transFile << curViewsGlobal[ch]->getId() << " " << curViewsLocal[ch]->getId() << endl;
            }
            transFile << fullConstrObjs << endl;
            transFile << resError << endl;
            transFile << svdr << endl;
            transFile << svdt << endl;
        }

        bool isAdded = false;
        if (fullConstrObjs && resError < 1.003) {
            // cout << "resError = " << resError << ", trans = " << transformObjs.transpose() << endl;

            transforms.emplace_back(transformObjs,
                                    potSets[s],
                                    resError);
            isAdded = true;
        }
    }

    chrono::high_resolution_clock::time_point endTransformTime = chrono::high_resolution_clock::now();


    cout << "transforms.size() = " << transforms.size() << endl;
    retTransforms = transforms;
    // std::ofstream evalFile("eval.log", ios_base::app);
    std::ofstream evalFile;
    if (evalFile.is_open()) {
        evalFile << transforms.size() << endl;
    }
    for (int t = 0; t < transforms.size(); ++t) {

//		cout << "intAreas = " << transforms[t].intAreas << endl;
//		cout << "mapObjInvWeights = " << mapObjInvWeights << endl;
        double curWeight = 0.0;
        for (int p = 0; p < transforms[t].matchSet.size(); ++p) {
            curWeight += viewsLocal[transforms[t].matchSet[p].plane2]->getHull().getTotalArea();
//             PRINT(transforms[t].matchSet[p].planeAppDiff);
//             curWeight += exp(-transforms[t].matchSet[p].planeAppDiff / 100.0);
        }

        if (evalFile.is_open()) {
            evalFile << transforms[t].transform(0) << " "
                    << transforms[t].transform(1) << " "
                    << transforms[t].transform(2) << " "
                    << transforms[t].transform(3) << " "
                    << transforms[t].transform(4) << " "
                    << transforms[t].transform(5) << " "
                    << transforms[t].transform(6) << endl;
            evalFile << transforms[t].matchSet.size() << endl;
            for (int p = 0; p < transforms[t].matchSet.size(); ++p) {
                evalFile << viewsGlobal[transforms[t].matchSet[p].plane1]->getId() << " "
                         << viewsLocal[transforms[t].matchSet[p].plane2]->getId() << endl;
                evalFile << viewsGlobal[transforms[t].matchSet[p].plane1]->getHull().getTotalArea() << " "
                         << viewsLocal[transforms[t].matchSet[p].plane2]->getHull().getTotalArea() << endl;
            }
        }

        transforms[t].weight = curWeight;
//		cout << "score = " << transforms[t].score << endl;
    }

    chrono::high_resolution_clock::time_point endScoreTime = chrono::high_resolution_clock::now();

    if (transforms.size() > 0) {
        vector<double> weights;
        for (int t = 0; t < transforms.size(); ++t) {
            weights.push_back(-transforms[t].weight);
        }
        sort(weights.begin(), weights.end());
        int nKern = std::min(1000, (int)weights.size() - 1);
        double minWeight = -weights[nKern];

//        cout << "construct probability distribution using gaussian kernels" << endl;
        // construct probability distribution using gaussian kernels
        vectorProbDistKernel dist;
        Eigen::Matrix<double, 6, 6> distInfMat = Eigen::Matrix<double, 6, 6>::Identity();
        // information matrix for position
        distInfMat.block<3, 3>(0, 0) = 1.0 / 0.367 * Eigen::Matrix<double, 3, 3>::Identity();
        // information matrix for orientation
        distInfMat.block<3, 3>(3, 3) = 1.0 / 0.102 * Eigen::Matrix<double, 3, 3>::Identity();
        for (int t = 0; t < transforms.size(); ++t) {
            if (minWeight < transforms[t].weight) {
                dist.emplace_back(transforms[t].transform, distInfMat, transforms[t].weight);
            }
        }

        cout << "dist.size() = " << dist.size() << endl;

        vector<pair<double, int>> transProb;
        // find point for which the probability is the highest
//		int bestInd = 0;
//		double bestScore = numeric_limits<double>::lowest();
        for (int t = 0; t < transforms.size(); ++t) {
            double curProb = evalPoint(transforms[t].transform, dist);
            // double curProb = 0.0;
//			cout << "transform = " << transforms[t].transpose() << endl;
//			cout << "prob = " << curProb << endl;
//			if(bestScore < curScore){
//				bestScore = curScore;
//				bestInd = t;
//			}
            transProb.emplace_back(curProb, t);
        }
        sort(transProb.begin(), transProb.end());

        // seeking for at most 2 best maximas
        for (int t = transProb.size() - 1; t >= 0 && bestTrans.size() < 2; --t) {
            bool add = true;
            for (int i = 0; i < bestTrans.size() && add; ++i) {
                double diff = Misc::transformLogDist(bestTrans[i], transforms[transProb[t].second].transform);
                // if close to already added transform
                if (diff < 0.16) {
                    add = false;
                }
            }
            if (add) {
                bestTrans.push_back(transforms[transProb[t].second].transform);
                bestTransProbs.push_back(transProb[t].first);
            }
        }

        {
            for (int t = 0; t < bestTrans.size(); ++t) {

                std::vector<ObjInstanceView::ConstPtr> viewsLocalTrans;
                for (const auto &view : viewsLocal) {
                    ObjInstanceView::Ptr viewTrans(new ObjInstanceView(*view));
                    viewTrans->transform(bestTrans[t]);
                    viewsLocalTrans.push_back(viewTrans);
                }

                vector<pair<int, int>> matches;
                set<int> localMatchedIdSet;
                set<int> localMatchedPlaneIdSet;
                double localMatchedArea = 0.0;
                set<int> globalMatchedIdSet;
                set<int> globalMatchedPlaneIdSet;
                double globalMatchedArea = 0.0;
                for (int vl = 0; vl < viewsLocalTrans.size(); ++vl) {
                    for (int vg = 0; vg < viewsGlobal.size(); ++vg) {
                        const ObjInstanceView::ConstPtr &localView = viewsLocalTrans[vl];
                        const ObjInstanceView::ConstPtr &globalView = viewsGlobal[vg];
                        if (localView->isMatching(*globalView)) {
                            matches.emplace_back(vg, vl);
                            localMatchedPlaneIdSet.insert(idToPlaneIdLocal.at(viewsLocalTrans[vl]->getId()));
                            globalMatchedPlaneIdSet.insert(idToPlaneIdGlobal.at(viewsGlobal[vg]->getId()));

                            if (localMatchedIdSet.count(viewsLocalTrans[vl]->getId()) == 0) {
                                localMatchedArea += viewsLocalTrans[vl]->getHull().getTotalArea();
                                localMatchedIdSet.insert(viewsLocalTrans[vl]->getId());
                            }
                            if (globalMatchedIdSet.count(viewsGlobal[vg]->getId()) == 0) {
                                globalMatchedArea += viewsGlobal[vg]->getHull().getTotalArea();
                                globalMatchedIdSet.insert(viewsGlobal[vg]->getId());
                            }
                        }
                    }
                }

                int nPlaneIdsGlobal = 0;
                double areaViewsGlobal = 0.0;
                std::vector<ObjInstanceView::ConstPtr> visibleViewsGlobal = getVisibleObjs(viewsLocalTrans,
                                                                                           globalMap,
                                                                                           cameraMatrix,
                                                                                           nrows,
                                                                                           ncols,
                                                                                           nPlaneIdsGlobal,
                                                                                           areaViewsGlobal);

                double matchedRatio = std::min(localMatchedArea / areaViewsLocal,
                                               globalMatchedArea / areaViewsGlobal);
                int matchedDistinct = std::min(localMatchedPlaneIdSet.size(),
                                               globalMatchedPlaneIdSet.size());


                bestTransDistinct.push_back(matchedDistinct);
                bestTransMatchedRatio.push_back(matchedRatio);

                if (matches.size() >= 3) {
                    // calculate transform using all inliers
                    std::vector<ObjInstanceView::ConstPtr> curViewsGlobal;
                    std::vector<ObjInstanceView::ConstPtr> curViewsLocal;
                    for (int ch = 0; ch < matches.size(); ++ch) {
                        curViewsGlobal.push_back(viewsGlobal[matches[ch].first]);
                        curViewsLocal.push_back(viewsLocal[matches[ch].second]);
                    }

                    bool fullConstrObjs = false;
                    double resError;
                    double svdr, svdt;
                    Vector7d transformObjs = bestTransformObjs(curViewsGlobal,
                                                               curViewsLocal,
                                                               resError,
                                                               fullConstrObjs,
                                                               svdr,
                                                               svdt);
                    bestTrans[t] = transformObjs;
                    bestTransProbs[t] = evalPoint(transformObjs, dist);

                    if (refTrans != Eigen::Matrix4d::Zero()) {
                        Eigen::Matrix4d diff = Misc::inverseTrans(Misc::toMatrix(transformObjs)) * refTrans;
                        Vector6d diffLog = Misc::logMap(diff);
                        double logError = diffLog.norm();
                        double linError = diff.matrix().block<3, 1>(0, 3).norm();
                        double angError = diffLog.tail<3>().norm();

                        // std::ofstream poseFile("pose.log", ios_base::app);
                        std::ofstream poseFile;
                        if (poseFile.is_open()) {
                            bool isCorrect = linError <= 0.5 && angError * 180.0 / M_PI <= 10.0;

                            poseFile << isCorrect << " "
                                     << bestTransProbs[t] << " "
                                     << bestTransMatchedRatio[t] << " "
                                     << bestTransDistinct[t] << endl;
                        }

                        // if (viewer && (linError > 0.5 || angError * 180.0 / M_PI > 10.0)
                        //         && bestTransProbs[t] >= 0.001 && bestTransMatchedRatio[t] >= 0.45 && bestTransDistinct[t] >= 5) {
                        //     PRINT(linError);
                        //     PRINT(angError * 180.0 / M_PI);
                        //     PRINT(bestTransProbs[t]);
                        //     PRINT(bestTransDistinct[t]);
                        //     PRINT((double)localMatchedPlaneIdSet.size() / nPlaneIdsLocal);
                        //     PRINT((double)globalMatchedPlaneIdSet.size() / nPlaneIdsGlobal);
                        //     PRINT(localMatchedArea / areaViewsLocal);
                        //     PRINT(globalMatchedArea / areaViewsGlobal);
                        //
                        //     viewer->removeAllPointClouds();
                        //     viewer->removeAllShapes();
                        //     viewer->removeAllCoordinateSystems();
                        //
                        //     viewer->addCoordinateSystem();
                        //     viewer->addCoordinateSystem(1.0, Eigen::Affine3f(refTrans.cast<float>()), "ref");
                        //     viewer->addCoordinateSystem(1.0, Eigen::Affine3f(Misc::toMatrix(transformObjs).cast<float>()), "est");
                        //
                        //     for (auto &view: curViewsGlobal) {
                        //         PRINT(view->getId());
                        //         view->display(viewer, viewPort1, 0.8, 1.0, 0.0, 0.0);
                        //     }
                        //
                        //     for (auto &view: curViewsLocal) {
                        //         ObjInstanceView::Ptr viewTrans = view->copy();
                        //         viewTrans->transform(transformObjs);
                        //         PRINT(viewTrans->getId());
                        //         // PRINT(reinterpret_cast<size_t>(&(*viewTrans)));
                        //         viewTrans->display(viewer, viewPort1);
                        //     }
                        //
                        //     viewer->resetStoppedFlag();
                        //     viewer->initCameraParameters();
                        //     viewer->setCameraPosition(0.0, 0.0, -6.0, 0.0, 1.0, 0.0);
                        //     viewer->spinOnce(100);
                        //     while (!viewer->wasStopped()) {
                        //         viewer->spinOnce(100);
                        //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        //     }
                        // }
                    }
                }
            }

        }

    }

    // No satisfying transformations - returning identity
    if (bestTrans.size() == 0) {
//		bestTrans << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        return MatchType::Unknown;
    }

    return MatchType::Ok;
}


double Matching::compAngleDiffBetweenNormals(const Eigen::Vector3d &nf1,
                                             const Eigen::Vector3d &ns1,
                                             const Eigen::Vector3d &nf2,
                                             const Eigen::Vector3d &ns2) {
    double ang1 = acos(nf1.dot(ns1));
//    ang1 = min(ang1, pi - ang1);
    double ang2 = acos(nf2.dot(ns2));
//    ang2 = min(ang2, pi - ang2);
    double angDiff = std::abs(ang1 - ang2);

    return angDiff;
}


bool Matching::checkPlaneToPlaneAng(const vectorVector4d &planes1,
                                    const vectorVector4d &planes2,
                                    double planeToPlaneAngThresh) {
    bool isConsistent = true;
    for (int pf = 0; pf < planes1.size(); ++pf) {
        for (int ps = pf + 1; ps < planes1.size(); ++ps) {
            Eigen::Vector3d nf1 = planes1[pf].head<3>().normalized();
            Eigen::Vector3d ns1 = planes1[ps].head<3>().normalized();
            Eigen::Vector3d nf2 = planes2[pf].head<3>().normalized();
            Eigen::Vector3d ns2 = planes2[ps].head<3>().normalized();

            double angDiff = compAngleDiffBetweenNormals(nf1, ns1, nf2, ns2);

            if (angDiff > planeToPlaneAngThresh) {
                isConsistent = false;
                break;
            }
        }
    }

    return isConsistent;
}


std::vector<std::vector<Matching::PotMatch>> Matching::findPotSets(const std::vector<PotMatch> &potMatches,
                                                                   const std::vector<ObjInstanceView::ConstPtr> &globalViews,
                                                                   const std::vector<ObjInstanceView::ConstPtr> &localViews,
                                                                   double planeDistThresh,
                                                                   double planeToPlaneAngThresh,
                                                                   double planeToPlaneDistThresh,
                                                                   pcl::visualization::PCLVisualizer::Ptr viewer,
                                                                   int viewPort1,
                                                                   int viewPort2)
{
    // static constexpr double planeToPlaneDistThresh = 1.0;

    vector<vector<PotMatch> > potSets;
    vector<vector<int>> potSetsIdxs;

    vector<vector<double>> globalObjDistances;
    compObjDistances(globalViews, globalObjDistances);
    vector<vector<double>> localObjDistances;
    compObjDistances(localViews, localObjDistances);


    // initialize with single matches
    potSets.resize(potMatches.size());
    for (int p = 0; p < potMatches.size(); ++p) {
        potSets[p].push_back(potMatches[p]);
        potSetsIdxs.push_back(vector<int>{p});
    }

    // std::ofstream setsFile("sets.log", ios_base::app);
    std::ofstream setsFile;
    // generate dublets and triplets
    for (int ne = 2; ne <= 3; ++ne) {
//        cout << "ne = " << ne << endl;
//        cout << "potSets.size() = " << potSets.size() << endl;

        vector<vector<PotMatch> > newPotSets;
        vector<vector<int>> newPotSetsIdxs;
        unordered_set<uint64_t> newPotSetsIdxsSet;

        for (int s = 0; s < potSets.size(); ++s) {

            for (int p = 0; p < potMatches.size(); ++p) {
                vector<PotMatch> curSet = potSets[s];
                curSet.push_back(potMatches[p]);
//                cout << "matches:" << endl;
//                for(int ch = 0; ch < curSet.size(); ++ch){
//                    cout << curSet[ch].plane1 << " " << curSet[ch].plane2 << endl;
//                }
                vector<int> curIdxs = potSetsIdxs[s];
                curIdxs.push_back(p);
//                sort(curIdxs.begin(), curIdxs.end());
                for (int i = 0; i < curIdxs.size() - 1; ++i) {
                    for (int j = 0; j < curIdxs.size() - i - 1; ++j) {
                        if (curIdxs[j] > curIdxs[j + 1]) {
                            swap(curIdxs[i], curIdxs[j]);
                        }
                    }
                }

                static constexpr int mult = 10000;
                int curMult = 1;
                uint64_t curHashValue = 0;
                for (int i = 0; i < curIdxs.size(); ++i) {
                    curHashValue += curIdxs[i] * curMult;
                    curMult *= mult;
                }

                bool valid = true;
                // if there was the same combination
                if (newPotSetsIdxsSet.count(curHashValue) > 0) {
                    valid = false;
                }

                if (valid) {
                    for (int ch1 = 0; ch1 < curSet.size(); ++ch1) {
                        for (int ch2 = ch1 + 1; ch2 < curSet.size(); ++ch2) {
                            if (curSet[ch1].plane1 == curSet[ch2].plane1 || curSet[ch1].plane2 == curSet[ch2].plane2) {
                                valid = false;
                            }
                        }
                    }
                }
                if (valid) {
                    for (int p1 = 0; p1 < curSet.size(); ++p1) {
                        for (int p2 = p1 + 1; p2 < curSet.size(); ++p2) {
                            int pm1 = curSet[p1].plane1;
                            int pm2 = curSet[p2].plane1;
                            // if planes are not close enough
                            if (globalObjDistances[pm1][pm2] > planeDistThresh) {
                                valid = false;
                            }
                            int pf1 = curSet[p1].plane2;
                            int pf2 = curSet[p2].plane2;
                            if (localObjDistances[pf1][pf2] > planeDistThresh) {
                                valid = false;
                            }
                            // if difference of distances is too large
                            if (std::abs(globalObjDistances[pm1][pm2] - localObjDistances[pf1][pf2]) > planeToPlaneDistThresh) {
                                valid = false;
                            }
                        }
                    }
                }
                if (valid) {
                    // check angles between planes

                    vectorVector4d planesMap;
                    vectorVector4d planesFrame;
                    for (int ch = 0; ch < curSet.size(); ++ch) {
                        planesMap.push_back(globalViews[curSet[ch].plane1]->getPlaneEq());
                        planesFrame.push_back(localViews[curSet[ch].plane2]->getPlaneEq());
                    }

                    if (!checkPlaneToPlaneAng(planesMap, planesFrame, planeToPlaneAngThresh)) {
                        valid = false;
                    }
                    if (valid) {
                        newPotSets.push_back(curSet);
                        newPotSetsIdxs.push_back(curIdxs);
                        newPotSetsIdxsSet.insert(curHashValue);
                    }
                }

                if (curSet.size() == 2) {
                    vectorVector4d planesMap;
                    vectorVector4d planesFrame;
                    for (int ch = 0; ch < curSet.size(); ++ch) {
                        planesMap.push_back(globalViews[curSet[ch].plane1]->getPlaneEq());
                        planesFrame.push_back(localViews[curSet[ch].plane2]->getPlaneEq());
                    }

                    for (int p1 = 0; p1 < curSet.size(); ++p1) {
                        for (int p2 = p1 + 1; p2 < curSet.size(); ++p2) {
                            int pm1 = curSet[p1].plane1;
                            int pm2 = curSet[p2].plane1;

                            int pf1 = curSet[p1].plane2;
                            int pf2 = curSet[p2].plane2;

                            if (pm1 != pm2 && pf1 != pf2) {
                                Eigen::Vector3d nf1 = planesMap[p1].head<3>().normalized();
                                Eigen::Vector3d ns1 = planesMap[p2].head<3>().normalized();
                                Eigen::Vector3d nf2 = planesFrame[p1].head<3>().normalized();
                                Eigen::Vector3d ns2 = planesFrame[p2].head<3>().normalized();

                                double angDiff = compAngleDiffBetweenNormals(nf1, ns1, nf2, ns2);

                                if (setsFile.is_open()) {
                                    setsFile << globalViews[pm1]->getId() << " " << globalViews[pm2]->getId() << endl;
                                    setsFile << globalObjDistances[pm1][pm2] << endl;
                                    setsFile << localViews[pf1]->getId() << " " << localViews[pf2]->getId() << endl;
                                    setsFile << localObjDistances[pf1][pf2] << endl;
                                    setsFile << std::abs(globalObjDistances[pm1][pm2] - localObjDistances[pf1][pf2])
                                             << endl;
                                    setsFile << angDiff << endl;
                                }
                            }
                        }
                    }
                }
            }
        }

        potSets.swap(newPotSets);
        potSetsIdxs.swap(newPotSetsIdxs);
    }

    return potSets;
}


void Matching::compObjDistances(const std::vector<ObjInstanceView::ConstPtr> &views,
                             std::vector<std::vector<double>> &objDistances)
{
    objDistances.resize(views.size(), vector<double>(views.size(), 0));

    for (int o1 = 0; o1 < views.size(); ++o1) {
//        cout << "o1 = " << o1 << endl;
        for (int o2 = o1 + 1; o2 < views.size(); ++o2) {
//            cout << "o2 = " << o2 << endl;
            double dist = (views[o1]->getPlaneEstimator().getCentroid() - views[o2]->getPlaneEstimator().getCentroid()).norm();
//			cout << "o1 = " << o1 << ", o2 = " << o2 << endl;
            objDistances[o1][o2] = dist;
            objDistances[o2][o1] = dist;
        }
    }
}


Vector7d Matching::bestTransformObjs(const std::vector<ObjInstanceView::ConstPtr> &objs1,
                                     const std::vector<ObjInstanceView::ConstPtr> &objs2,
                                     double &resError,
                                     bool &fullConstr,
                                     double &svdr,
                                     double &svdt)
{
    // Build the problem for the linear algorithm
    Eigen::MatrixXd A(objs1.size()*4*3, 12);
    Eigen::VectorXd b(objs1.size()*4*3);
    int nEq = 0;
    for(int o = 0; o < objs1.size(); ++o){
        Eigen::Matrix3d constrVectors1 = objs1[o]->getPlaneEstimator().getConstrVectors();
        // Eigen::Vector3d centroid1 = objs1[o]->getPlaneEstimator().getCentroid();
        Eigen::Vector3d centroid1 = objs1[o]->getCentroid();
        // Eigen::MatrixXd eqPoints2 = objs2[o]->getPlaneEstimator().getEqPoints();
        Eigen::MatrixXd eqPoints2 = objs2[o]->getEqPoints();
        // PRINT(constrVectors1);
        // PRINT(centroid1);
        // PRINT(eqPoints2);
        for(int cv = 0; cv < constrVectors1.cols(); ++cv) {
            for(int pt = 0; pt < eqPoints2.cols(); ++pt) {
                A.block<1, 3>(nEq, 0) = constrVectors1.col(cv)(0) * eqPoints2.col(pt).transpose();
                A.block<1, 3>(nEq, 3) = constrVectors1.col(cv)(1) * eqPoints2.col(pt).transpose();
                A.block<1, 3>(nEq, 6) = constrVectors1.col(cv)(2) * eqPoints2.col(pt).transpose();
                A.block<1, 3>(nEq, 9) = constrVectors1.col(cv).transpose();

                b(nEq) = constrVectors1.col(cv).transpose() * centroid1;

                ++nEq;
            }
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd rt = svd.solve(b);

    // PRINT(svd.singularValues().transpose());
    // PRINT(A);
    // PRINT(b);
    // PRINT(rt);

    Eigen::Matrix3d R;
    R.row(0) = rt.segment<3>(0);
    R.row(1) = rt.segment<3>(3);
    R.row(2) = rt.segment<3>(6);
    Eigen::Vector3d t = rt.segment<3>(9);

    // Orthonormalization of the rotation matrix
    Eigen::JacobiSVD<Eigen::MatrixXd> svdR(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // PRINT(svdR.singularValues().transpose());
    R = svdR.matrixU() * svdR.matrixV().transpose();

    // PRINT(R);
    // PRINT(t);

    Eigen::MatrixXd J(objs1.size()*4*3, 6);
    Eigen::VectorXd error(objs1.size()*4*3);
    // Gauss Newton optimization
    for(int iter = 0; iter < 5; ++iter) {
        nEq = 0;
        for(int o = 0; o < objs1.size(); ++o) {
            Eigen::Matrix3d constrVectors1 = objs1[o]->getPlaneEstimator().getConstrVectors();
            Eigen::Vector3d centroid1 = objs1[o]->getPlaneEstimator().getCentroid();
            Eigen::MatrixXd eqPoints2 = objs2[o]->getPlaneEstimator().getEqPoints();
            for (int cv = 0; cv < constrVectors1.cols(); ++cv) {
                for (int pt = 0; pt < eqPoints2.cols(); ++pt) {
                    J.block<1, 3>(nEq, 0) = -constrVectors1.col(cv).transpose() * R * Misc::skew(eqPoints2.col(pt));
                    J.block<1, 3>(nEq, 3) = constrVectors1.col(cv).transpose();

                    error(nEq) = constrVectors1.col(cv).transpose() * (R * eqPoints2.col(pt) + t - centroid1);

                    ++nEq;
                }
            }
        }
        // PRINT(J);
        // PRINT(error);

        Eigen::MatrixXd Jt = J.transpose();
        Eigen::MatrixXd JtJ = Jt * J;
        Eigen::MatrixXd Jte = Jt * error;

        Eigen::JacobiSVD<Eigen::MatrixXd> svdIter(JtJ, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd delta = svdIter.solve(-Jte);

        // PRINT(delta);
        if(delta.norm() < 1.0e-3) {
            break;
        }
        // Update
        Eigen::Matrix3d R_d = Misc::expMap((Eigen::Vector3d)delta.head<3>()).matrix();
        // PRINT(R_d);
        R = R * R_d;
        t = t + delta.tail<3>();

        // PRINT(R);
        // PRINT(t.transpose());
    }

    {
        Eigen::MatrixXd JRt = J.leftCols(3).transpose();
        Eigen::MatrixXd JRtJR = JRt * J.leftCols(3);
        Eigen::JacobiSVD<Eigen::MatrixXd> svdIterR(JRtJR, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd Jtt = J.rightCols(3).transpose();
        Eigen::MatrixXd JttJt = Jtt * J.rightCols(3);
        Eigen::JacobiSVD<Eigen::MatrixXd> svdItert(JttJt, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // PRINT(svdIterR.singularValues()(2));
        // PRINT(svdIterR.singularValues()(0) / svdIterR.singularValues()(2));
        // PRINT(svdItert.singularValues()(2));
        // PRINT(svdItert.singularValues()(0) / svdItert.singularValues()(2));

        // If the smallest singular value is below the threshold
        svdr = svdIterR.singularValues()(2);
        svdt = svdItert.singularValues()(2);
        if (svdr < 323.464 || svdt < 83.088) {
            fullConstr = false;
        }
        else {
            fullConstr = true;
        }
    }
    // PRINT(svdIter.singularValues().transpose());
    // PRINT(error.transpose());
    // PRINT(error.squaredNorm() / error.size());

    resError = sqrt(error.squaredNorm() / error.size());
    // PRINT(resError);

    return Misc::toVector(R, t);
}


std::vector<ObjInstanceView::ConstPtr>
        Matching::getVisibleObjs(const std::vector<ObjInstanceView::ConstPtr> &viewsLocalTrans,
                                 const Map &map,
                                 const Eigen::Matrix3d &cameraMatrix,
                                 int rows,
                                 int cols,
                                 int &nPlaneIds,
                                 double &areaViews)
{
    static constexpr int visThresh = 1600;

    std::vector<ObjInstanceView::ConstPtr> retViews;

    Eigen::vectorMatrix4d poses;
    for (const auto &view : viewsLocalTrans) {
        bool unique = true;
        for (const auto &curPose : poses) {
            Eigen::Matrix4d Td = curPose.inverse() * view->getPose();
            double transDiff = Td.block<3, 1>(0, 3).norm();
            double angDiff = Misc::logMap(Eigen::Quaterniond(Td.block<3, 3>(0, 0))).norm();
            if (transDiff < 0.01 && angDiff * 180.0 / M_PI < 5.0) {
                unique = false;
            }
        }
        if (unique) {
            poses.push_back(view->getPose());
        }
    }

    std::unordered_set<int> idSet;
    std::unordered_map<int, double> planeIdToArea;
    for (const auto &curPose : poses) {
        // PRINT(curPose);
        std::unordered_map<int, std::pair<int, ObjInstanceView::ConstPtr>>
            curVisible = map.getVisibleObjs(Misc::toVector(curPose),
                                            cameraMatrix,
                                            rows,
                                            cols);
        for (const auto &planeIdCntView : curVisible) {
            if (planeIdCntView.second.first > visThresh) {
                if (idSet.count(planeIdCntView.second.second->getId()) == 0) {
                    // PRINT(planeIdCntView.second.second->getId());
                    // PRINT(planeIdCntView.first);

                    retViews.push_back(planeIdCntView.second.second);

                    idSet.insert(planeIdCntView.second.second->getId());
                    if (planeIdToArea.count(planeIdCntView.first) == 0) {
                        planeIdToArea[planeIdCntView.first] = 0.0;
                    }
                    planeIdToArea[planeIdCntView.first] = std::max(planeIdToArea[planeIdCntView.first],
                                                                   planeIdCntView.second.second->getHull().getTotalArea());
                }
            }
        }
    }

    nPlaneIds = planeIdToArea.size();
    areaViews = 0.0;
    for (const auto &planeIdArea : planeIdToArea) {
        areaViews += planeIdArea.second;
    }

    return retViews;
}


double Matching::evalPoint(const Vector7d &pt,
                           const vectorProbDistKernel &dist) {
    double res = 0.0;
    double wSum = 0.0;
    for (int k = 0; k < dist.size(); ++k) {
        res += dist[k].eval(pt);
        wSum += dist[k].getW();
    }
    // if (wSum > 1.0e-12) {
    //     res /= wSum;
    // }
    return res;
}

double Matching::evalPoint(const Eigen::Matrix4d &ptInvMat,
                           const vectorProbDistKernel &dist) {
    double res = 0.0;
    double wSum = 0.0;
    for (int k = 0; k < dist.size(); ++k) {
        res += dist[k].eval(ptInvMat);
        wSum += dist[k].getW();
    }
    // if (wSum > 1.0e-12) {
    //     res /= wSum;
    // }
    return res;
}

Matching::ProbDistKernel::ProbDistKernel(const Vector7d &ikPt,
                                         const Eigen::Matrix<double, 6, 6> &iinfMat,
                                         const double &iweight)
        :
        kPt(ikPt),
        infMat(iinfMat),
        denom(std::sqrt(std::pow(2 * M_PI, 6) * iinfMat.inverse().determinant())),
        weight(iweight),
        kPtMat(Misc::toMatrix(ikPt)) {

}

double Matching::ProbDistKernel::eval(const Vector7d &pt) const {
    Eigen::Matrix4d ptMat = Misc::toMatrix(pt);
    double res = eval(Misc::inverseTrans(ptMat));
    return res;
}

double Matching::ProbDistKernel::eval(const Eigen::Matrix4d &ptInvMat) const {
    Vector6d diff = Misc::logMap(ptInvMat * kPtMat);
    double res = weight / denom *  exp(-0.5 *diff.transpose() * infMat * diff);
    return res;
}
