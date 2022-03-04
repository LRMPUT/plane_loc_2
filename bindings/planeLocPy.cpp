//
// Created by janw on 04.01.2020.
//

// Based on https://github.com/pybind/cmake_example

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// STL
#include <map>
#include <iostream>
#include <random>
#include <unordered_set>
#include <memory>

// Eigen
#include <Eigen/Dense>

// Boost
#include <boost/archive/binary_oarchive.hpp>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/impl/point_types.hpp>

// local
#include <Types.hpp>
#include <ObjInstanceView.hpp>
#include <Misc.hpp>
#include <Matching.hpp>


namespace py = pybind11;

#define PRINT(x) std::cout << #x << " = " << std::endl << x << std::endl


std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
  std::unordered_set<int> elems;
  for (int r = N - k; r < N; ++r) {
    int v = std::uniform_int_distribution<>(1, r)(gen);

    // there are two cases.
    // v is not in candidates ==> add it
    // v is in candidates ==> well, r is definitely not, because
    // this is the first iteration in the loop that we could've
    // picked something that big.

    if (!elems.insert(v).second) {
      elems.insert(r);
    }
  }
  return elems;
}

std::vector<ObjInstanceView::Ptr> getObjInstanceViews(const py::array_t<float> &mask,
                                                      const py::array_t<float> &params,
                                                      const py::array_t<int> &ids,
                                                      const py::array_t<uint64_t> &tss,
                                                      const py::array_t<float> &descs,
                                                      const py::array_t<float> &image,
                                                      const py::array_t<float> &K,
                                                      const py::array_t<float> &depth,
                                                      const py::array_t<float> &depthCovar,
                                                      bool projectOntoPlane = false)
{
    int nplanes = params.shape(0);
    int nrows = mask.shape(1);
    int ncols = mask.shape(2);

    std::vector<ObjInstanceView::Ptr> objInstanceViews;

    double fx = K.at(0, 0);
    double fy = K.at(1, 1);
    double cx = K.at(0, 2);
    double cy = K.at(1, 2);

    cv::Mat pointsXyz;
    cv::Mat KMat(3, 3, CV_32FC1, cv::Scalar(0));
    KMat.at<float>(0, 0) = K.at(0, 0);
    KMat.at<float>(0, 2) = K.at(0, 2);
    KMat.at<float>(1, 1) = K.at(1, 1);
    KMat.at<float>(1, 2) = K.at(1, 2);
    KMat.at<float>(2, 2) = 1.0f;
    if(!projectOntoPlane) {
        // std::cout << "projecting depth to 3-D" << std::endl;
        cv::Mat depthMat(nrows, ncols, CV_32FC1);

        // std::cout << "K = " << std::endl << KMat << std::endl;
        for (int r = 0; r < nrows; ++r) {
            for (int c = 0; c < ncols; ++c) {
                // do not use points with too large covariance
                if (sqrt(depthCovar.at(r, c)) / max(depth.at(r, c), 0.2f) < 0.1 && sqrt(depthCovar.at(r, c)) < 0.5) {
                    depthMat.at<float>(r, c) = depth.at(r, c);
                }
                else {
                    depthMat.at<float>(r, c) = 0.0f;
                }

                // if (r < 10 && c < 10) {
                //     std::cout << "depthMat.at<float>(" << r << ", " << c << ") = " << depthMat.at<float>(r, c) << std::endl;
                // }
            }
        }
//        std::cout << "calling" << std::endl;
        pointsXyz = Misc::projectTo3D(depthMat, KMat);
    }

    for(int n = 0; n < nplanes; ++n) {
        // std::cout << "plane " << n << std::endl;
        Eigen::Vector4d planeEq;
        planeEq(0) = params.at(n, 0);
        planeEq(1) = params.at(n, 1);
        planeEq(2) = params.at(n, 2);

        double d = planeEq.head<3>().norm();
        planeEq(3) = -d;
        planeEq.head<3>() /= std::max(d, 1.0e-10);
        // std::cout << "planeEq = " << planeEq.transpose() << std::endl;

        bool pointsCorr = true;
        vectorVector3d  pts3d;
        vectorVector3d pts3dCol;
        std::vector<double>  pts3dCovar;
        if(!projectOntoPlane){
            pointsCorr = true;
            for (int r = 0; r < nrows; ++r) {
                for (int c = 0; c < ncols; ++c) {
                    float z = pointsXyz.at<cv::Vec3f>(r, c)[2];
                    // if mask non-zero and depth > 0.2
                    if (std::abs(mask.at(n, r, c)) > 0.5 && z > 0.2) {
                        pts3d.emplace_back(pointsXyz.at<cv::Vec3f>(r, c)[0],
                                           pointsXyz.at<cv::Vec3f>(r, c)[1],
                                           pointsXyz.at<cv::Vec3f>(r, c)[2]);

                        pts3dCol.emplace_back(image.at(r, c, 0),
                                              image.at(r, c, 1),
                                              image.at(r, c, 2));

                        // Eigen::Matrix3d covar = Eigen::Matrix3d::Zero();
                        // // for X and Y stddev of 1 pixel
                        // covar(0, 0) = (z / fx) * (z / fx);
                        // covar(1, 1) = (z / fy) * (z / fy);
                        // // for Z covar of depth
                        // covar(2, 2) = depthCovar.at(r, c);

                        pts3dCovar.emplace_back(depthCovar.at(r, c));
                    }
                }
            }
            // std::cout << "found " << pts3d.size() << " 3-D points" << std::endl;
            if(pts3d.size() < 500){
                pointsCorr = false;
            }
        }
        else{
            vectorVector2d pts2d;
            for (int r = 0; r < nrows; ++r) {
                for (int c = 0; c < ncols; ++c) {
                    // if mask non-zero
                    if(std::abs(mask.at(n, r, c)) > 0.5){
                        pts2d.emplace_back(c, r);

                        pts3dCol.emplace_back(image.at(r, c, 0),
                                              image.at(r, c, 1),
                                              image.at(r, c, 2));
                    }
                }
            }
            // std::cout << "number of 2-D points = " << pts2d.size() << std::endl;


            for (int p = 0; p < pts2d.size(); ++p) {
                Eigen::Vector3d curPt3d = Misc::projectPointOnPlane(pts2d[p], planeEq, KMat);

                pts3d.push_back(curPt3d);

                // std dev of 0.05 m
                pts3dCovar.emplace_back(0.05 * 0.05);
            }
            if (std::abs(planeEq(3)) < 0.2) {
                pointsCorr = false;
            }
            // std::cout << "number of 3-D points = " << pts3d.size() << std::endl;
            // for (int p = 0; p < std::min(10lu, pts3d.size()); ++p) {
            //     std::cout << "pt = " << pts3d[p].transpose() << std::endl;
            // }
            // std::cout << "pointsCorr = " << pointsCorr << std::endl;
            // if(pts2d.size() > 0 &&
            //   Misc::projectImagePointsOntoPlane(pts2d,
            //                                     pts3d,
            //                                     fx, fy, cx, cy,
            //                                     planeEq))
            // {
            //     pointsCorr = true;
            // }
        }



        if(pointsCorr){
            static constexpr float dimLimit = 20;
            Eigen::MatrixPt pointCloud(4, pts3d.size());
            // Eigen::MatrixPt pointCloudOrig(4, pts3d.size());
            Eigen::MatrixCol pointCloudCol(4, pts3d.size());
            Eigen::MatrixXd pointCloudCovar(1, pts3d.size());
            int nextIdx = 0;
            for(int p = 0; p < pts3d.size(); ++p){
                if(std::abs(pts3d[p](0)) < dimLimit &&
                   std::abs(pts3d[p](1)) < dimLimit &&
                   std::abs(pts3d[p](2)) < dimLimit)
                {
                    pointCloud.col(nextIdx) << pts3d[p], 1.0;
                    // pointCloudOrig.col(nextIdx) << 0.0, 0.0, 0.0, 1.0;
                    pointCloudCol(0, nextIdx) = pts3dCol[p](2);
                    pointCloudCol(1, nextIdx) = pts3dCol[p](1);
                    pointCloudCol(2, nextIdx) = pts3dCol[p](0);
                    pointCloudCol(3, nextIdx) = 255;

                    pointCloudCovar(nextIdx) = pts3dCovar[p];

                    ++nextIdx;
                }
            }
            pointCloud.conservativeResize(Eigen::NoChange, nextIdx);
            // pointCloudOrig.conservativeResize(Eigen::NoChange, nextIdx);
            pointCloudCol.conservativeResize(Eigen::NoChange, nextIdx);
            pointCloudCovar.conservativeResize(Eigen::NoChange, nextIdx);

            Eigen::VectorXd desc(descs.shape(1));
            for (int di = 0; di < descs.shape(1); ++di) {
                desc(di) = descs.at(n, di);
            }
            // PRINT(pointCloud.cols());
            // PRINT(pointCloudCol.cols());
            // PRINT(pointCloudCovar.cols());
            // std::cout << "point cloud size = " << pointCloud->size() << std::endl;

            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            // std::cout << "creating obj instance" << std::endl;
            objInstanceViews.emplace_back(new ObjInstanceView(ids.at(n),
                                                          ObjInstanceView::ObjType::Plane,
                                                          tss.at(n),
                                                          pointCloud,
                                                          pointCloudCol,
                                                          pointCloudCovar,
                                                          desc,
                                                          planeEq,
                                                          pose));

        }

    }

    return objInstanceViews;
}

py::bytes createObjInstanceViews(const py::array_t<float> &mask,
                                 const py::array_t<float> &params,
                                 const py::array_t<int> &ids,
                                 const py::array_t<uint64_t> &tss,
                                 const py::array_t<float> &descs,
                                 const py::array_t<float> &image,
                                 const py::array_t<float> &K,
                                 const py::array_t<float> &depth,
                                 const py::array_t<float> &depthCovar,
                                 bool projectOntoPlane = true)
{
    std::vector<ObjInstanceView::Ptr> objInstanceViews = getObjInstanceViews(mask,
                                                                             params,
                                                                             ids,
                                                                             tss,
                                                                             descs,
                                                                             image,
                                                                             K,
                                                                             depth,
                                                                             depthCovar,
                                                                             projectOntoPlane);

    std::stringstream bufferSs;
    boost::archive::binary_oarchive oa(bufferSs);
    oa << objInstanceViews;
    std::string buffer = bufferSs.str();
    return py::bytes(buffer);
}

py::array_t<float> createPointCloud(const py::array_t<float> &mask,
                                     const py::array_t<float> &params,
                                    const py::array_t<int> &ids,
                                    const py::array_t<uint64_t> &tss,
                                    const py::array_t<float> &descs,
                                     const py::array_t<float> &image,
                                     const py::array_t<float> &K,
                                    const py::array_t<float> &depth,
                                    const py::array_t<float> &depthCovar,
                                    bool projectOntoPlane = true)
{
    std::vector<ObjInstanceView::Ptr> objInstanceViews = getObjInstanceViews(mask,
                                                                             params,
                                                                             ids,
                                                                             tss,
                                                                             descs,
                                                                             image,
                                                                             K,
                                                                             depth,
                                                                             depthCovar,
                                                                             projectOntoPlane);

    int pointsCnt = 0;
    for(int o = 0; o < objInstanceViews.size(); ++o){
        // PRINT(objInstances[o].getPointCloud()->size());
        pointsCnt += objInstanceViews[o]->getPointCloud()->size();
    }

    py::array_t<float> pointCloud({pointsCnt, 7}/*,
                                   {pointsCnt * 7 * sizeof(float), 7 * sizeof(float), sizeof(float)}*/);

    int idx = 0;
    for(int o = 0; o < objInstanceViews.size(); ++o){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr curPc = objInstanceViews[o]->getPointCloud();
        for(int p = 0; p < curPc->size(); ++p){
            Eigen::Matrix<float, 1, 7> curPt;
            pointCloud.mutable_at(idx, 0) = curPc->at(p).x;
            pointCloud.mutable_at(idx, 1) = curPc->at(p).y;
            pointCloud.mutable_at(idx, 2) = curPc->at(p).z;
            pointCloud.mutable_at(idx, 3) = curPc->at(p).r;
            pointCloud.mutable_at(idx, 4) = curPc->at(p).g;
            pointCloud.mutable_at(idx, 5) = curPc->at(p).b;
            pointCloud.mutable_at(idx, 6) = o;
            ++idx;
        }
    }

    return pointCloud;
}

double pdfEval(const py::array_t<double> &x,
               const py::list &refPoses,
               const py::list &transPoses,
               const py::list &matchedIds,
               const py::list &matchedAreas,
               const py::list &matchedAppDiffs)
{
    double val = 0.0;
    for (int v = 0; v < refPoses.size(); ++v) {
        py::array_t<double> curRefPose = py::cast<py::array_t<double>>(refPoses[v]);

        py::list curTransPoses = py::cast<py::list>(transPoses[v]);
        py::list curMatchedIds = py::cast<py::list>(matchedIds[v]);
        py::list curMatchedAreas = py::cast<py::list>(matchedAreas[v]);
        py::list curMatchedAppDiffs = py::cast<py::list>(matchedAppDiffs[v]);
        if (curTransPoses.size() > 0) {
            Eigen::Matrix6d Sall = Eigen::Matrix6d::Zero();
            Eigen::Vector7d curRefPoseVec;
            curRefPoseVec << curRefPose.at(0),
                            curRefPose.at(1),
                            curRefPose.at(2),
                            curRefPose.at(3),
                            curRefPose.at(4),
                            curRefPose.at(5),
                            curRefPose.at(6);
            Eigen::Matrix4d curRefPoseMat = Misc::toMatrix(curRefPoseVec);
            Eigen::Matrix4d curRefPoseMatInv = Misc::inverseTrans(curRefPoseMat);

            // if (v == 4) {
            //     PRINT(curRefPoseMatInv);
            // }
            Matching::vectorProbDistKernel kernels;
            Eigen::vectorMatrix4d curTransPoseInvMats;

            double wSum = 0.0;
            for (int t = 0; t < curTransPoses.size(); ++t) {
                py::array_t<double> curTransPose = py::cast<py::array_t<double>>(curTransPoses[t]);
                Eigen::Vector7d curTransPoseVec;
                curTransPoseVec << curTransPose.at(0),
                        curTransPose.at(1),
                        curTransPose.at(2),
                        curTransPose.at(3),
                        curTransPose.at(4),
                        curTransPose.at(5),
                        curTransPose.at(6);
                Eigen::Matrix4d curTransPoseMat = Misc::toMatrix(curTransPoseVec);
                Eigen::Matrix4d curTransPoseInvMat = Misc::inverseTrans(curTransPoseMat);
                curTransPoseInvMats.push_back(curTransPoseInvMat);

                double w = 0.0;
                py::list curTransMatchedIds = py::cast<py::list>(curMatchedIds[t]);
                py::list curTransMatchedAreas = py::cast<py::list>(curMatchedAreas[t]);
                py::list curTransMatchedAppDiffs = py::cast<py::list>(curMatchedAppDiffs[t]);
                for (int m = 0; m < curTransMatchedIds.size(); ++m) {
                    w += x.at(0) * py::cast<double>(py::cast<py::tuple>(curTransMatchedAreas[m])[1]);
                    w += x.at(1) * exp(-py::cast<double>(curTransMatchedAppDiffs[m]) / x.at(2));
                    // w += 1.0;
                }

                wSum += w;

                // Eigen::Matrix6d S = Eigen::Matrix6d::Identity();
                // S.block<3, 3>(0, 0) *= x.at(3);
                // S.block<3, 3>(3, 3) *= x.at(4);

                Eigen::Matrix6d Sinv = Eigen::Matrix6d::Identity();
                Sinv.block<3, 3>(0, 0) /= 0.367;
                Sinv.block<3, 3>(3, 3) /= 0.102;

                kernels.emplace_back(curTransPoseVec, Sinv, w);
            }

            double curP = Matching::evalPoint(curRefPoseVec, kernels) / wSum * curTransPoses.size();
            // double curP = Matching::evalPoint(curRefPoseVec, kernels) / curTransPoses.size();
            // double maxOtherP = 1.0e-12;
            //
            // for (int t = 0; t < curTransPoseInvMats.size(); ++t) {
            //     double curOtherP = Matching::evalPoint(curTransPoseInvMats[t], kernels) / wSum;
            //
            //     if (maxOtherP < curOtherP) {
            //         maxOtherP = curOtherP;
            //     }
            // }

            // PRINT(curP);
            // PRINT(maxOtherP);
            // val += SallGen - curP;
            val += -curP;

            // std::cout << SallGen - curP << endl;
        }
    }

    return val;
}

py::array_t<double> logMapSE3(const py::array_t<double> &pT) {
    Eigen::Matrix4d T;
    T << pT.at(0, 0), pT.at(0, 1), pT.at(0, 2), pT.at(0, 3),
         pT.at(1, 0), pT.at(1, 1), pT.at(1, 2), pT.at(1, 3),
         pT.at(2, 0), pT.at(2, 1), pT.at(2, 2), pT.at(2, 3),
         pT.at(3, 0), pT.at(3, 1), pT.at(3, 2), pT.at(3, 3);
    Eigen::Vector6d log = Misc::logMap(T);

    py::array_t<double> plog(py::array::ShapeContainer{6});
    plog.mutable_at(0) = log(0);
    plog.mutable_at(1) = log(1);
    plog.mutable_at(2) = log(2);
    plog.mutable_at(3) = log(3);
    plog.mutable_at(4) = log(4);
    plog.mutable_at(5) = log(5);

    return plog;
}

PYBIND11_MODULE(plane_loc_py, m) {
    m.doc() = R"pbdoc(
            Python bindings for PlaneLoc
            -----------------------
            .. currentmodule:: plane_loc_py
            .. autosummary::
               :toctree: _generate
               createObjInstances
               createPointCloud
        )pbdoc";

    m.def("createObjInstanceViews",
            &createObjInstanceViews,
            py::arg("mask"),
            py::arg("params"),
            py::arg("ids"),
            py::arg("tss"),
            py::arg("descs"),
            py::arg("image"),
            py::arg("K"),
            py::arg("depth"),
            py::arg("depthCovar"),
            py::arg("projectOntoPlane") = true,
            R"pbdoc(
            Create serialized vector of object instances.
            Create serialized vector of object instances from mask and plane parameters.
        )pbdoc");

    m.def("createPointCloud",
            &createPointCloud,
            py::arg("mask"),
            py::arg("params"),
            py::arg("ids"),
            py::arg("tss"),
            py::arg("descs"),
            py::arg("image"),
            py::arg("K"),
            py::arg("depth"),
            py::arg("depthCovar"),
            py::arg("projectOntoPlane") = true,
            R"pbdoc(
            Create point cloud representing segmented planar segments.
            Create point cloud representing segmented planar segments from mask and plane parameters.
        )pbdoc");

    m.def("pdfEval",
          &pdfEval,
          py::arg("x"),
          py::arg("refPoses"),
          py::arg("transPoses"),
          py::arg("matchedIds"),
          py::arg("matchedAreas"),
          py::arg("matchedAppDiffs"),
          R"pbdoc(
            Evaluate pdf.
            Evaluate pdf.
        )pbdoc");

    m.def("logMapSE3",
          &logMapSE3,
          py::arg("T"),
          R"pbdoc(
            Compte log map.
            Compte log map.
        )pbdoc");

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}
