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

#include <thread>
#include <chrono>
#include <vector>
#include <list>

//#include <pcl/sample_consensus/model_types.h>
// #include <pcl/filters/project_inliers.h>
// #include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>

#include "ObjInstanceView.hpp"
#include "Misc.hpp"
#include "Exceptions.hpp"
#include "Types.hpp"
#include "Matching.hpp"

#define PRINT(x) std::cout << #x << " = " << std::endl << x << std::endl

using namespace std;

ObjInstanceView::ObjInstanceView() : id(-1) {}

ObjInstanceView::ObjInstanceView(int iid,
                                 ObjType itype,
                                 uint64_t its,
                                 const Eigen::MatrixPt &ipoints,
                                 const Eigen::MatrixCol &ipointsCol,
                                 const Eigen::MatrixXd &ipointsCovar,
                                 const Eigen::MatrixXd &idescriptor,
                                 const Eigen::Vector4d &iplaneEq,
                                 const Eigen::Matrix4d &ipose)
	: id(iid),
	  type(itype),
      ts(its),
	  points(ipoints),
	  pointsCol(ipointsCol),
	  pointsCovar(ipointsCovar),
      descriptor(idescriptor),
      planeEq(iplaneEq),
      pose(ipose),
	  hull(new ConcaveHull())
{
    // normalize, so first components are (X, Y, Z)
    points.array().rowwise() /= points.array().row(3);

    if (((pose - Eigen::Matrix4d::Identity()).array().abs() > 1.0e-6).any()) {
        throw PLANE_EXCEPTION("Pose has to be identity when creating ObjInstanceView");
    }
    planeEstimator.init(points, pointsCovar);
    eqPoints = planeEstimator.compEqPointsPlaneEq(planeEq);
    centroid = planeEstimator.compCentroidPlaneEq(planeEq);

    imageArea = points.cols();

    filter();
    // cout << "\n\nid = " << id << endl;
    // cout << "points = \n" << points << endl;
    hull->init(points, getPlaneEq());
    // hull->check();
    // cout << "hull->getTotalArea() = " << hull->getTotalArea() << endl;

    // filter();
    projectOntoPlane();

//    cout << "points->size() = " << points->size() << endl;

}


ObjInstanceView::ObjInstanceView(const ObjInstanceView &other)
    : hull(new ConcaveHull())
{
    id = other.id;
    type = other.type;
    ts = other.ts;
    points = other.points;
    pointsCol = other.pointsCol;
    pointsCovar = other.pointsCovar;
    pointsProj = other.pointsProj;
    descriptor = other.descriptor;
    planeEq = other.planeEq;
    pose = other.pose;
    *hull = *other.hull;
    planeEstimator = other.planeEstimator;
    eqPoints = other.eqPoints;
    centroid = other.centroid;
    imageArea = other.imageArea;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ObjInstanceView::getPointCloud() const {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>(points.cols(), 1));
    for(int p = 0; p < points.cols(); ++p) {
        pcl::PointXYZRGB pt;
        pt.getVector4fMap() = points.col(p).cast<float>();
        pt.r = pointsCol.col(p)(0);
        pt.g = pointsCol.col(p)(1);
        pt.b = pointsCol.col(p)(2);

        pointCloud->at(p) = pt;
    }

    return pointCloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ObjInstanceView::getPointCloudProj() const {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for(int p = 0; p < points.cols(); ++p) {
        pcl::PointXYZRGB pt;
        pt.getVector4fMap() = pointsProj.col(p).cast<float>();
        pt.r = pointsCol.col(p)(0);
        pt.g = pointsCol.col(p)(1);
        pt.b = pointsCol.col(p)(2);

        pointCloud->push_back(pt);
    }

    return pointCloud;
}

bool ObjInstanceView::isMatching(const ObjInstanceView &other,
                             pcl::visualization::PCLVisualizer::Ptr viewer,
                             int viewPort1,
                             int viewPort2) const
{
    // double normDot = getNormal().dot(other.getNormal());
    // cout << "normDot = " << normDot << endl;
    // cout << "angle diff = " << acos(normDot) * 180.0 / M_PI << endl;
    // if the faces are roughly oriented in the same direction
    // if (acos(normDot) < 45 * M_PI / 180.0) {
    if (true) {

        double dist1 = planeEstimator.distance(other.getEqPoints());
        double dist2 = other.getPlaneEstimator().distance(getEqPoints());

        if (viewer) {
            display(viewer, viewPort1);
            other.display(viewer, viewPort1, 1.0, 1.0, 0.0, 0.0);
            // other.display(viewer, viewPort2);

            static bool init = false;
            if (!init) {

                viewer->resetStoppedFlag();
                viewer->initCameraParameters();
                viewer->setCameraPosition(0.0, 0.0, 6.0, 0.0, -1.0, 0.0);

                while (!viewer->wasStopped()) {
                    viewer->spinOnce(100);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }

                // viewer->spinOnce(100);

                init = true;
            } else {
                // viewer->spinOnce(100);
                viewer->resetStoppedFlag();
                while (!viewer->wasStopped()) {
                    viewer->spinOnce(100);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }

            cleanDisplay(viewer, viewPort1);
            other.cleanDisplay(viewer, viewPort1);
            // other.cleanDisplay(viewer, viewPort2);
        }
        // if point distributions are similar
        if (dist1 < 2.0 && dist2 < 2.0) {

            double descDist = descriptorDist(descriptor, other.getDescriptor());
            // cout << "histDist = " << histDist << endl;
            if (descDist < std::numeric_limits<double>::infinity()) {
            // if (descDist < 0.5) {
                return true;
            }
        }
    }
    return false;
}


void ObjInstanceView::transform(const Vector7d &transform) {
    Eigen::Matrix4d T = Misc::toMatrix(transform);
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);

    points = T * points;
    pointsProj = T * pointsProj;

    planeEq = T.transpose().inverse() * planeEq;

    pose = T * pose;

    // do not transform pointsCovar

    *hull = hull->transform(transform);

    planeEstimator.transform(transform);

    eqPoints = (R * eqPoints).colwise() + t;

    centroid = R * centroid + t;
}

// void ObjInstanceView::compEqPoints() {
//
// }

void ObjInstanceView::compColorHist() {
    // color histogram
    int hbins = 32;
    int sbins = 32;
    int histSizeH[] = {hbins};
    int histSizeS[] = {sbins};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    const float* rangesH[] = {hranges};
    const float* rangesS[] = {sranges};
    int channelsH[] = {0};
    int channelsS[] = {0};
    
    int npts = points.cols();
    cv::Mat matPts(1, npts, CV_8UC3);
    for(int p = 0; p < npts; ++p){
        matPts.at<cv::Vec3b>(p)[0] = pointsCol(2, p);
        matPts.at<cv::Vec3b>(p)[1] = pointsCol(1, p);
        matPts.at<cv::Vec3b>(p)[2] = pointsCol(0, p);
    }
    cv::cvtColor(matPts, matPts, cv::COLOR_RGB2HSV);
    cv::Mat hist;
    cv::calcHist(&matPts,
                 1,
                 channelsH,
                 cv::Mat(),
                 hist,
                 1,
                 histSizeH,
                 rangesH);
    // normalization
    hist /= npts;
    hist.reshape(1,hbins);
    
    cv::Mat histS;
    cv::calcHist(&matPts,
                 1,
                 channelsS,
                 cv::Mat(),
                 histS,
                 1,
                 histSizeS,
                 rangesS);
    // normalization
    histS /= npts;
    histS.reshape(1,sbins);
    
    // add S part of histogram
    hist.push_back(histS);

    descriptor = Eigen::VectorXd(hist.rows);
    for (int i = 0; i < hist.rows; ++i) {
        descriptor(i) = hist.at<float>(i);
    }
}

void ObjInstanceView::projectOntoPlane() {
    pointsProj = Misc::projectImagePointsOntoPlane(points, pose.block<3, 1>(0, 3), planeEq);
}

void ObjInstanceView::filter() {
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pointCloud = getPointCloud();

    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB> octree(0.05f);
    octree.setInputCloud(pointCloud);
    octree.addPointsFromInputCloud();
    std::set<int> pointIdxs;
    for (auto it = octree.leaf_depth_begin(); it != octree.leaf_depth_end(); ++it) {
        // we're keeping the point with the smallest variance
        double smallestCovar = std::numeric_limits<float>::max();
        int smallestCovarIdx = -1;

        const auto &voxelIdxs = it.getLeafContainer().getPointIndicesVector();
        for (const auto &idx : voxelIdxs) {
            if (pointsCovar(idx) < smallestCovar) {
                smallestCovar = pointsCovar(idx);
                smallestCovarIdx = idx;
            }
        }

        if (smallestCovarIdx >= 0) {
            pointIdxs.insert(smallestCovarIdx);
        }
    }

    Eigen::MatrixPt newPoints(points.rows(), pointIdxs.size());
    Eigen::MatrixCol newPointsCol(pointsCol.rows(), pointIdxs.size());
    Eigen::MatrixXd newPointsCovar(pointsCovar.rows(), pointIdxs.size());

    int newIdx = 0;
    for (const auto &idx : pointIdxs) {
        newPoints.col(newIdx) = points.col(idx);
        newPointsCol.col(newIdx) = pointsCol.col(idx);
        newPointsCovar.col(newIdx) = pointsCovar.col(idx);
        ++newIdx;
    }

    std::swap(points, newPoints);
    std::swap(pointsCol, newPointsCol);
    std::swap(pointsCovar, newPointsCovar);
}


double ObjInstanceView::descriptorDist(const Eigen::VectorXd &desc1, const Eigen::VectorXd &desc2) {
//            double histDist = cv::compareHist(frameObjFeats[of], mapObjFeats[om], cv::HISTCMP_CHISQR);
    return (desc1 - desc2).norm() / desc1.norm();
}

double ObjInstanceView::viewDist(const Eigen::Matrix4d &viewPose) const {
    static constexpr double angW = 5.0;

    Eigen::Matrix4d T = viewPose.inverse() * pose;
    double angDiff = Misc::logMap(Eigen::Quaterniond(T.block<3, 3>(0, 0))).norm();
    double transDiff = T.block<3, 1>(0, 3).norm();

    return angDiff * angW + transDiff;
}

double ObjInstanceView::getQuality() const {
    return 1.0 / planeEstimator.getCurv();
}

void ObjInstanceView::display(pcl::visualization::PCLVisualizer::Ptr viewer,
                              int vp,
                              double shading,
                              double r, double g, double b) const {
    // string idStr = to_string(reinterpret_cast<size_t>(this));
    string idStr = to_string(id);

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pointCloud = getPointCloud();
    viewer->addPointCloud(pointCloud, string("obj_instance_") + idStr, vp);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                             shading,
                                             string("obj_instance_") + idStr,
                                             vp);
    if (r != 0.0 || g != 0.0 || b != 0.0) {
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 r, g, b,
                                                 string("obj_instance_") + idStr,
                                                 vp);
    }
}

void ObjInstanceView::cleanDisplay(pcl::visualization::PCLVisualizer::Ptr viewer, int vp) const {
    // string idStr = to_string(reinterpret_cast<size_t>(this));
    string idStr = to_string(id);

    viewer->removePointCloud(string("obj_instance_") + idStr,
                             vp);
}

ObjInstanceView::Ptr ObjInstanceView::copy() const {
    // use copy constructor
    return ObjInstanceView::Ptr(new ObjInstanceView(*this));
}
