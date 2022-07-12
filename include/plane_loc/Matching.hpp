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

#ifndef INCLUDE_MATCHING_HPP_
#define INCLUDE_MATCHING_HPP_

#include <vector>
#include <tuple>

#include <opencv2/opencv.hpp>

#include <Eigen/Eigen>

#include "Types.hpp"
#include "ObjInstanceView.hpp"
#include "Map.hpp"

class Matching {
public:
    enum class MatchType {
        Ok,
        Unknown
    };

    struct PotMatch {
        PotMatch() {}

        PotMatch(int plane1,
                 int plane2)
                : plane1(plane1),
                  plane2(plane2) {}

        PotMatch(int plane1,
                 int plane2,
                 double planeAppDiff)
                : plane1(plane1),
                  plane2(plane2),
                  planeAppDiff(planeAppDiff) {}

        int plane1;

        int plane2;

        double planeAppDiff;
    };

    struct ValidTransform {
        ValidTransform() {}

        ValidTransform(const Vector7d &itransform,
                       const std::vector<Matching::PotMatch> &imatchSet,
                       const double &iresError)
                : transform(itransform),
                  matchSet(imatchSet),
                  resError(iresError),
                  weight(0.0) {}

        Vector7d transform;
        std::vector<Matching::PotMatch> matchSet;
        double resError;
        double weight;
    };

    static MatchType matchLocalToGlobal(const cv::FileStorage &fs,
                                        Map &globalMap,
                                        Map &localMap,
                                        vectorVector7d &bestTrans,
                                        std::vector<double> &bestTransProbs,
                                        std::vector<double> &bestTransMatchedRatio,
                                        std::vector<int> &bestTransDistinct,
                                        std::vector<Matching::ValidTransform> &retTransforms,
                                        pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                        int viewPort1 = -1,
                                        int viewPort2 = -1,
                                        const Eigen::Matrix4d &refTrans = Eigen::Matrix4d::Zero());


    static std::vector<ObjInstanceView::ConstPtr> getVisibleObjs(const std::vector<ObjInstanceView::ConstPtr> &viewsLocalTrans,
                                                            const Map &map,
                                                            const Eigen::Matrix3d &cameraMatrix,
                                                            int rows,
                                                            int cols,
                                                            int &nPlaneIds,
                                                            double &areaViews);

    class ProbDistKernel {
    public:
        ProbDistKernel(const Vector7d &ikPt,
                       const Eigen::Matrix<double, 6, 6> &iinfMat,
                       const double &iweight);

        double eval(const Vector7d &pt) const;

        double eval(const Eigen::Matrix4d &ptInvMat) const;

        double getW() const {
            return weight;
        };

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        Vector7d kPt;

        Eigen::Matrix4d kPtMat;

        Eigen::Matrix<double, 6, 6> infMat;

        // Eigen::Matrix<double, 6, 6> covMat;
        double denom;

        double weight;
    };

    typedef std::vector<Matching::ProbDistKernel, Eigen::aligned_allocator<Matching::ProbDistKernel> > vectorProbDistKernel;

    static double evalPoint(const Vector7d &pt,
                            const vectorProbDistKernel &dist);

    static double evalPoint(const Eigen::Matrix4d &ptInvMat,
                            const vectorProbDistKernel &dist);

private:


    static double compAngleDiffBetweenNormals(const Eigen::Vector3d &nf1,
                                              const Eigen::Vector3d &ns1,
                                              const Eigen::Vector3d &nf2,
                                              const Eigen::Vector3d &ns2);

    static bool checkPlaneToPlaneAng(const vectorVector4d &planes1,
                                     const vectorVector4d &planes2,
                                     double planeToPlaneAngThresh);


    static std::vector<std::vector<PotMatch> > findPotSets(const std::vector<PotMatch> &potMatches,
                                                           const std::vector<ObjInstanceView::ConstPtr> &globalViews,
                                                           const std::vector<ObjInstanceView::ConstPtr> &localViews,
                                                           double planeDistThresh,
                                                           double planeToPlaneAngThresh,
                                                           double planeToPlaneDistThresh,
                                                           pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                                           int viewPort1 = -1,
                                                           int viewPort2 = -1);


    static void compObjDistances(const std::vector<ObjInstanceView::ConstPtr> &views,
                                 std::vector<std::vector<double>> &objDistances);


    static Vector7d bestTransformObjs(const std::vector<ObjInstanceView::ConstPtr> &objs1,
                                      const std::vector<ObjInstanceView::ConstPtr> &objs2,
                                      double &resError,
                                      bool &fullConstr,
                                      double &svdr,
                                      double &svdt,
                                      bool debug = false);
};

#endif /* INCLUDE_MATCHING_HPP_ */
