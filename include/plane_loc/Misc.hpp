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

#ifndef INCLUDE_MISC_HPP_
#define INCLUDE_MISC_HPP_

#include <ostream>
#include <vector>

#include <Eigen/Eigen>

#include <pcl/common/common_headers.h>
#include <pcl/impl/point_types.hpp>

#include <opencv2/opencv.hpp>

// #include <ATen/ATen.h>

#include "Types.hpp"

static constexpr float pi = 3.14159265359;

template<class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec){
	out << "[";
	for(int v = 0; v < (int)vec.size(); ++v){
		out << vec[v];
		if(v < vec.size() - 1){
			out << ", ";
		}
	}
	out << "]";

	return out;
}

class Misc{
public:

	static cv::Mat projectTo3D(cv::Mat depth, cv::Mat cameraParams);

    static cv::Mat reprojectTo2D(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr points, cv::Mat cameraParams);

    static cv::Mat reprojectTo2D(const Eigen::MatrixPt &points, const Eigen::Matrix3d &cameraParams);

    static Eigen::Vector3d projectPointOnPlane(const Eigen::Vector3d &pt, const Eigen::Vector4d &plane);

    static Eigen::Vector3d projectPointOnPlane(const Eigen::Vector2d &pt, const Eigen::Vector4d &plane, cv::Mat cameraMatrix);

	static bool projectImagePointsOntoPlane(const vectorVector2d &pts,
										    vectorVector3d &pts3d,
                                            const Eigen::Matrix3d &cameraMatrix,
										    const Eigen::Vector4d &planeEq);

    static Eigen::MatrixPt projectImagePointsOntoPlane(const Eigen::MatrixPt &pts,
                                                       const Eigen::Vector3d &ptsOrig,
                                                       const Eigen::Vector4d &planeEq);

    static bool projectImagePointsOntoPlane(const vectorVector2d &pts,
                                            vectorVector3d &pts3d,
                                            const double &fx,
                                            const double &fy,
                                            const double &cx,
                                            const double &cy,
                                            const Eigen::Vector4d &planeEq);
	
	static bool nextChoice(std::vector<int>& choice, int N);

	static Eigen::Quaterniond planeEqToQuat(const Eigen::Vector4d &planeEq);

	static void normalizeAndUnify(Eigen::Quaterniond& q);

	static void normalizeAndUnify(Eigen::Vector4d& q);

    static Eigen::Vector4d toNormalPlaneEquation(const Eigen::Vector4d &plane);

	static Eigen::Vector3d logMap(const Eigen::Quaterniond &quat);

	static Eigen::Quaterniond expMap(const Eigen::Vector3d &vec);

    static Vector6d logMap(const Eigen::Matrix4d &T);

    static Eigen::Matrix4d expMap(const Vector6d &vec);

	static Eigen::Matrix4d matrixQ(const Eigen::Quaterniond &q);

	static Eigen::Matrix4d matrixW(const Eigen::Quaterniond &q);

	static Eigen::Matrix3d matrixK(const Eigen::Quaterniond &q);

	static Eigen::Matrix3d skew(const Eigen::Vector3d v);

	static Vector7d toVector(const Eigen::Matrix4d &T);

    static Vector7d toVector(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);

    static Eigen::Matrix4d toMatrix(const Vector7d &v);

    static Eigen::Matrix4d inverseTrans(const Eigen::Matrix4d &mat);

	static bool checkIfAlignedWithNormals(const Eigen::Vector3d& testedNormal,
                                            pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                            bool& alignConsistent);

	static double transformLogDist(const Vector7d &trans1,
								   const Vector7d &trans2);

    static double rotLogDist(const Eigen::Vector4d &rot1,
							 const Eigen::Vector4d &rot2);

    static cv::Mat colorIds(cv::Mat ids);
	
	static cv::Mat colorIdsWithLabels(cv::Mat ids);
	
	static Eigen::Vector3d closestPointOnLine(const Eigen::Vector3d &pt,
									   const Eigen::Vector3d &p,
									   const Eigen::Vector3d &n);
    
    template<typename MatrixTypeOut, typename MatrixTypeIn>
    static MatrixTypeOut pseudoInverse(const MatrixTypeIn &a, double epsilon = std::numeric_limits<double>::epsilon())
    {
        Eigen::JacobiSVD< MatrixTypeIn > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
//        return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
        
        typename Eigen::JacobiSVD< MatrixTypeIn >::SingularValuesType singularValues_inv = svd.singularValues();
        for ( long i = 0; i < singularValues_inv.cols(); ++i) {
            if ( fabs(svd.singularValues()(i)) > tolerance ) {
                singularValues_inv(i) = 1.0 / svd.singularValues()(i);
            }
            else{
                singularValues_inv(i)=0;
            }
        }
        return (svd.matrixV() * singularValues_inv.asDiagonal());
    }
};

static constexpr uint8_t colors[][3] = {
		{0xFF, 0x00, 0x00}, //Red
		{0xFF, 0xFF, 0xFF}, //White
		{0x00, 0xFF, 0xFF}, //Cyan
		{0xC0, 0xC0, 0xC0}, //Silver
		{0x00, 0x00, 0xFF}, //Blue
		{0x80, 0x80, 0x80}, //Gray
		{0x00, 0x00, 0xA0}, //DarkBlue
		{0x00, 0x00, 0x00}, //Black
		{0xAD, 0xD8, 0xE6}, //LightBlue
		{0xFF, 0xA5, 0x00}, //Orange
		{0x80, 0x00, 0x80}, //Purple
		{0xA5, 0x2A, 0x2A}, //Brown
		{0xFF, 0xFF, 0x00}, //Yellow
		{0x80, 0x00, 0x00}, //Maroon
		{0x00, 0xFF, 0x00}, //Lime
		{0x00, 0x80, 0x00}, //Green
		{0xFF, 0x00, 0xFF}, //Magenta
		{0x80, 0x80, 0x00} //Olive
};

class Visualizer{
public:

//	static pcl::PointCloud<pcl::PointXYZRGBA>::Ptr makeColorPointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr
};

#endif /* INCLUDE_MISC_HPP_ */
