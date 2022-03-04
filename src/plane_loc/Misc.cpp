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

#include <cmath>

#include "Misc.hpp"

#define PRINT(x) std::cout << #x << " = " << std::endl << x << std::endl

using namespace std;
using namespace cv;

cv::Mat Misc::projectTo3D(cv::Mat depth, cv::Mat cameraParams){
	float fx = cameraParams.at<float>(0, 0);
	float fy = cameraParams.at<float>(1, 1);
	float cx = cameraParams.at<float>(0, 2);
	float cy = cameraParams.at<float>(1, 2);
//	cout << "cx = " << cx << ", cy = " << cy << ", fx = " << fx << ", fy = " << fy << endl;
	Mat xyz(depth.rows, depth.cols, CV_32FC3);
	for(int row = 0; row < depth.rows; ++row){
		for(int col = 0; col < depth.cols; ++col){
			float d = depth.at<float>(row, col);

			// x
			xyz.at<Vec3f>(row, col)[0] = (col - cx) * d / fx;
			// y
			xyz.at<Vec3f>(row, col)[1] = (row - cy) * d / fy;
			// z
			xyz.at<Vec3f>(row, col)[2] = d;

//			cout << "d = " << d <<
//					", x = " << xyz.at<Vec3f>(row, col)[0] <<
//					", y = " << xyz.at<Vec3f>(row, col)[1] <<
//					", z = " << xyz.at<Vec3f>(row, col)[2] << endl;
		}
	}
	return xyz;
}

cv::Mat Misc::reprojectTo2D(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr points, cv::Mat cameraParams)
{
    float fx = cameraParams.at<float>(0, 0);
    float fy = cameraParams.at<float>(1, 1);
    float cx = cameraParams.at<float>(0, 2);
    float cy = cameraParams.at<float>(1, 2);
    Mat uvd(1, points->size(), CV_32FC3);
    for(int p = 0; p < points->size(); ++p){
        double d = points->at(p).z;

        uvd.at<Vec3f>(0, p)[0] = points->at(p).x * fx / d + cx;
        uvd.at<Vec3f>(0, p)[1] = points->at(p).y * fy / d + cy;
        uvd.at<Vec3f>(0, p)[2] = d;
    }
    return uvd;
}

cv::Mat Misc::reprojectTo2D(const Eigen::MatrixPt &points, const Eigen::Matrix3d &cameraParams)
{
    float fx = cameraParams(0, 0);
    float fy = cameraParams(1, 1);
    float cx = cameraParams(0, 2);
    float cy = cameraParams(1, 2);
    Mat uvd(1, points.cols(), CV_32FC3);
    for(int p = 0; p < points.cols(); ++p){
        double d = points(2, p);

        uvd.at<Vec3f>(0, p)[0] = points(0, p) * fx / d + cx;
        uvd.at<Vec3f>(0, p)[1] = points(1, p) * fy / d + cy;
        uvd.at<Vec3f>(0, p)[2] = d;
    }
    return uvd;
}

Eigen::Vector3d Misc::projectPointOnPlane(const Eigen::Vector3d &pt, const Eigen::Vector4d &plane)
{
    Eigen::Vector4d planeNorm = toNormalPlaneEquation(plane);
    Eigen::Vector3d n = planeNorm.head<3>();
    double d = -planeNorm[3];
    Eigen::Vector3d plPt = n * d;
    double ndist = n.dot(pt - plPt);
    return pt - ndist * n;
}

Eigen::Vector3d Misc::projectPointOnPlane(const Eigen::Vector2d &pt, const Eigen::Vector4d &plane, cv::Mat cameraMatrix)
{
//    cout << "pt = " << pt.transpose() << endl;
//    cout << "plane = " << plane.transpose() << endl;
    float fx = cameraMatrix.at<float>(0, 0);
    float fy = cameraMatrix.at<float>(1, 1);
    float cx = cameraMatrix.at<float>(0, 2);
    float cy = cameraMatrix.at<float>(1, 2);
    Eigen::Vector4d planeNorm = toNormalPlaneEquation(plane);
    Eigen::Vector3d n = planeNorm.head<3>();
    double dpl = planeNorm[3];
//    cout << "n = " << n.transpose() << endl;
//    cout << "dpl = " << dpl << endl;
    Eigen::Vector3d pnnorm;
    pnnorm[0] = (pt[0] - cx) / fx;
    pnnorm[1] = (pt[1] - cy) / fy;
    pnnorm[2] = 1;
//    cout << "pnnorm = " << pnnorm.transpose() << endl;
    double ncast = n.dot(pnnorm);
//    cout << "ncast = " << ncast << endl;
    if(std::abs(ncast) > 1e-6) {
        double d = -dpl / ncast;
//        cout << "d = " << d << endl;
//        cout << "pnnorm * d = " << (pnnorm * d).transpose() << endl;
        return pnnorm * d;
    }
    else{
        return Eigen::Vector3d::Zero();
    }
}

Eigen::MatrixPt Misc::projectImagePointsOntoPlane(const Eigen::MatrixPt &pts,
                                                  const Eigen::Vector3d &ptsOrig,
                                                  const Eigen::Vector4d &planeEq)
{
    static constexpr double eps = 1.0e-4;
    static constexpr double minRange = 0.2;
    static constexpr double maxRange = 15.0;

    Eigen::Vector4d planeEqNorm = toNormalPlaneEquation(planeEq);
    Eigen::Vector3d nPl = planeEqNorm.head<3>();
    double dPl = -planeEqNorm(3);

    Eigen::MatrixPt ptsProj(pts.rows(), pts.cols());
    for (int pt = 0; pt < pts.cols(); ++pt) {
        Eigen::Vector3d ray = pts.col(pt).head<3>() - ptsOrig;
        ray.normalize();
        double rayInN = nPl.dot(ray);
        double origInN = nPl.dot(ptsOrig);

        if (std::abs(rayInN) < eps) {
            rayInN = rayInN < 0.0 ? -eps : eps;
        }

        double t = (dPl - origInN) / rayInN;
        t = std::max(std::min(t, maxRange), minRange);

        ptsProj.col(pt) << ptsOrig + ray * t, 1.0;
    }

    return ptsProj;
}

bool Misc::projectImagePointsOntoPlane(const vectorVector2d &pts,
                                       vectorVector3d &pts3d,
                                       const Eigen::Matrix3d &cameraMatrix,
                                       const Eigen::Vector4d &planeEq)
{
    double fx = cameraMatrix(0, 0);
    double fy = cameraMatrix(1, 1);
    double cx = cameraMatrix(0, 2);
    double cy = cameraMatrix(1, 2);

    return projectImagePointsOntoPlane(pts,
            pts3d,
            fx, fy, cx, cy,
            planeEq);
}

bool Misc::projectImagePointsOntoPlane(const vectorVector2d &pts,
                                       vectorVector3d &pts3d,
                                       const double &fx,
                                       const double &fy,
                                       const double &cx,
                                       const double &cy,
                                       const Eigen::Vector4d &planeEq)
{
    static constexpr double eps = 1e-6;

    
    // camera center in homogeneous cooridinates
    Eigen::Vector4d C;
    C << 0, 0, 0, 1;
    
    double den = planeEq.transpose() * C;
//    cout << "den = " << den << endl;
    
    // plane through camera center
    if(abs(den) < eps){
        return false;
    }
    else{
        // pseudoinverse of the camera matrix P
        Eigen::Matrix<double, 4, 3> Ppinv;
        Ppinv <<    1/fx,   0, -cx/fx,
                    0,   1/fy, -cy/fy,
                    0,      0,      1,
                    0,      0,      0;
        
        Eigen::Matrix<double, 1, 3> pPpinv = planeEq.transpose() * Ppinv;
//        cout << "pPinv = " << pPpinv << endl;
        
        for(const Eigen::Vector2d &pt2d : pts) {
            // point in homogeneous cooridinates
            Eigen::Vector3d pt;
            pt << pt2d(0), pt2d(1), 1;
            
//            cout << "pt = " << pt.transpose() << endl;
            
            double lambda = - (pPpinv * pt)(0) / den;
            
//            cout << "lambda = " << lambda << endl;
            
//            // if 0 or negative
//            if(lambda < eps){
//                pts3d.clear();
//                return false;
//            }
            Eigen::Vector4d pt3d = Ppinv * pt + lambda * C;
            
//            cout << "pt3d = " << pt3d.transpose() << endl;
            
            // adding point in inhomogeneous coordinates
            pts3d.push_back(pt3d.head<3>()/pt3d(3));
        }
        return true;
    }
}


bool Misc::nextChoice(std::vector<int>& choice, int N)
{
	int chidx = choice.size() - 1;
	int chendidx = 0;
	bool valid = false;
	while(chidx >= 0 && !valid){
		++choice[chidx];
		// if in valid range, we found correct choice
		if(choice[chidx] < N - chendidx){
			valid = true;
		}
		--chidx;
		++chendidx;
	}
	// moving all choice values after last value incremented
	if(valid){
		// move chidx back to last value incremented
		++chidx;
		while(chidx < choice.size() - 1){
			choice[chidx + 1] = choice[chidx] + 1;
			++chidx;
		}
	}
	return valid;
}

Eigen::Quaterniond Misc::planeEqToQuat(const Eigen::Vector4d &planeEq)
{
	Eigen::Quaterniond ret(planeEq(3), planeEq(0), planeEq(1), planeEq(2));
	normalizeAndUnify(ret);
	return ret;
}

void Misc::normalizeAndUnify(Eigen::Quaterniond& q){
	q.normalize();
	static constexpr double eps = 1e-9;
	if(q.w() < 0.0 ||
		(fabs(q.w()) < eps && q.z() < 0.0) ||
		(fabs(q.w()) < eps && fabs(q.z()) < eps && q.y() < 0.0) ||
		(fabs(q.w()) < eps && fabs(q.z()) < eps && fabs(q.y()) < eps && q.x() < 0.0))
	{
		q.coeffs() = -q.coeffs();
	}
}

void Misc::normalizeAndUnify(Eigen::Vector4d& v){
	v.normalize();
	static constexpr double eps = 1e-9;
	if(v(3) < 0.0 ||
		(fabs(v(3)) < eps && v(2) < 0.0) ||
		(fabs(v(3)) < eps && fabs(v(2)) < eps && v(1) < 0.0) ||
		(fabs(v(3)) < eps && fabs(v(2)) < eps && fabs(v(1)) < eps && v(0) < 0.0))
	{
		v = -v;
	}
}

Eigen::Vector4d Misc::toNormalPlaneEquation(const Eigen::Vector4d &plane)
{
    double nnorm = plane.head<3>().norm();
    return plane / nnorm;
}

Eigen::Vector3d Misc::logMap(const Eigen::Quaterniond &quat)
{
//     Eigen::Quaterniond lquat = quat;
// 	Eigen::Vector3d res;
//
//     normalizeAndUnify(lquat);
// 	double qvNorm = sqrt(lquat.x()*lquat.x() + lquat.y()*lquat.y() + lquat.z()*lquat.z());
// 	if(qvNorm > 1e-6){
// 		res[0] = lquat.x()/qvNorm;
// 		res[1] = lquat.y()/qvNorm;
// 		res[2] = lquat.z()/qvNorm;
// 	}
// 	else{
// 		// 1/sqrt(3), so norm = 1
// 		res[0] = 0.57735026919;
// 		res[1] = 0.57735026919;
// 		res[2] = 0.57735026919;
// 	}
// 	double acosQw = acos(lquat.w());
// 	res *= 2.0*acosQw;
// //		cout << "2.0*acosQw = " << 2.0*acosQw << endl;
// 	return res;

    Eigen::AngleAxisd ax(quat);
    Eigen::Vector3d res = ax.axis() * ax.angle();

    return res;
}

Eigen::Quaterniond Misc::expMap(const Eigen::Vector3d &vec)
{
//     double arg = 0.5 * std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
// //    if(fabs(arg) > 0.5 * pi){
// //        std::cout << "arg = " << arg << std::endl;
// //        char a;
// //        std::cin >> a;
// //    }
//     double sincArg = 1.0;
//     if(arg > 1e-6){
//         sincArg = sin(arg)/arg;
//     }
//     else{
//         //taylor expansion
//         sincArg = 1 - arg*arg/6 + pow(arg, 4)/120;
//     }
//     double cosArg = cos(arg);
//
//     Eigen::Quaterniond res(cosArg,
//                          0.5*sincArg*vec[0],
//                          0.5*sincArg*vec[1],
//                          0.5*sincArg*vec[2]);
//
//     normalizeAndUnify(res);
//
//     return res;

    Eigen::Vector3d axis = vec.normalized();
    if (vec.norm() < 1e-15) {
        axis = Eigen::Vector3d::UnitX();
    }

    Eigen::AngleAxisd ax(vec.norm(), axis);

    return Eigen::Quaterniond(ax);
}

// From https://github.com/strasdat/Sophus/blob/master/sophus/se3.hpp
Vector6d Misc::logMap(const Eigen::Matrix4d &T) {
    using std::abs;
    using std::cos;
    using std::sin;
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);

    Vector6d upsilon_omega;

    Eigen::Vector3d omega = Misc::logMap(Eigen::Quaterniond(R));

    double theta = omega.norm();
    upsilon_omega.tail<3>() = omega;
    Eigen::Matrix3d Omega = Misc::skew(omega);

    static constexpr double eps = 1e-10;
    if (abs(theta) < eps) {
        Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity() -
                                      0.5 * Omega +
                                      (1. / 12.) * (Omega * Omega);

        upsilon_omega.head<3>() = V_inv * t;
    } else {
        double half_theta = 0.5 * theta;

        Eigen::Matrix3d V_inv =
                (Eigen::Matrix3d::Identity() - 0.5 * Omega +
                 (1.0 - theta * cos(half_theta) / (2.0 * sin(half_theta))) /
                 (theta * theta) * (Omega * Omega));
        upsilon_omega.template head<3>() = V_inv * t;
    }
    return upsilon_omega;
}

Eigen::Matrix4d Misc::expMap(const Vector6d &vec) {
    using std::cos;
    using std::sin;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    Eigen::Vector3d omega = vec.tail<3>();

    double theta = omega.norm();
    Eigen::Matrix3d R = Misc::expMap(omega).toRotationMatrix();
    Eigen::Matrix3d Omega = Misc::skew(omega);
    Eigen::Matrix3d Omega_sq = Omega * Omega;
    Eigen::Matrix3d V;

    static constexpr double eps = 1e-10;
    if (theta < eps) {
        V = R;
        /// Note: That is an accurate expansion!
    } else {
        double theta_sq = theta * theta;
        V = (Eigen::Matrix3d::Identity() +
             (1.0 - cos(theta)) / (theta_sq)*Omega +
             (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
    }
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = V * vec.head<3>();
    return T;
}

Eigen::Matrix4d Misc::matrixQ(const Eigen::Quaterniond &q)
{
	Eigen::Matrix4d ret;
	ret.block<3, 3>(0, 0) = matrixK(q) + Eigen::Matrix3d::Identity() * q.w();
	ret.block<3, 1>(0, 3) = q.vec();
	ret.block<1, 3>(3, 0) = -q.vec();
	ret.block<1, 1>(3, 3) = Eigen::Matrix<double, 1, 1>::Ones() * q.w();
	return ret;
}

Eigen::Matrix4d Misc::matrixW(const Eigen::Quaterniond &q)
{
	Eigen::Matrix4d ret;
	ret.block<3, 3>(0, 0) = -matrixK(q) + Eigen::Matrix3d::Identity() * q.w();
	ret.block<3, 1>(0, 3) = q.vec();
	ret.block<1, 3>(3, 0) = -q.vec();
	ret.block<1, 1>(3, 3) = Eigen::Matrix<double, 1, 1>::Ones() * q.w();
	return ret;
}

Eigen::Matrix3d Misc::matrixK(const Eigen::Quaterniond &q)
{
	Eigen::Matrix3d ret;
	ret <<	0,		-q.z(),	q.y(),
			q.z(),	0,		-q.x(),
			-q.y(),	q.x(),	0;
	return ret;
}

Eigen::Matrix3d Misc::skew(const Eigen::Vector3d v) {
    Eigen::Matrix3d ret;
    ret <<	0,		-v(2),	v(1),
            v(2),	0,		-v(0),
            -v(1),	v(0),	0;
    return ret;
}

Vector7d Misc::toVector(const Eigen::Matrix4d &T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);

    return toVector(R, t);
}

Vector7d Misc::toVector(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    Vector7d v;
    v.head<3>() = t;
    v.tail<4>() = Eigen::Quaterniond(R).normalized().coeffs();

    return v;
}

Eigen::Matrix4d Misc::toMatrix(const Vector7d &v) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = v.head<3>();
    T.block<3, 3>(0, 0) = Eigen::Quaterniond(v(6), v(3), v(4), v(5)).normalized().toRotationMatrix();

    return T;
}

Eigen::Matrix4d Misc::inverseTrans(const Eigen::Matrix4d &mat) {
    Eigen::Matrix4d inv = Eigen::Matrix4d::Identity();
    inv.block<3, 3>(0, 0) = mat.block<3, 3>(0, 0).transpose();
    inv.block<3, 1>(0, 3) = -inv.block<3, 3>(0, 0) * mat.block<3, 1>(0, 3);
//    inv(3, 3) = 1.0;

    return inv;
}

bool Misc::checkIfAlignedWithNormals(const Eigen::Vector3d& testedNormal,
									  pcl::PointCloud<pcl::Normal>::ConstPtr normals,
									  bool& alignConsistent)
{
    bool isAligned = false;
    alignConsistent = false;

    // check if aligned with majority of normals
    int alignedCnt = 0;
    for(int p = 0; p < normals->size(); ++p){
        Eigen::Vector3d curNorm = normals->at(p).getNormalVector3fMap().cast<double>();
        if(testedNormal.dot(curNorm) > 0){
            ++alignedCnt;
        }
    }
    double alignedFrac = (double)alignedCnt / normals->size();
//    cout << "alignedFrac = " << alignedFrac << endl;
    static constexpr double errorThresh = 0.1;
    if(alignedFrac >= 0.5){
        isAligned = true;
    }
    else if(alignedFrac < 0.5){
        alignConsistent = false;
        isAligned = false;
    }

    // some normals aligned and some not - something went wrong
    if(min(alignedFrac, 1.0 - alignedFrac) > errorThresh) {
        alignConsistent = true;
    }

    return isAligned;
}

double Misc::transformLogDist(const Vector7d &trans1, const Vector7d &trans2) {
    Eigen::Matrix4d trans = Misc::toMatrix(trans1);
    Eigen::Matrix4d transComp = Misc::toMatrix(trans2);
	Eigen::Matrix4d diff = trans.inverse() * transComp;
	Vector6d logMapDiff = Misc::logMap(diff);
	double dist = logMapDiff.transpose() * logMapDiff;
	return dist;
}

double Misc::rotLogDist(const Eigen::Vector4d &rot1, const Eigen::Vector4d &rot2) {
    Eigen::Quaterniond r1(rot1[3], rot1[0], rot1[1], rot1[2]);
    Eigen::Quaterniond r2(rot2[3], rot2[0], rot2[1], rot2[2]);
    Eigen::Vector3d logDiff = logMap(r1.inverse() * r2);

    return logDiff.transpose() * logDiff;
}

cv::Mat Misc::colorIds(cv::Mat ids) {
    cv::Mat colIm(ids.size(), CV_8UC3, Scalar(0, 0, 0));
    for(int r = 0; r < ids.rows; ++r){
        for(int c = 0; c < ids.cols; ++c){
            int id = ids.at<int>(r, c);
            if(id >= 0){
                int colIdx = (id % (sizeof(colors)/sizeof(uint8_t)/3));
                colIm.at<Vec3b>(r, c) = cv::Vec3b(colors[colIdx][2],
                                                  colors[colIdx][1],
                                                  colors[colIdx][0]);
            }
        }
    }
    return colIm;
}

cv::Mat Misc::colorIdsWithLabels(cv::Mat ids) {
    cv::Mat colIm(ids.size(), CV_8UC3, Scalar(0, 0, 0));
    map<int, tuple<float, float, int>> centroids;
    for(int r = 0; r < ids.rows; ++r){
        for(int c = 0; c < ids.cols; ++c){
            int id = ids.at<int>(r, c);
            if(id >= 0){
                int cntId = centroids.count(id);
                auto &ct = centroids[id];
                if(cntId == 0){
                    ct = make_tuple(0.0f, 0.0f, 0);
                }
                get<0>(ct) += r;
                get<1>(ct) += c;
                get<2>(ct) += 1;
                
                int colIdx = (id % (sizeof(colors)/sizeof(uint8_t)/3));
                colIm.at<Vec3b>(r, c) = cv::Vec3b(colors[colIdx][2],
                                                  colors[colIdx][1],
                                                  colors[colIdx][0]);
            }
        }
    }
    for(auto &c : centroids){
        int id = c.first;
        auto &ct = c.second;
        float y = get<0>(ct) / get<2>(ct);
        float x = get<1>(ct) / get<2>(ct);
        cv::putText(colIm,
                    to_string(id),
                    cv::Point(x, y),
                    cv::FONT_HERSHEY_PLAIN,
                    0.5,
                    cv::Scalar(255, 255, 255));
    }
    return colIm;
}


Eigen::Vector3d Misc::closestPointOnLine(const Eigen::Vector3d &pt,
                                         const Eigen::Vector3d &p,
                                         const Eigen::Vector3d &n)
{
	static constexpr double eps = 1e-6;
	double nnorm = n.norm();
	if(nnorm > eps) {
		double t = (pt - p).dot(n) / (nnorm * nnorm);
//        cout << "pt = " << pt.transpose() << endl;
//        cout << "(" << p.transpose() << ") + " << t << " * (" << n.transpose() << ") = (" << (p + t * n).transpose() << ")" << endl;
		return p + t * n;
	}
	else{
		return Eigen::Vector3d::Zero();
	}
}

// at::Tensor Misc::solve(const at::Tensor &A, const at::Tensor &b) {
//     at::Tensor U, S, V_t;
//     std::tie(U, S, V_t) = torch::linalg_svd(A, false);
//
//     at::Tensor b_prim = torch::matmul(U.transpose(1, 2), b);
//     at::Tensor y = b_prim / torch::clamp(S.unsqueeze(2), 1.0e-3f);
//     y = torch::where(S.unsqueeze(2) > (float)1.0e-3f, y, torch::zeros_like(y));
//     at::Tensor x = torch::matmul(V_t.transpose(1, 2), y);
//
//     return x;
// }
//
// at::Tensor Misc::smallestSingularVal(const at::Tensor &A) {
//     using namespace torch::indexing;
//
//     at::Tensor U, S, V_t;
//     std::tie(U, S, V_t) = torch::linalg_svd(A, false);
//
//     return S.index({Slice(), Slice(-1, None)});
// }
//
// at::Tensor Misc::orthonormalize(const at::Tensor &R) {
//     at::Tensor U, S, V_t;
//     std::tie(U, S, V_t) = torch::linalg_svd(R, false);
//
//     at::Tensor R_o = torch::matmul(U, V_t);
//
//     return R_o;
// }
//
// // based on https://github.com/utiasSTARS/liegroups/blob/master/liegroups/torch/so3.py
// at::Tensor Misc::skew(const at::Tensor &v) {
//     using namespace torch::indexing;
//
//     auto bsize = v.size(0);
//     at::Tensor S = at::zeros({bsize, 3, 3}, v.options());
//
//     S.index_put_({Slice(), 0, 1}, -v.index({Slice(), 2}));
//     S.index_put_({Slice(), 1, 0}, v.index({Slice(), 2}));
//     S.index_put_({Slice(), 0, 2}, v.index({Slice(), 1}));
//     S.index_put_({Slice(), 2, 0}, -v.index({Slice(), 1}));
//     S.index_put_({Slice(), 1, 2}, -v.index({Slice(), 0}));
//     S.index_put_({Slice(), 2, 1}, v.index({Slice(), 0}));
//
//     return S;
// }
//
// at::Tensor Misc::expMapSO3(const at::Tensor &om) {
//     using namespace torch::indexing;
//
//     auto bsize = om.size(0);
//     at::Tensor R = at::empty({bsize, 3, 3}, om.options());
//
//     at::Tensor angle = om.norm(2, 1);
//
//     auto small_angle_mask = angle < 1.0e-6f;
//     auto small_angle_idxs = small_angle_mask.nonzero().squeeze_(1);
//     int64_t n_sa = small_angle_idxs.size(0);
//     if (n_sa > 0) {
//         R.index_put_({small_angle_idxs},
//                      at::eye(3, om.options()).expand({n_sa, 3, 3}) + Misc::skew(om.index({small_angle_idxs})));
//     }
//
//     auto large_angle_mask = small_angle_mask.logical_not();
//     auto large_angle_idxs = large_angle_mask.nonzero().squeeze_(1);
//     int64_t n_la = large_angle_idxs.size(0);
//     if (n_la > 0) {
//         at::Tensor large_angle = angle.index({large_angle_idxs});
//         // PRINT(om.index({large_angle_idxs}).sizes());
//         // PRINT(large_angle.view({-1, 1}).sizes());
//         at::Tensor axis = om.index({large_angle_idxs}) / large_angle.view({-1, 1});
//
//         at::Tensor s = large_angle.sin().view({-1, 1, 1});
//         at::Tensor c = large_angle.cos().view({-1, 1, 1});
//
//         at::Tensor A = c * at::eye(3, om.options()).expand({n_la, 3, 3});
//         at::Tensor B = (1.0f - c) * torch::matmul(axis.view({n_la, 3, 1}), axis.view({n_la, 1, 3}));
//         at::Tensor C = s * Misc::skew(axis);
//
//         R.index_put_({large_angle_idxs}, A + B + C);
//     }
//
//     return R;
// }
