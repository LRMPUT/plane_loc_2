//
// Created by jachu on 17.05.18.
//

#ifndef LOAM_VELODYNE_POSEMISC_HPP
#define LOAM_VELODYNE_POSEMISC_HPP

#include <Eigen/Dense>
#include <vector>

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<float, 7, 1> Vector7f;

typedef std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d> > vectorOfEigenVector4d;
typedef std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > vectorOfEigenVector3d;

class PoseMisc {
public:
    /**
     * Converts pose stored as orientation in the Euler angles (RPY) and position (XYZ) to the 4x4
     * homogeneous transformation matrix.
     * @param[in] transRpy pose stored as orientation in the Euler angles (RPY) and position (XYZ).
     * @return homogeneous transformation matrix.
     */
    static Eigen::Matrix4d toMatrix(float transRpy[6]);
    
    static Eigen::Matrix3d toMatrixRot(float rpy[3]);
    
    /**
     * Takes the 4x4 homogeneous transformation matrix and returns the pose as orientation in
     * the Euler angles (RPY) and position (XYZ) in the given an array.
     * @param[in] mat homogeneous transformation matrix
     * @param[out] transRpy pose as orientation in the Euler angles (RPY) and position (XYZ).
     */
    static void toTransRpy(const Eigen::Matrix4d &mat,
                    float transRpy[6]);
    
    /**
     * Takes the 3x3 rotation matrix and returns the pose as orientation in
     * the Euler angles (RPY).
     * @param[in] R rotation matrix
     * @return orientation in the Euler angles (RPY).
     */
    static Eigen::Vector3d toRpy(const Eigen::Matrix3d &R);
    
    /**
     * Returns the inverse of the 4x4 homogeneous transformation matrix taking into account that R^T = R^-1
     * @param[in] mat homogeneous transformation matrix.
     * @return inverse of the transformation matrix.
     */
    static Eigen::Matrix4d inverseTrans(const Eigen::Matrix4d &mat);


    static Vector7d toVector(const Eigen::Matrix4d &mat);

    static Eigen::Matrix4d toMatrix(const Vector7d &v);

    /**
     * The cross product matrix of the vector \p v (it is skew-symmetric).
     * @param[in] v input vector (orientation in angle-axis representation or point).
     * @return skew-symmetric cross product matrix of the \p v.
     */
    static Eigen::Matrix3d skew(const Eigen::Vector3d &v);

    /**
     * 
     * @param R
     * @return
     */
    static Eigen::Vector3d vee(const Eigen::Matrix3d &R);
    
    /**
     * Computes the Lie algebra representation of the SE(3) transformation given as 4x4 homogeneous matrix
     * @param[in] m homogeneous transformation matrix
     * @return pose in the Lie algebra representation (as 6-D vector with 3 components of orientation first).
     */
    static Vector6d log(const Eigen::Matrix4d &m);
    
    /**
     * Computes the 4x4 homogeneous transformation matrix from the Lie algebra representation.
     * @param[in] u pose in the Lie algebra representation (as 6-D vector with 3 components of orientation first).
     * @return homogeneous transformation matrix
     */
    static Eigen::Matrix4d exp(const Vector6d &u);
    
    /**
     * The method that takes the 3x3 rotation matrix R and returns the off-diagonal entries of the difference R - R^T.
     * @param[in] R rotation matrix.
     * @return off-diagonal entries of the difference R - R^T.
     */
    static Eigen::Vector3d deltaR(const Eigen::Matrix3d &R);
    
    /**
     * Computes the Lie algebra representation of the SO(3) transformation given as 3x3 rotation matrix
     * @param[in] R rotation matrix
     * @return pose in the Lie algebra representation (as 3-D vector).
     */
    static Eigen::Vector3d log(const Eigen::Matrix3d &R);
    
    /**
     * Computes the 3x3 rotation matrix from the Lie algebra representation.
     * @param[in] omega pose in the Lie algebra representation (as 3-D vector).
     * @return rotation matrix
     */
    static Eigen::Matrix3d exp(const Eigen::Vector3d &omega);

    /**
     * Computes unnormalized cardinal sine function sin(x)/x.
     * @param x input argument for the function in radians.
     * @return computed value.
     */
    static double sinc(double x);


    static Eigen::Vector3d alignAxis(const Eigen::Vector3d &axis);


    /**
     * Code used to find a matrix that rotates around X and Y to align the Z axis
     * @param[in] matrix
     * @param[in] axis
     * @return rotation matrix
     */
    static Eigen::Affine3d alignZ(const Eigen::Affine3d &matrix, const Eigen::Vector3d &axis);
    
};


#endif //LOAM_VELODYNE_POSEMISC_HPP
