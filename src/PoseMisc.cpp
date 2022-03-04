//
// Created by jachu on 17.05.18.
//

#include <iostream>
#include <PoseMisc.h>


Eigen::Matrix4d PoseMisc::toMatrix(float *transRpy) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat.block<3, 3>(0, 0) = (Eigen::AngleAxisd(transRpy[1], Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(transRpy[0], Eigen::Vector3d::UnitX())
                             * Eigen::AngleAxisd(transRpy[2], Eigen::Vector3d::UnitZ())).toRotationMatrix();

//    mat.block<3, 3>(0, 0) = rotMat;
    mat(0, 3) = transRpy[3];
    mat(1, 3) = transRpy[4];
    mat(2, 3) = transRpy[5];
    
    
    return mat;
}

Eigen::Matrix3d PoseMisc::toMatrixRot(float *rpy) {
    return Eigen::Matrix3d((Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX())
                            * Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ())).toRotationMatrix());
}

void PoseMisc::toTransRpy(const Eigen::Matrix4d &mat, float *transRpy) {
//    Eigen::Vector3d rpy = mat.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
//    Eigen::Vector3d rpy = getRPY(mat.block<3, 3>(0, 0));
    // to orthogonalize
    Eigen::Quaterniond q(mat.block<3, 3>(0, 0));
    q.normalize();
    Eigen::Vector3d pry = q.toRotationMatrix().eulerAngles(1, 0, 2);
//    Eigen::Vector3d rpy = getRPY(q.toRotationMatrix());
    
    transRpy[0] = pry(1);
    transRpy[1] = pry(0);
    transRpy[2] = pry(2);
    transRpy[3] = mat(0, 3);
    transRpy[4] = mat(1, 3);
    transRpy[5] = mat(2, 3);
}

Eigen::Vector3d PoseMisc::toRpy(const Eigen::Matrix3d &R) {
    Eigen::Quaterniond q(R);
    q.normalize();
    Eigen::Vector3d pry = q.toRotationMatrix().eulerAngles(1, 0, 2);
    Eigen::Vector3d rpy;
    rpy << pry(1), pry(0), pry(2);
    
    return rpy;
}


Eigen::Matrix4d PoseMisc::inverseTrans(const Eigen::Matrix4d &mat) {
    Eigen::Matrix4d inv = Eigen::Matrix4d::Identity();
    inv.block<3, 3>(0, 0) = mat.block<3, 3>(0, 0).transpose();
    inv.block<3, 1>(0, 3) = -inv.block<3, 3>(0, 0) * mat.block<3, 1>(0, 3);
//    inv(3, 3) = 1.0;
    
    return inv;
}

Vector7d PoseMisc::toVector(const Eigen::Matrix4d &mat) {
    Vector7d v;
    v.head<3>() = mat.block<3, 1>(0, 3);
    Eigen::Quaterniond q(mat.block<3, 3>(0, 0));
    v.tail<4>() = q.normalized().coeffs();
    return v;
}

Eigen::Matrix4d PoseMisc::toMatrix(const Vector7d &v) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(v(6), v(3), v(4), v(5));
    mat.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    mat.block<3, 1>(0, 3) = v.head<3>();
    return mat;
}

Eigen::Matrix3d PoseMisc::skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d ret;
    ret <<	0,		-v(2),	v(1),
            v(2),	0,		-v(0),
            -v(1),	v(0),	0;
    return ret;
}

Eigen::Vector3d PoseMisc::vee(const Eigen::Matrix3d &R) {
    Eigen::Vector3d ret;
    ret << R(2, 1), R(0, 2), R(1, 0);

    return ret;
}

Vector6d PoseMisc::log(const Eigen::Matrix4d &m) {
    static constexpr double eps = 1.0e-8;

    Eigen::Matrix3d R = m.block<3, 3>(0, 0);
    Eigen::Vector3d t = m.block<3, 1>(0, 3);

    Eigen::Vector3d omega = log(R);

    double theta = omega.norm();
    Eigen::Matrix3d invLeftJ;
    if (theta < eps) {
        invLeftJ = Eigen::Matrix3d::Identity() - 0.5 * skew(omega);
    }
    else {
        Eigen::Vector3d axis = omega / theta;
        double halfTheta = theta / 2.0;
        double cotHalfTheta = 1.0 / tan(halfTheta);

        invLeftJ = halfTheta * cotHalfTheta * Eigen::Matrix3d::Identity() +
                (1.0 - halfTheta * cotHalfTheta) * (axis * axis.transpose()) -
                halfTheta * skew(axis);
    }
    Eigen::Vector3d upsilon = invLeftJ * t;

    Vector6d res;
    res << omega, upsilon;
    
    return res;
}

Eigen::Matrix4d PoseMisc::exp(const Vector6d &u) {
    static constexpr double eps = 1.0e-8;

    Eigen::Vector3d omega = u.head<3>();
    Eigen::Vector3d upsilon = u.tail<3>();

    double theta = omega.norm();
    Eigen::Matrix3d leftJ;
    if (theta < eps)
    {
        leftJ = Eigen::Matrix3d::Identity() + 0.5 * skew(omega);
    }
    else
    {
        Eigen::Vector3d axis = omega / theta;
        double s = sin(theta);
        double c = cos(theta);

        leftJ = (s / theta) * Eigen::Matrix3d::Identity() +
                (1.0 - s / theta) * (axis * axis.transpose()) +
                ((1.0 - c) / theta) * skew(axis);
    }

    Eigen::Matrix3d R = exp(omega);
    Eigen::Vector3d t = leftJ * upsilon;
    
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.block<3, 3>(0, 0) = R;
    ret.block<3, 1>(0, 3) = t;
    
    return ret;
}

Eigen::Vector3d PoseMisc::deltaR(const Eigen::Matrix3d &R) {
    Eigen::Vector3d v;
    v(0)=R(2,1)-R(1,2);
    v(1)=R(0,2)-R(2,0);
    v(2)=R(1,0)-R(0,1);
    return v;
}

#define PRINT(x) std::cout << #x << " = " << std::endl << x << std::endl

Eigen::Vector3d PoseMisc::log(const Eigen::Matrix3d &R) {
    static constexpr double eps = 1.0e-8;

    // 0.5 * (trace(R) - 1)
    double cosTheta =  0.5*(R.trace() - 1.0);
    // make sure it is in <-1.0, 1.0>
    cosTheta = std::max(std::min(cosTheta, 1.0), -1.0);
    double theta = acos(cosTheta);

    Eigen::Vector3d omega;
    if (std::abs(theta) < eps) {
        omega = vee(R - Eigen::Matrix3d::Identity());
    }
    else if (std::abs(theta - M_PI) < eps) {
        Eigen::Matrix3d B = 0.5 * (R + Eigen::Matrix3d::Identity());
        Eigen::Vector3d axis = B.diagonal().cwiseSqrt();
        Eigen::Vector3d signs = vee(R - R.transpose()).cwiseSign();
        bool signsCorrect = true;
        for (int i = 0; i < 3; ++i) {
            if (signs(i) == 0.0 && axis(i) != 0.0) {
                signsCorrect = false;
            }
        }
        if (!signsCorrect) {
            signs = Eigen::Vector3d::Ones();
            int maxIdx = 0;
            for (int i = 0; i < 3; ++i) {
                if (axis(maxIdx) < axis(i)) {
                    maxIdx = i;
                }
            }
            for (int i = 0; i < 3; ++i) {
                if (i != maxIdx) {
                    if (B(maxIdx, i) < 0.0) {
                        signs(i) = -1.0;
                    }
                }
            }
        }
        omega = theta * axis.cwiseProduct(signs);
    }
    else {
        // if (std::abs(theta - M_PI) < 1e-3) {
        //     PRINT(theta);
        //     PRINT((0.5 * theta / sin(theta)));
        //     PRINT((R - R.transpose()));
        //     PRINT((0.5 * theta / sin(theta)) * (R - R.transpose()));
        // }
        omega = vee((0.5 * theta / sin(theta)) * (R - R.transpose()));
    }

    return omega;
}

Eigen::Matrix3d PoseMisc::exp(const Eigen::Vector3d &omega) {
    static constexpr double eps = 1.0e-8;

    double theta = omega.norm();
    
    Eigen::Matrix3d R;
    if (theta < eps)
    {
        R = Eigen::Matrix3d::Identity() + skew(omega);
    }
    else
    {
        Eigen::Vector3d axis = omega / theta;
        double s = sin(theta);
        double c = cos(theta);

        R = c * Eigen::Matrix3d::Identity() +
            (1.0 - c) * (axis * axis.transpose()) +
            s * skew(axis);
    }

    return R;
}

double PoseMisc::sinc(double x) {
    if(fabs(x) < 1e-5){
        // using Taylor expansion
        double x2 = x*x;
        return 1.0 - x2/6.0 + x2*x2/120.0;
    }
    else {
        return sin(x)/x;
    }
}

Eigen::Vector3d PoseMisc::alignAxis(const Eigen::Vector3d &axis) {

    // Getting log-map parameters
    double theta = acos(axis(2));
    double sincVal = sinc(theta);
    Eigen::Vector3d om = Eigen::Vector3d::Zero();
    if(sincVal > 1e-5){
        om(0) = -axis(1) / sincVal;
        om(1) = axis(0) / sincVal;
    }
    
    return om;
}

Eigen::Affine3d PoseMisc::alignZ(const Eigen::Affine3d &matrix, const Eigen::Vector3d &axis) {
    Eigen::Vector3d axisInLocal = matrix.rotation().inverse() * axis;

    Eigen::Vector3d rotLogMap = alignAxis(axisInLocal);

    Vector6d vec = Vector6d::Zero();
    vec.head<3>() = rotLogMap;

    Eigen::Matrix4d correctionMatrix = PoseMisc::exp(vec);

    // rotation matrix from log-map that aligns Z axis to match global Z
    Eigen::Affine3d correction = Eigen::Affine3d::Identity();
    correction.matrix().block<3, 3>(0, 0) = correctionMatrix.block<3, 3>(0, 0);
    return correction;
}



