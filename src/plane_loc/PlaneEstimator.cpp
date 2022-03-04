//
// Created by jachu on 27.02.18.
//

#include <iostream>

#include <Eigen/Dense>

#include "PlaneEstimator.hpp"
#include "Misc.hpp"

using namespace std;

PlaneEstimator::PlaneEstimator() {

}

PlaneEstimator::PlaneEstimator(const Eigen::MatrixXd &pts) {
    init(pts);
}

PlaneEstimator::PlaneEstimator(const Eigen::MatrixXd &pts,
                               const Eigen::MatrixXd &ptsCovar)
{
    init(pts, ptsCovar);
}

PlaneEstimator::PlaneEstimator(const Eigen::Vector3d &icentroid,
                               const Eigen::Matrix3d &icovar,
                               double inpts)
{
    init(icentroid,
         icovar,
         inpts);
}

void PlaneEstimator::init(const Eigen::MatrixXd &pts) {
    Eigen::Vector3d icentroid;
    Eigen::Matrix3d icovar;
    compCentroidAndCovar(pts, icentroid, icovar);
    init(icentroid, icovar, pts.cols());
}

void PlaneEstimator::init(const Eigen::MatrixXd &pts,
                          const Eigen::MatrixXd &ptsCovar)
{
    // Eigen::Vector3d icentroid;
    // Eigen::Matrix3d icovar;
    // compCentroidAndCovar(pts, icentroid, icovar);
    // init(icentroid, icovar, pts.cols());

    centroid = Eigen::Vector3d::Zero();
    covar = Eigen::Matrix3d::Zero();
    for(int p = 0; p < pts.cols(); ++p) {
        centroid += pts.col(p).head<3>();
    }
    centroid /= pts.cols();
    for(int p = 0; p < pts.cols(); ++p) {
        // assuming camera frame of reference
        Eigen::Matrix3d curCovar = Eigen::Matrix3d::Zero();
        curCovar(0, 0) = 0.05 * 0.05;
        curCovar(1, 1) = 0.05 * 0.05;
        curCovar(2, 2) = ptsCovar(p);

        covar += curCovar + pts.col(p).head<3>() * pts.col(p).head<3>().transpose() - centroid * centroid.transpose();
    }
    npts = pts.cols();

    compPlaneParams(centroid,
                    covar,
                    evecs,
                    evals,
                    planeEq);
}

void PlaneEstimator::init(const Eigen::Vector3d &icentroid, const Eigen::Matrix3d &icovar, double inpts) {
    centroid = icentroid;
    covar = icovar;
    npts = inpts;
    compPlaneParams(centroid,
                    covar,
                    evecs,
                    evals,
                    planeEq);
}

void PlaneEstimator::update(const Eigen::Vector3d &ucentroid, const Eigen::Matrix3d &ucovar, double unpts) {
    // cout << "centroid = " << centroid.transpose() << endl;
    // cout << "covar = " << covar << endl;
    // cout << "npts = " << npts << endl;
    // cout << "ucentroid = " << ucentroid.transpose() << endl;
    // cout << "ucovar = " << ucovar << endl;
    // cout << "unpts = " << unpts << endl;

    updateCentroidAndCovar(centroid,
                           covar,
                           npts,
                           ucentroid,
                           ucovar,
                           unpts,
                           centroid,
                           covar,
                           npts);

    // cout << "centroid = " << centroid.transpose() << endl;
    // cout << "covar = " << covar << endl;
    // cout << "npts = " << npts << endl;
    
    static constexpr double ptsLimit = 500000;
    if(npts > ptsLimit){
        double scale = (double)npts/ptsLimit;
        covar /= scale;
        npts = ptsLimit;
    }
//    cout << "centroid = " << centroid.transpose() << endl;
//    cout << "covar = " << covar << endl;
//    cout << "npts = " << npts << endl;
    
    compPlaneParams(centroid,
                    covar,
                    evecs,
                    evals,
                    planeEq);
}

double PlaneEstimator::distance(const PlaneEstimator &other) const {
    const Eigen::Vector3d &centroid1 = centroid;
    const Eigen::Vector3d &centroid2 = other.centroid;
    Eigen::Matrix3d covar1 = covar / npts;
    Eigen::Matrix3d covar2 = other.covar / other.npts;
    double npts1 = npts;
    double npts2 = other.npts;
    
//    cout << "centroid1 = " << centroid1.transpose() << endl;
//    cout << "covar1 = " << covar1 << endl;
//    cout << "centroid2 = " << centroid2.transpose() << endl;
//    cout << "covar2 = " << covar2 << endl;

    // Eigen::Vector3d centrDiff = centroid2 - centroid1;
    // // covariance of the second plane relative to the centroid of the first plane
    // Eigen::Matrix3d relCovar = covar2 + (centrDiff * centrDiff.transpose());
    //
    // const Eigen::Vector3d &normal = evecs.col(2);
    // double varNorm = normal.transpose() * relCovar * normal;

    Eigen::Matrix3d constrVectors1 = getConstrVectors();
    Eigen::MatrixXd eqPoints2 = other.getEqPoints();
    double error = 0.0;
    int n = 0;
    for (int cv = 0; cv < constrVectors1.cols(); ++cv) {
        for (int pt = 0; pt < eqPoints2.cols(); ++pt) {
            double cur_error = constrVectors1.col(cv).transpose() * (eqPoints2.col(pt) - centroid1);
            error += cur_error * cur_error;
            ++n;
        }
    }
    error = sqrt(error / n);

    return error;
}

double PlaneEstimator::distance(const Eigen::MatrixXd &eqPoints2) const {
    const Eigen::Vector3d &centroid1 = centroid;
    // const Eigen::Vector3d &centroid2 = other.centroid;
    Eigen::Matrix3d covar1 = covar / npts;
    // Eigen::Matrix3d covar2 = other.covar / other.npts;
    double npts1 = npts;
    // double npts2 = other.npts;

//    cout << "centroid1 = " << centroid1.transpose() << endl;
//    cout << "covar1 = " << covar1 << endl;
//    cout << "centroid2 = " << centroid2.transpose() << endl;
//    cout << "covar2 = " << covar2 << endl;

    // Eigen::Vector3d centrDiff = centroid2 - centroid1;
    // // covariance of the second plane relative to the centroid of the first plane
    // Eigen::Matrix3d relCovar = covar2 + (centrDiff * centrDiff.transpose());
    //
    // const Eigen::Vector3d &normal = evecs.col(2);
    // double varNorm = normal.transpose() * relCovar * normal;

    Eigen::Matrix3d constrVectors1 = getConstrVectors();
    // Eigen::MatrixXd eqPoints2 = other.getEqPoints();
    double error = 0.0;
    int n = 0;
    for (int cv = 0; cv < constrVectors1.cols(); ++cv) {
        for (int pt = 0; pt < eqPoints2.cols(); ++pt) {
            double cur_error = constrVectors1.col(cv).transpose() * (eqPoints2.col(pt) - centroid1);
            error += cur_error * cur_error;
            ++n;
        }
    }
    error = sqrt(error / n);

    return error;
}

void PlaneEstimator::compCentroidAndCovar(const Eigen::MatrixXd &pts,
                                         Eigen::Vector3d &centroid,
                                         Eigen::Matrix3d &covar)
{
    Eigen::Vector4d mean = Eigen::Vector4d::Zero();
    for(int i = 0; i < pts.cols(); ++i){
        mean += pts.col(i);
    }
    mean /= pts.cols();
    
    centroid = mean.head<3>();
    
    Eigen::MatrixXd demeanPts = pts;
    for(int i = 0; i < demeanPts.cols(); ++i){
        demeanPts.col(i) -= mean;
    }
    
    covar = demeanPts.topRows<3>() * demeanPts.topRows<3>().transpose();
}

void PlaneEstimator::compPlaneParams(const Eigen::Vector3d &centroid,
                                     const Eigen::Matrix3d &covar,
                                     Eigen::Matrix3d &evecs,
                                     Eigen::Vector3d &evals,
                                     Eigen::Vector4d &planeEq)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> evd(covar);
    
    for(int i = 0; i < 3; ++i){
        evecs.col(i) = evd.eigenvectors().col(2 - i);
        evals(i) = evd.eigenvalues()(2 - i);
    }
    
    planeEq.head<3>() = evecs.col(2).cast<double>();
    // distance is the dot product of normal and point lying on the plane
    planeEq(3) = -planeEq.head<3>().dot(centroid);
}

Eigen::MatrixXd PlaneEstimator::compEqPointsPlaneEq(const Eigen::Vector4d &curPlaneEq) {
    Eigen::Vector3d normal = curPlaneEq.head<3>();
    normal.normalize();
    // cout << "normal = \n" << normal << endl;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(normal.transpose());
    Eigen::MatrixXd nullSpace = lu.kernel().colwise().normalized();
    // cout << "nullSpace = \n" << nullSpace << endl;

    Eigen::Matrix2d covar2d = nullSpace.transpose() * covar * nullSpace;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> evd(covar2d);
    // cout << "covar = \n" << covar << endl;
    // cout << "covar2d = \n" << covar2d << endl;

    const Eigen::Vector2d &curEvals = evd.eigenvalues();
    Eigen::MatrixXd curEvecs = nullSpace * evd.eigenvectors();
    // cout << "curEvals = \n" << curEvals << endl;
    // cout << "curEvecs = \n" << curEvecs << endl;
    // cout << "curEvecs.transposed() * normal = \n" << curEvecs.transpose() * normal << endl;

    Eigen::Vector3d centroidPlane = Misc::projectPointOnPlane(centroid, curPlaneEq);

    Eigen::MatrixXd points(3, 4);
    points.block<3, 1>(0, 0) = centroidPlane + curEvecs.col(0) * sqrt(curEvals(0) / npts);
    points.block<3, 1>(0, 1) = centroidPlane - curEvecs.col(0) * sqrt(curEvals(0) / npts);
    points.block<3, 1>(0, 2) = centroidPlane + curEvecs.col(1) * sqrt(curEvals(1) / npts);
    points.block<3, 1>(0, 3) = centroidPlane - curEvecs.col(1) * sqrt(curEvals(1) / npts);

    return points;
}

Eigen::Vector3d PlaneEstimator::compCentroidPlaneEq(const Eigen::Vector4d &curPlaneEq) {
    Eigen::Vector3d centroidPlane = Misc::projectPointOnPlane(centroid, curPlaneEq);

    return centroidPlane;
}

void PlaneEstimator::transform(const Vector7d &transform) {
    Eigen::Matrix4d transformMat = Misc::toMatrix(transform);
    Eigen::Matrix3d R = transformMat.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformMat.block<3, 1>(0, 3);
    Eigen::Matrix4d Tinvt = transformMat.inverse();
    Tinvt.transposeInPlace();
    
    // Eigen::Vector3d centroid;
    centroid = R * centroid + t;
    
    // Eigen::Matrix3d covar;
    covar = R * covar * R.transpose();
    
    // Eigen::Matrix3d evecs;
    evecs = R * evecs;
    
    // Eigen::Vector3d evals;
    // no need to transform
    
    // Eigen::Vector4d planeEq;
    planeEq = Tinvt * planeEq;
    
    // double npts;
    // no need to transform

    // measCovar = R * measCovar * R.transpose();
}

Eigen::MatrixXd PlaneEstimator::getEqPoints() const {
    // Four points in directions of two largest eigenvectors, distant one standard deviation from the centroid
    Eigen::MatrixXd points(3, 4);
    points.block<3, 1>(0, 0) = centroid + evecs.col(0) * sqrt(evals(0) / npts);
    points.block<3, 1>(0, 1) = centroid - evecs.col(0) * sqrt(evals(0) / npts);
    points.block<3, 1>(0, 2) = centroid + evecs.col(1) * sqrt(evals(1) / npts);
    points.block<3, 1>(0, 3) = centroid - evecs.col(1) * sqrt(evals(1) / npts);

    return points;
}

Eigen::Matrix3d PlaneEstimator::getConstrVectors() const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> evd(covar / npts);

    Eigen::Matrix3d evecsConstr;
    Eigen::Vector3d evalsConstr;
    for(int i = 0; i < 3; ++i){
        evecsConstr.col(i) = evd.eigenvectors().col(2 - i);
        evalsConstr(i) = evd.eigenvalues()(2 - i);
    }

    Eigen::Matrix3d constrVectors = (evecsConstr.array().rowwise() / evalsConstr.array().transpose().sqrt()).matrix();

    return constrVectors;
}

double PlaneEstimator::getCurv() const {
    return evals(2) / (evals(0) + evals(1) + evals(2));
}

void PlaneEstimator::updateCentroidAndCovar(const Eigen::Vector3d &centroid1,
                                            const Eigen::Matrix3d &covar1,
                                            const double &npts1,
                                            const Eigen::Vector3d &centroid2,
                                            const Eigen::Matrix3d &covar2,
                                            const double &npts2,
                                            Eigen::Vector3d &ocentroid,
                                            Eigen::Matrix3d &ocovar,
                                            double &onpts)
{
    ocentroid = (npts1 * centroid1 + npts2 * centroid2)/(npts1 + npts2);
    Eigen::Vector3d centrDiff = centroid1 - centroid2;
    double fact = ((double)npts1 * npts2)/(npts1 + npts2);
//    cout << "(npts1 * npts2)/(npts1 + npts2) = " << fact << endl;
//    cout << "(centrDiff * centrDiff.transpose()) = " << (centrDiff * centrDiff.transpose()) << endl;
    ocovar = covar1 + covar2 + fact*(centrDiff * centrDiff.transpose());
    onpts = npts1 + npts2;
}
