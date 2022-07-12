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

#ifndef INCLUDE_OBJINSTANCE_HPP_
#define INCLUDE_OBJINSTANCE_HPP_

class ObjInstanceView;

#include <vector>
#include <string>
#include <memory>

#include <boost/serialization/vector.hpp>

// #include <opencv2/opencv.hpp>

#include <Eigen/Eigen>

// #include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/impl/point_types.hpp>
// #include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "Types.hpp"
#include "ConcaveHull.hpp"
#include "Serialization.hpp"
#include "PlaneEstimator.hpp"

// only planes in a current version
class ObjInstanceView{
public:
	enum class ObjType{
		Plane,
		Unknown
	};

    typedef std::shared_ptr<ObjInstanceView> Ptr;
    typedef std::shared_ptr<const ObjInstanceView> ConstPtr;

    ObjInstanceView();
    
    /**
     *
     * @param iid
     * @param itype
     * @param ipoints
     */
	ObjInstanceView(int iid,
				ObjType itype,
                uint64_t its,
                const Eigen::MatrixPt &ipoints,
                const Eigen::MatrixCol &ipointsCol,
                const Eigen::MatrixXd &ipointsCovar,
                const Eigen::MatrixXd &idescriptor,
                const Eigen::Vector4d &iplaneEq,
                const Eigen::Matrix4d &ipose);

    explicit ObjInstanceView(const ObjInstanceView &other);
    
	// void merge(const ObjInstanceView &other);

	inline int getId() const {
		return id;
	}
    
    inline void setId(int nid){
        id = nid;
    }

	inline ObjType getType() const {
		return type;
	}

    inline uint64_t getTs() const {
        return ts;
    }

	inline const Eigen::MatrixPt &getPoints() const {
		return points;
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud() const;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloudProj() const;

    inline Eigen::Vector3d getNormal() const {
        return planeEq.head<3>();
    }

    inline const Eigen::Vector4d &getPlaneEq() const {
        return planeEq;
    }

    inline const Eigen::Matrix4d &getPose() const {
        return pose;
    }
	
    inline const ConcaveHull &getHull() const {
        return *hull;
	}

    inline const PlaneEstimator &getPlaneEstimator() const {
        return planeEstimator;
    }

    inline const Eigen::MatrixXd &getEqPoints() const {
        return eqPoints;
    }

    inline const Eigen::Vector3d &getCentroid() const {
        return centroid;
    }

    inline const int& getImageArea() const {
        return imageArea;
    }

    void transform(const Vector7d &transform);

    const Eigen::VectorXd &getDescriptor() const {
        return descriptor;
    }
    
    bool isMatching(const ObjInstanceView &other,
                    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                    int viewPort1 = -1,
                    int viewPort2 = -1) const;
    
    static double descriptorDist(const Eigen::VectorXd &desc1, const Eigen::VectorXd &desc2);

    double viewDist(const Eigen::Matrix4d &viewPose) const;

    double getQuality() const;
    
    void display(pcl::visualization::PCLVisualizer::Ptr viewer,
                 int vp,
                 double shading = 1.0,
                 double r = 0.0, double g = 0.0, double b = 0.0) const ;
    
    void cleanDisplay(pcl::visualization::PCLVisualizer::Ptr viewer,
                      int vp) const;

    ObjInstanceView::Ptr copy() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    // void compEqPoints();

    void compColorHist();

    void projectOntoPlane();

    void filter();
    
	int id;

	ObjType type;

    uint64_t ts;

	Eigen::MatrixPt points;

	Eigen::MatrixCol pointsCol;

    // always in original camera frame of reference
    Eigen::MatrixXd pointsCovar;

    Eigen::MatrixPt pointsProj;

    Eigen::VectorXd descriptor;

    Eigen::Vector4d planeEq;

    Eigen::Matrix4d pose;

    std::shared_ptr<ConcaveHull> hull;
	
    PlaneEstimator planeEstimator;

    Eigen::MatrixXd eqPoints;

    Eigen::Vector3d centroid;

    int imageArea;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & id;
        ar & type;
        ar & ts;
        ar & points;
        ar & pointsCol;
        ar & pointsCovar;
        ar & pointsProj;
        ar & descriptor;
        ar & planeEq;
        ar & pose;
        ar & hull;
        ar & planeEstimator;
        ar & eqPoints;
        ar & centroid;
        ar & imageArea;
    }
};



#endif /* INCLUDE_OBJINSTANCE_HPP_ */
