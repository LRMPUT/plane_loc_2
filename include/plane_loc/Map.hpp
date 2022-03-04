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

#ifndef INCLUDE_MAP_HPP_
#define INCLUDE_MAP_HPP_

class Map;

#include <vector>
#include <list>
#include <set>
#include <memory>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>

#include <nabo/nabo.h>

#include "ObjInstance.hpp"
#include "Serialization.hpp"
#include "UnionFindHash.hpp"


class Map{
public:
    struct Settings{
        int eolObjInstInit;
        
        int eolObjInstIncr;
        
        int eolObjInstDecr;
        
        int eolObjInstThresh;
        
        // int eolPendingInit;
        //
        // int eolPendingIncr;
        //
        // int eolPendingDecr;
        //
        // int eolPendingThresh;
    
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & eolObjInstInit;
            ar & eolObjInstIncr;
            ar & eolObjInstDecr;
            ar & eolObjInstThresh;
            // ar & eolPendingInit;
            // ar & eolPendingIncr;
            // ar & eolPendingDecr;
            // ar & eolPendingThresh;
        }
    };
    
	Map();
	
	explicit Map(const cv::FileStorage& fs);

    explicit Map(const Map &other);

    void transform(const Vector7d &transform);

	void createNewObj(const ObjInstanceView::Ptr &view);
    
    // void addObjs(std::vector<ObjInstanceView::Ptr>::iterator beg,
    //              std::vector<ObjInstanceView::Ptr>::iterator end);
    
	inline int size() const {
		return planeIdToObj.size();
	}

//	inline ObjInstance& operator[](int i){
//		return objInstances[i];
//	}
    
    inline std::unordered_map<int, ObjInstance::Ptr>::const_iterator begin() const {
        return planeIdToObj.begin();
    }
    
    inline std::unordered_map<int, ObjInstance::Ptr>::const_iterator end() const {
        return planeIdToObj.end();
    }
    
    void mergeNewObjInstanceViews(const std::vector<ObjInstanceView::Ptr> &newObjInstanceViews,
                                  const Vector7d &pose,
                                  const Eigen::Matrix3d &cameraMatrix,
                                  int rows,
                                  int cols,
                                  pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                                  int viewPort1 = -1,
                                  int viewPort2 = -1);
    
    void mergeMapObjInstances(pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                              int viewPort1 = -1,
                              int viewPort2 = -1);
    
    void decreaseObjEol(int eolSub);
    
    void removeObjsEol();
    
    void removeObjsEolThresh(int eolThresh);
    
    void shiftIds(int startId);

    std::vector<std::vector<ObjInstanceView::ConstPtr>> getKNN(const std::vector<ObjInstanceView::ConstPtr> &viewsQuery,
                                                          int k,
                                                          double maxDist2 = std::numeric_limits<double>::infinity());

    std::unordered_map<int, std::pair<int, ObjInstanceView::ConstPtr>>
    getVisibleObjs(const Vector7d &pose,
                   const Eigen::Matrix3d &cameraMatrix,
                   int rows,
                   int cols,
                   pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
                   int viewPort1 = -1,
                   int viewPort2 = -1) const;

    void removeViews(uint64_t tsThresh);
    
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr getLabeledColorPointCloud();

    pcl::PointCloud<pcl::PointXYZL>::Ptr getLabeledPointCloud();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorPointCloud();

    void display(pcl::visualization::PCLVisualizer::Ptr viewer,
                 int v1,
                 int v2,
                 float r = -1.0,
                 float g = -1.0,
                 float b = -1.0,
                 bool dispInfo = true);

    void exportPointCloud(const std::string &path);

private:
    void merge(const ObjInstance::Ptr &mapObj, const ObjInstance::Ptr &newObj);

	void buildKdtree();


    std::unordered_map<int, ObjInstance::Ptr> planeIdToObj;

    Eigen::MatrixXd descs;

    std::vector<std::pair<int, int>> planeIdViews;

    std::shared_ptr<Nabo::NNSearchD> nns;

    bool nnsUptodate;

    Settings settings;
    
    friend class boost::serialization::access;
    
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const {
        // ar << objInstances;
        ar << planeIdToObj;
        ar << descs;
        ar << planeIdViews;
        ar << nnsUptodate;
        ar << settings;
    }
    
    template<class Archive>
    void load(Archive & ar, const unsigned int version) {
        ar >> planeIdToObj;
        ar >> descs;
        ar >> planeIdViews;
        ar >> nnsUptodate;
        ar >> settings;

        if (descs.cols() > 0) {
            nns.reset(Nabo::NNSearchD::createKDTreeLinearHeap(descs));
        }
    }
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        boost::serialization::split_member(ar, *this, version);
    }
};



#endif /* INCLUDE_MAP_HPP_ */
