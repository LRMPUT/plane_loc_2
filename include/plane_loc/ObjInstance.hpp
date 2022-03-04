//
// Created by janw on 19.10.2021.
//

#ifndef PLANE_LOC_OBJINSTANCE_HPP
#define PLANE_LOC_OBJINSTANCE_HPP

class ObjInstance;

#include <vector>
#include <string>
#include <memory>

#include <boost/serialization/vector.hpp>

// #include <opencv2/opencv.hpp>

#include <Eigen/Eigen>

// #include <pcl/segmentation/supervoxel_clustering.h>
// #include <pcl/point_types.hpp>
// #include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "Types.hpp"
#include "Serialization.hpp"
#include "ObjInstanceView.hpp"

// only planes in a current version
class ObjInstance {
public:
    enum class ObjType {
        Plane,
        Unknown
    };

    typedef std::shared_ptr<ObjInstance> Ptr;
    typedef std::shared_ptr<const ObjInstance> ConstPtr;

    ObjInstance();

    ObjInstance(int iid,
                ObjType itype,
                int ieol = 4);

    explicit ObjInstance(const ObjInstance &other);

    void merge(const ObjInstance &other);

    void transform(const Vector7d &transform);

    void addView(const ObjInstanceView::Ptr &nview);

    ObjInstanceView::ConstPtr getBestView(const Eigen::Matrix4d &pose) const;

    ObjInstanceView::ConstPtr getBestQualityView() const;

    const std::vector<ObjInstanceView::Ptr> &getViews() const {
        return views;
    }

    inline int getId() const {
        return id;
    }

    inline void setId(int nid){
        id = nid;
    }

    void shiftIds(int startId);

    inline ObjType getType() const {
        return type;
    }

    int getEolCnt() const {
        return eolCnt;
    }

    void setEolCnt(int neolCnt) {
        eolCnt = neolCnt;
    }

    void increaseEolCnt(int eolAdd){
        ObjInstance::eolCnt += eolAdd;
    }

    void decreaseEolCnt(int eolSub){
        ObjInstance::eolCnt -= eolSub;
    }

    void removeViews(uint64_t tsThresh);

    ObjInstance::Ptr copy();

private:
    int id;

    ObjType type;

    std::vector<ObjInstanceView::Ptr> views;

    int eolCnt;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & id;
        ar & type;
        ar & views;
        ar & eolCnt;
    }
};


#endif //PLANE_LOC_OBJINSTANCE_HPP
