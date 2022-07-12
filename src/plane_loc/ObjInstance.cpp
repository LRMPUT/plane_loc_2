//
// Created by janw on 19.10.2021.
//

#include "ObjInstance.hpp"

ObjInstance::ObjInstance() : id(-1) {}

ObjInstance::ObjInstance(int iid,
                         ObjInstance::ObjType itype,
                         int ieol)
 : id(iid),
   type(itype),
   eolCnt(ieol)
{

}

ObjInstance::ObjInstance(const ObjInstance &other) {
    id = other.id;
    type = other.type;
    for (const auto &view : other.views) {
        views.push_back(view->copy());
    }
    eolCnt = other.eolCnt;
}

void ObjInstance::merge(const ObjInstance &other) {
    views.insert(views.end(), other.views.begin(), other.views.end());
}

void ObjInstance::transform(const Vector7d &transform) {
    for (auto &curView : views) {
        curView->transform(transform);
    }
}

void ObjInstance::addView(const ObjInstanceView::Ptr &nview) {
    views.push_back(nview);
}

ObjInstanceView::ConstPtr ObjInstance::getBestView(const Eigen::Matrix4d &pose) const {
    double largestArea = 0.0;
    for (int v = 0; v < views.size(); ++v) {
        double curA = views[v]->getImageArea();
        if (largestArea < curA) {
            largestArea = curA;
        }
    }

    double bestDist = std::numeric_limits<double>::max();
    int bestIdx = -1;

    for (int v = 0; v < views.size(); ++v) {
        // best view among at most two times smaller than largest
        if (largestArea < 2 * views[v]->getImageArea()) {
            double curDist = views[v]->viewDist(pose);
            if (curDist < bestDist) {
                bestDist = curDist;
                bestIdx = v;
            }
        }
    }

    if (bestIdx >= 0) {
        return views[bestIdx];
    }
    else {
        return ObjInstanceView::ConstPtr();
    }
}

ObjInstanceView::ConstPtr ObjInstance::getBestQualityView() const {
    double largestArea = 0.0;
    for (int v = 0; v < views.size(); ++v) {
        double curA = views[v]->getImageArea();
        if (largestArea < curA) {
            largestArea = curA;
        }
    }

    double bestQ = 0.0;
    int bestIdx = -1;

    for (int v = 0; v < views.size(); ++v) {
        // best quality among at most two times smaller than largest
        if (largestArea < 2 * views[v]->getImageArea()) {
            double curQ = views[v]->getQuality();
            if (bestQ < curQ) {
                bestQ = curQ;
                bestIdx = v;
            }
        }
    }

    if (bestIdx >= 0) {
        return views[bestIdx];
    }
    else {
        return ObjInstanceView::ConstPtr();
    }
}

void ObjInstance::shiftIds(int startId) {
    id += startId;
    for (auto &view : views) {
        view->setId(view->getId() + startId);
    }
}

void ObjInstance::removeViews(uint64_t tsThresh) {
    std::vector<ObjInstanceView::Ptr> newViews;
    // cout << "tsThresh = " << tsThresh << endl;
    for (const auto &view : views) {
        // cout << "view->getTs() = " << view->getTs() << endl;
        if (tsThresh <= view->getTs()) {
            newViews.push_back(view);
        }
    }
    views.swap(newViews);
}

ObjInstance::Ptr ObjInstance::copy() {
    // use copy constructor
    return ObjInstance::Ptr(new ObjInstance(*this));
}