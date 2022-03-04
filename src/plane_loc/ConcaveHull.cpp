//
// Created by jachu on 31.01.18.
//

#define CGAL_DISABLE_ROUNDING_MATH_CHECK

#include <iostream>
#include <map>


#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
// #include <g2o/types/slam3d/se3quat.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <CGAL/Boolean_set_operations_2.h>

#include "ConcaveHull.hpp"
#include "Misc.hpp"


using namespace std;

ConcaveHull::ConcaveHull() : totalArea(0.0) {}

ConcaveHull::ConcaveHull(const Eigen::MatrixPt &points3d,
                         const Eigen::Vector4d &planeEq)
{
    init(points3d,
         planeEq);
}


ConcaveHull::ConcaveHull(const vector<ConcaveHull::Polygon_2> &ipolygons,
                         const vector<Eigen::MatrixPt> &ipolygons3d,
                         const Eigen::Vector3d &iplNormal,
                         double iplD,
                         const Eigen::Vector3d &iorigin,
                         const Eigen::Vector3d &ixAxis,
                         const Eigen::Vector3d &iyAxis)
{
//    computeFrame();
//    Eigen::Vector4d planeEq;
//    planeEq.head<3>() = plNormal;
//    planeEq(3) = -plD;
//    cout << "ConcaveHull::ConcaveHull, planeEq = " << planeEq.transpose() << endl;
    
    init(ipolygons,
         ipolygons3d,
         iplNormal,
         iplD,
         iorigin,
         ixAxis,
         iyAxis);
}


ConcaveHull::ConcaveHull(const ConcaveHull &other)
{
    polygons = other.polygons;
    areas = other.areas;
    totalArea = other.totalArea;
    polygons3d = other.polygons3d;

    plNormal = other.plNormal;
    plD = other.plD;
    origin = other.origin;
    xAxis = other.xAxis;
    yAxis = other.yAxis;
}

void ConcaveHull::init(const Eigen::MatrixPt &points3d,
                       const Eigen::Vector4d &planeEq)
{
    totalArea = 0.0;
    
    // pcl::ModelCoefficients::Ptr mdlCoeff (new pcl::ModelCoefficients);
    // mdlCoeff->values.resize(4);
    // mdlCoeff->values[0] = planeEq(0);
    // mdlCoeff->values[1] = planeEq(1);
    // mdlCoeff->values[2] = planeEq(2);
    // mdlCoeff->values[3] = planeEq(3);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr points3dProj(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    // proj.setModelType(pcl::SACMODEL_PLANE);
    // proj.setInputCloud(points3d);
    // proj.setModelCoefficients(mdlCoeff);
    // proj.filter(*points3dProj);

    pcl::PointCloud<pcl::PointXYZ>::Ptr points3dFilt(new pcl::PointCloud<pcl::PointXYZ>());
    for(int p = 0; p < points3d.cols(); ++p) {
        Eigen::Vector3d ptProj = Misc::projectPointOnPlane(points3d.col(p).head<3>(), planeEq);
        points3dFilt->push_back(pcl::PointXYZ(ptProj(0), ptProj(1), ptProj(2)));
    }

//    cout << "points3dProj->size() = " << points3dProj->size() << endl;
    pcl::VoxelGrid<pcl::PointXYZ> downsamp;
    downsamp.setInputCloud(points3dFilt);
    downsamp.setLeafSize (0.01f, 0.01f, 0.01f);
    downsamp.filter(*points3dFilt);
//    cout << "filtered points3dFilt->size() = " << points3dFilt->size() << endl;
    
    plNormal = planeEq.head<3>();
    plD = -planeEq(3);
    
    computeFrame();
    
    list<Point_2ie> points2d;
    for(int i = 0; i < points3dFilt->size(); ++i){
        points2d.push_back(point3dTo2die(points3dFilt->at(i).getVector3fMap().cast<double>()));
    }
    
    
    Alpha_shape_2 A(points2d.begin(), points2d.end(),
                    FTie(0.05),
                    Alpha_shape_2::GENERAL);
//    cout << "alpha = " << A.get_alpha() << endl;
    
    std::vector<Segment_2ie> segments;
    auto outIt = std::back_inserter(segments);
    Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
            end = A.alpha_shape_edges_end();
    for( ; it!=end; ++it) {
        if(A.classify(*it) == Alpha_shape_2::REGULAR) {
            *outIt = A.segment(*it);
            ++outIt;
        }
    }
    
    {
        multimap<Point_2ie, int> tgtToIdx;
        multimap<Point_2ie, int> srcToIdx;
        for (int s = 0; s < segments.size(); ++s) {
            
            
            tgtToIdx.insert(make_pair(segments[s].target(), s));
            srcToIdx.insert(make_pair(segments[s].source(), s));
        }
        vector<vector<int>> nextSeg(segments.size());
        vector<vector<int>> prevSeg(segments.size());
        for (int s = 0; s < segments.size(); ++s) {
            // pair of iterators
            auto next = srcToIdx.equal_range(segments[s].target());
            auto prev = tgtToIdx.equal_range(segments[s].source());
            for (auto it = next.first; it != next.second; ++it) {
                nextSeg[s].push_back(it->second);
            }
            for (auto it = prev.first; it != prev.second; ++it) {
                prevSeg[s].push_back(it->second);
            }
//                if(nextSeg[s].size() != prevSeg[s].size()){
//                    cout << "nextSeg[s].size() = " << nextSeg[s].size() << endl;
//                    cout << "prevSeg[s].size() = " << prevSeg[s].size() << endl;
//                    throw PLANE_EXCEPTION("Number of in and out connections not equal");
//                }
        }

//        double bestArea = 0.0;
        vector<int> isVisited(segments.size(), -1);
//        cout << "starting" << endl;
        for (int s = 0; s < segments.size(); ++s) {
            if (isVisited[s] < 0) {
                // cout << "starting from " << s << endl;
                
                stack<int> sSeg;
                sSeg.push(s);
                bool afterLoop = false;
                while (!sSeg.empty()) {
                    bool explore = true;
                    
                    set<int> lastNext;
                    while (explore) {
                        int curIdx = sSeg.top();
                        isVisited[curIdx] = s;

                        // cout << "curIdx = " << curIdx << endl;
                        // cout << "src = (" << segments[curIdx].source().x() << " " << segments[curIdx].source().y() << ")"
                        //      << ", tgt = (" << segments[curIdx].target().x() << " " << segments[curIdx].target().y() << ")" << endl;
//                            Segment &curSeg = segments[curIdx];
                        
                        bool advanced = false;
                        bool loop = false;
                        lastNext.clear();
                        // cout << "next segments:" << endl;
                        for (int n = 0; n < nextSeg[curIdx].size(); ++n) {
                            // cout << nextSeg[curIdx][n] << ", tgt = (" << segments[nextSeg[curIdx][n]].target().x() << " " << segments[nextSeg[curIdx][n]].target().y() << ")" << endl;
                            lastNext.insert(nextSeg[curIdx][n]);
                        }
                        // cout << endl;
                        // if loop was detected previously then there will be a visited segment in nextSeg, so skip this check
                        if (!afterLoop) {
                            for (int n = 0; n < nextSeg[curIdx].size(); ++n) {
                                if (isVisited[nextSeg[curIdx][n]] == s) {
                                    // cout << "loop" << endl;
                                    // cout << nextSeg[curIdx][n] << ", tgt = (" << segments[nextSeg[curIdx][n]].target().x() << " " << segments[nextSeg[curIdx][n]].target().y() << ")" << endl;
                                    loop = true;
                                    afterLoop = true;
                                }
                            }
                        }
                        if (!loop) {
                            for (int n = 0; n < nextSeg[curIdx].size(); ++n) {
                                if (isVisited[nextSeg[curIdx][n]] < 0) {
                                    sSeg.push(nextSeg[curIdx][n]);
                                    advanced = true;
                                    afterLoop = false;

                                    break;
                                }
                            }
                        }
                        // cout << "advanced = " << advanced << endl;
                        if (!advanced) {
                            explore = false;
                        }
                    }
                    // no path or encountered visited segment, so we have a loop
                    vector<int> curVisited;
                    bool loopStart = false;
                    // cout << "adding new poly" << endl;
                    while (!sSeg.empty() && !loopStart) {
                        int curIdx = sSeg.top();
                        // cout << "popping " << curIdx << endl;
                        // cout << "src = (" << segments[curIdx].source().x() << " " << segments[curIdx].source().y() << ")"
                        //      << ", tgt = (" << segments[curIdx].target().x() << " " << segments[curIdx].target().y() << ")" << endl;
                        
                        curVisited.push_back(curIdx);
                        sSeg.pop();

//                            // if there is another path then it is the loop start
//                            for (int n = 0; n < nextSeg[curIdx].size(); ++n) {
//                                if (!isVisited[nextSeg[curIdx][n]]) {
//                                    loopStart = true;
//                                }
//                            }
                        // if last visited segment had this segment in next then it is the loop start
                        if (lastNext.count(curIdx)) {
                            loopStart = true;
                        }
                    }
                    // cout << "all popped" << endl;

                    if (curVisited.size() > 2) {
//                        vector<Segment> curSegments;
                        Polygon_2ie poly;
                        for (auto it = curVisited.rbegin(); it != curVisited.rend(); ++it) {
                            const Segment_2ie &curSegment = segments[*it];
                            poly.push_back(curSegment.target());
                        }

                        // cout << "simplifying" << endl;
                        // cout << "before: " << poly.is_simple() << endl;
                        poly = CGAL::Polyline_simplification_2::simplify(poly,
                                                                         Cost(),
                                                                         Stop(0.05 * 0.05));
                        // cout << "after: " << poly.is_simple() << endl;
                        double area = CGAL::to_double(poly.area());
                        // cout << "area = " << area << endl;
//                        cout << curSegments.size() << "/" << segments.size() << endl;
                        if (abs(area) > 0.05) {
                            Polygon_2 polye;
                            if (poly.is_counterclockwise_oriented()) {
                                for (int pt = 0; pt < poly.size(); ++pt) {
                                    polye.push_back(Point_2(poly[pt].x(), poly[pt].y()));
                                }
                            }
                            else {
                                for (int pt = poly.size() - 1; pt >= 0; --pt) {
                                    polye.push_back(Point_2(poly[pt].x(), poly[pt].y()));
                                }
                            }
                            // cout << "poly.is_simple() = " << poly.is_simple() << endl;
                            // cout << "polye.is_simple() = " << polye.is_simple() << endl;

                            polygons.push_back(polye);
                            areas.push_back(abs(area));
                            totalArea += abs(area);
                        }
                    }
                }
            }
        }
    }
    for(int p = 0; p < polygons.size(); ++p) {
        polygons3d.push_back(Eigen::MatrixPt(4, polygons[p].size()));
        for (int s = 0; s < polygons[p].size(); ++s) {
//                Point src = polygons[p][s];
            Point_2 tgt = polygons[p][s];

//            Eigen::Vector2d curPtTgt(tgt.x(), tgt.y());
            
            {
                Eigen::Vector3d curCoordTgt = point2dTo3d(tgt);

                polygons3d.back().col(s) << curCoordTgt, 1.0;
            }
        }
    }
}

void ConcaveHull::init(const vector<ConcaveHull::Polygon_2> &ipolygons,
                       const std::vector<Eigen::MatrixPt> &ipolygons3d,
                       const Eigen::Vector3d &iplNormal,
                       double iplD,
                       const Eigen::Vector3d &iorigin,
                       const Eigen::Vector3d &ixAxis,
                       const Eigen::Vector3d &iyAxis)
{
    polygons = ipolygons;
    polygons3d = ipolygons3d;
    plNormal = iplNormal;
    plD = iplD;
    origin = iorigin;
    xAxis = ixAxis;
    yAxis = iyAxis;
    totalArea = 0.0;
    
    for(const Polygon_2 &curPoly : polygons){
        double area = CGAL::to_double(curPoly.area());
        areas.push_back(abs(area));
        totalArea += abs(area);
    }
}

ConcaveHull ConcaveHull::transform(const Vector7d &transform) const {
    Eigen::Matrix4d T = Misc::toMatrix(transform);
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::Matrix4d Tinvt = T.inverse();
    Tinvt.transposeInPlace();
    
    Eigen::Vector4d transPlaneEq;
    transPlaneEq.head<3>() = plNormal;
    transPlaneEq(3) = -plD;
    
    transPlaneEq = Tinvt * transPlaneEq;
    
    Eigen::Vector3d transOrigin = R * origin + t;
    Eigen::Vector3d transXAxis = R * xAxis;
    Eigen::Vector3d transYAxis = R * yAxis;
    
    vector<Polygon_2> transPolygons;
    vector<Eigen::MatrixPt> transPolygons3d;
//    auto it = polygons.begin();
    for(const Eigen::MatrixPt &curPoly3d: polygons3d){
        Eigen::MatrixPt transPoly3d = T * curPoly3d;

        Polygon_2 transPoly;
        for(int p = 0; p < transPoly3d.cols(); ++p){
            Eigen::Vector3d point3d = transPoly3d.col(p).head<3>();
            transPoly.push_back(Point_2((point3d - transOrigin).dot(transXAxis),
                                        (point3d - transOrigin).dot(transYAxis)));
        }
//        cout << "polygons.area() = " << it->area() << endl;
//        cout << "transPoly.area() = " << transPoly.area() << endl;
//        ++it;
        transPolygons.push_back(transPoly);
        transPolygons3d.push_back(transPoly3d);
    }


    return ConcaveHull(transPolygons,
                       transPolygons3d,
                       transPlaneEq.head<3>(),
                       -transPlaneEq(3),
                       transOrigin,
                       transXAxis,
                       transYAxis);
}

ConcaveHull ConcaveHull::intersect(const ConcaveHull &other,
                                   double areaThresh) const
{
    const std::vector<Eigen::MatrixPt> &otherPolygons3d = other.getPolygons3d();
    
    return intersect(otherPolygons3d, areaThresh);
}

ConcaveHull ConcaveHull::intersect(const std::vector<Eigen::MatrixPt> &otherPolygons3d,
                                   double areaThresh) const
{
    vector<Polygon_2> otherPolygonsProj;
    // project points onto plane of this hull
    for(const Eigen::MatrixPt &poly3d : otherPolygons3d){
        Polygon_2 polyProj;
//        cout << "other" << endl;
        for(int p = 0; p < poly3d.cols(); ++p){
            polyProj.push_back(point3dTo2d(poly3d.col(p).head<3>()));
//            cout << it->getVector3fMap().cast<double>().transpose() << endl;
//            cout << point3dTo2d(it->getVector3fMap().cast<double>()) << endl;
//            cout << point2dTo3d(point3dTo2d(it->getVector3fMap().cast<double>())).transpose() << endl;
        }
//        cout << "polyProj.is_counterclockwise_oriented() = " << polyProj.is_counterclockwise_oriented() << endl;
//        cout << "polyProj.is_simple() = " << polyProj.is_simple() << endl;
        if (!polyProj.is_counterclockwise_oriented()) {
            polyProj.reverse_orientation();
        }
        otherPolygonsProj.push_back(polyProj);
    }
//    for(const Polygon_2 &poly : polygons){
//        cout << "poly.is_counterclockwise_oriented() = " << poly.is_counterclockwise_oriented() << endl;
//        cout << "poly.is_simple() = " << poly.is_simple() << endl;
//    }

//    for(int p = 0; p < polygons.size(); ++p){
//        cout << "current" << endl;
//        for(int i = 0; i < polygons[p].size(); ++i){
//            cout << polygons3d[p]->at(i).getVector3fMap().cast<double>().transpose() << endl;
//            cout << point2dTo3d(polygons[p][i]).transpose() << endl;
//        }
//    }
    vector<Polygon_2> resPolygons;
    std::vector<Eigen::MatrixPt> resPolygons3d;
    {
        list<Polygon_holes_2> inter;
        for(const Polygon_2 &poly : polygons){
            for(const Polygon_2 &otherPoly : otherPolygonsProj){
                // cout << "poly = " << endl;
                // for (int p = 0; p < poly.size(); ++p) {
                //     cout << poly.vertex(p).x() << " " << poly.vertex(p).y() << endl;
                // }
                CGAL::intersection(poly,
                                   otherPoly,
                                   back_inserter(inter));
            }
        }
//        cout << "polygons.size() = " << polygons.size() << endl;
//        cout << "otherPolygonsProj.size() = " << otherPolygonsProj.size() << endl;
//        CGAL::intersection(polygons.begin(), polygons.end(),
//                           otherPolygonsProj.begin(), otherPolygonsProj.end(),
//                           back_inserter(inter));
//        cout << "inter.size() = " << inter.size() << endl;
        for(Polygon_holes_2 &pi : inter){
//            cout << "pi.outer_boundary().area() = " << pi.outer_boundary().area() << endl;
            if(abs(pi.outer_boundary().area()) > areaThresh){
//                    Polygon_2 resPoly;
//                    const Polygon_2e &curPolyInter = pi.outer_boundary();
//                    cout << "curPolyInter.area() = " << CGAL::to_double(curPolyInter.area()) << endl;
//                    for(int pt = 0; pt < curPolyInter.size(); ++pt){
//                        resPoly.push_back(Point_2(CGAL::to_double(curPolyInter[pt].x()),
//                                                CGAL::to_double(curPolyInter[pt].y())));
//                    }
//                    resPolygons.push_back(resPoly);
                resPolygons.push_back(pi.outer_boundary());
            }
        }
    }
    
    for(const Polygon_2 &p : resPolygons){
        Eigen::MatrixPt poly3d(4, p.size());
        for(int i = 0; i < p.size(); ++i){
            pcl::PointXYZRGB curPt3d;
            Eigen::Vector3d curCoordTgt = point2dTo3d(p[i]);
            poly3d.col(i) << curCoordTgt, 1.0;
        }
        resPolygons3d.push_back(poly3d);
    }
    
    return ConcaveHull(resPolygons,
                       resPolygons3d,
                       plNormal,
                       plD,
                       origin,
                       xAxis,
                       yAxis);
}

ConcaveHull ConcaveHull::clipToCameraFrustum(const Eigen::Matrix3d &K, int rows, int cols, double minZ)
{
    vector<Eigen::MatrixPt> clippedPolygons3d;
    for(const Eigen::MatrixPt &curPoly3d : polygons3d){
        // At most 2 times original number of points because of additional points
        Eigen::MatrixPt curClipped(4, 2 * curPoly3d.cols());
        int nextClippedIdx = 0;
        
        for(int pt = 0; pt < curPoly3d.cols(); ++pt){
            int prevPtIdx = (pt + curPoly3d.cols() - 1) % curPoly3d.cols();
            Eigen::Vector4d curPt = curPoly3d.col(pt);
            Eigen::Vector4d prevPt = curPoly3d.col(prevPtIdx);
            
            // always add current point if is valid
            if(curPt(2) >= minZ){
                curClipped.col(nextClippedIdx++) = curPt;
            }
            
            // if entering or leaving valid region, add point on a border
            if(curPt(2) >= minZ && prevPt(2) < minZ ||
               curPt(2) < minZ && prevPt(2) >= minZ)
            {
                double dprev = abs(prevPt(2) - minZ);
                double dcur = abs(curPt(2) - minZ);
                
                // both points are close to boundary
                if(dprev + dcur < 1e-6){
                    curClipped.col(nextClippedIdx++) = curPt;
                }
                else {
                    double s = dprev / (dprev + dcur);

                    Eigen::Vector4d clipPt = prevPt + s * (curPt - prevPt);

                    curClipped.col(nextClippedIdx++) = clipPt;
                }
            }
        }
        curClipped.conservativeResize(Eigen::NoChange, nextClippedIdx);
        if(curClipped.cols() > 2){
            clippedPolygons3d.push_back(curClipped);
        }
    }
    
    vector<Polygon_2> clippedPolygons;
    for(const Eigen::MatrixPt &clipPoly3d : clippedPolygons3d){
        clippedPolygons.push_back(Polygon_2());
        for(int pt = 0; pt < clipPoly3d.cols(); ++pt){
            clippedPolygons.back().push_back(point3dTo2d(clipPoly3d.col(pt).head<3>()));
        }
    }
    
    return ConcaveHull(clippedPolygons,
                      clippedPolygons3d,
                      plNormal,
                      plD,
                      origin,
                      xAxis,
                      yAxis);
}

double ConcaveHull::minDistance(const ConcaveHull &other) const {
    double minDist = std::numeric_limits<double>::max();
    
    // O(n^2) is enough for a small number of points
    for(const Eigen::MatrixPt &poly3d : polygons3d){
        for(int pt = 0; pt < poly3d.cols(); ++pt){
            for(const Eigen::MatrixPt &otherPoly3d : other.getPolygons3d()){
                for(int otherPt = 0; otherPt < otherPoly3d.cols(); ++otherPt){
                    Eigen::Vector3d pt1 = poly3d.col(pt).head<3>();
                    Eigen::Vector3d pt2 = otherPoly3d.col(otherPt).head<3>();
                    double curDist = (pt1 - pt2).norm();
                    minDist = min(curDist, curDist);
                }
            }
        }
    }
    
    return minDist;
}

void ConcaveHull::display(pcl::visualization::PCLVisualizer::Ptr viewer,
                          int vp,
                          double r,
                          double g,
                          double b) const
{
//    Eigen::Vector4d planeEq;
//    planeEq.head<3>() = plNormal;
//    planeEq(3) = -plD;
//    cout << "ConcaveHull::display, planeEq = " << planeEq.transpose() << endl;
    
    string idStr = to_string(reinterpret_cast<size_t>(this));
    
    for(int poly = 0; poly < polygons3d.size(); ++poly) {
        pcl::Vertices chullVertices;
        chullVertices.vertices.resize(polygons3d[poly].cols());
        iota(chullVertices.vertices.begin(), chullVertices.vertices.end(), 0);
        chullVertices.vertices.push_back(chullVertices.vertices.front());
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>());
        for (int p = 0; p < polygons3d[poly].cols(); ++p) {
            auto &pt = polygons3d[poly].col(p);
            points->push_back(pcl::PointXYZ(pt(0), pt(1), pt(2)));
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr chullPoints(new pcl::PointCloud<pcl::PointXYZ>());
        for (int p = 0; p < chullVertices.vertices.size(); ++p) {
            auto &pt = polygons3d[poly].col(chullVertices.vertices[p]);
            chullPoints->push_back(pcl::PointXYZ(pt(0), pt(1), pt(2)));
        }
        
        viewer->addPolygonMesh<pcl::PointXYZ>(points,
                                              vector<pcl::Vertices>{chullVertices},
                                              string("polygon_") + idStr +
                                              "_" + to_string(poly),
                                              vp);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 r,
                                                 g,
                                                 b,
                                                 string("polygon_") + idStr +
                                                 "_" + to_string(poly),
                                                 vp);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                 0.5,
                                                 string("polygon_") + idStr +
                                                 "_" + to_string(poly),
                                                 vp);
        viewer->addPolygon<pcl::PointXYZ>(chullPoints,
                                          1.0,
                                          0.0,
                                          0.0,
                                          string("polyline_") + idStr +
                                          "_" + to_string(poly),
                                          vp);
        viewer->addPointCloud(chullPoints,
                              string("polygon_pc_") + idStr +
                              "_" + to_string(poly),
                              vp);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                 0.0, 0.0, 1.0,
                                                 string("polygon_pc_") +
                                                 idStr +
                                                 "_" + to_string(poly),
                                                 vp);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 4,
                                                 string("polygon_pc_") +
                                                 idStr +
                                                 "_" + to_string(poly),
                                                 vp);
    }
}

void ConcaveHull::cleanDisplay(pcl::visualization::PCLVisualizer::Ptr viewer, int vp) const {
    string idStr = to_string(reinterpret_cast<size_t>(this));
    
    for(int poly = 0; poly < polygons3d.size(); ++poly) {
        viewer->removePolygonMesh(string("polygon_") + idStr +
                                  "_" + to_string(poly),
                                  vp);
        viewer->removeShape(string("polyline_") + idStr +
                            "_" + to_string(poly),
                            vp);
        viewer->removePointCloud(string("polygon_pc_") + idStr +
                                 "_" + to_string(poly),
                                 vp);
    }
}

void ConcaveHull::check() const {
    // for(int p = 0; p < polygons.size(); ++p){
    //     cout << "poly  " << p << endl;
    //     for (int i = 0; i < polygons[p].size(); ++i) {
    //         cout << polygons[p].vertex(i).x() << " " << polygons[p].vertex(i).y() << endl;
    //     }
    //     for(int i = 0; i < polygons[p].size(); ++i){
    //         cout << polygons3d[p].col(i).transpose() << endl;
    //         cout << point2dTo3d(polygons[p][i]).transpose() << endl;
    //     }
    // }

    for(int p = 0; p < polygons.size(); ++p){
        cout << "polygon " << p << endl;
        cout << "poly.is_counterclockwise_oriented() = " << polygons[p].is_counterclockwise_oriented() << endl;
        cout << "poly.is_simple() = " << polygons[p].is_simple() << endl;
        if (!polygons[p].is_counterclockwise_oriented() || !polygons[p].is_simple()) {
            for (int i = 0; i < polygons[p].size(); ++i) {
                cout << polygons[p].vertex(i).x() << " " << polygons[p].vertex(i).y() << endl;
            }
            for(int i = 0; i < polygons[p].size(); ++i){
                cout << polygons3d[p].col(i).transpose() << endl;
                cout << point2dTo3d(polygons[p][i]).transpose() << endl;
            }
        }
    }
}

void ConcaveHull::computeFrame() {
    // point on plane nearest to origin
    origin = plNormal * (plD);
    //if normal vector is not parallel to global x axis
    if(plNormal.cross(Eigen::Vector3d(1.0, 0.0, 0.0)).norm() > 1e-2){
        // plane x axis as a cross product - always perpendicular to normal vector
        xAxis = plNormal.cross(Eigen::Vector3d(1.0, 0.0, 0.0));
        xAxis.normalize();
        yAxis = plNormal.cross(xAxis);
    }
    else{
        xAxis = plNormal.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
        xAxis.normalize();
        yAxis = plNormal.cross(xAxis);
    }
}

ConcaveHull::Point_2 ConcaveHull::point3dTo2d(const Eigen::Vector3d &point3d) const {
    return Point_2((point3d - origin).dot(xAxis),
                 (point3d - origin).dot(yAxis));
}

ConcaveHull::Point_2ie ConcaveHull::point3dTo2die(const Eigen::Vector3d &point3d) const {
    return Point_2ie((point3d - origin).dot(xAxis),
                     (point3d - origin).dot(yAxis));
}

Eigen::Vector3d ConcaveHull::point2dTo3d(const ConcaveHull::Point_2 &point2d) const {
    return origin + CGAL::to_double(point2d.x()) * xAxis + CGAL::to_double(point2d.y()) * yAxis;
}
