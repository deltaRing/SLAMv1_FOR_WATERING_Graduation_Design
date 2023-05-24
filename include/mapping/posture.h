#ifndef _POSTURE_H_
#define _POSTURE_H_

#include "../define/map.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define _MAXIMUM_POINT_ 200
#define _MAXIMUM_POINT_ORB_ 2000
#define _scaleFactor_ 1.2
#define _nlevels_ 12

#include <Eigen/Core>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <chrono>

#undef SOPHUS_ENSURE

// g2o optimalizer
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

// get descriptors for keyframe
void GetDescriptor(pFrame kf);
// pixel -> camera
cv::Point2f pixel2cam(const cv::Point2d &p);
// Get Postures From images
void initComponents();
// get keypoints via multi-scale
std::vector<cv::KeyPoint> GetKeypointsMultiScale(cv::Mat Images, int scalefactor=12);
// compute oriented FAST points
std::vector<cv::KeyPoint> GetKeypoints(cv::Mat images, bool _USE_ORB_=false);
// compute Brief
cv::Mat GetBRIEF(cv::Mat images, std::vector<cv::KeyPoint> keypoints);
// compute matches
std::vector<cv::DMatch> GetMatches(cv::Mat descriptor1, cv::Mat descriptor2);
// filter matches
std::vector<cv::DMatch> FilterMatches(std::vector<cv::DMatch> matches, cv::Mat _descriptor_);
// 以上函数都是为了获得关键点

// init camera models D430
void initCameraParameter();
// Get transformation
bool pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<cv::DMatch> matches,
                          cv::Mat &R, cv::Mat &t);
// Get transition by using LK optical flow
bool pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<uchar> status,
                          std::vector<float> error,
                          cv::Mat &R, cv::Mat &t);
// Get filtered points
void orb_key_point_filter(std::vector<cv::KeyPoint> kps, 
                          std::vector<cv::KeyPoint> & best_kp,
                          cv::Point2i node_start,
                          cv::Point2i cell_size,
                          int current_level,
                          int max_level = 5
                         );
// Get optical flow
// input: old keypoints, images old and new
// output: keypoints status error of keypoints track_num
void calculate_optFlow(
    std::vector<cv::KeyPoint> kp_old, 
    std::vector<cv::KeyPoint> & kp_new,
    cv::Mat img_old,
    cv::Mat img_new,
    std::vector<uchar> & status, // if status is 1 is correctly tracked
    std::vector<float> & err,
    int & track_num
);

// get mappoint's depth
// using left_and_right
// using old and new
// input: kp_old_frame kp_new_frame
// input: matches position_old position_new
// output: points_3d
void triangulation(
    std::vector<cv::KeyPoint> &keypoint_1,
    std::vector<cv::KeyPoint> &keypoint_2,
    std::vector<cv::DMatch> &matches,
    cv::Mat &R1, const cv::Mat &t1,
    cv::Mat &R2, const cv::Mat &t2,
    std::vector<cv::Point3d> & points);

// g2o optimalizer part
// optimal 3d->2d
// vertex definition
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // init data number
    virtual void setToOriginImpl() override{
        _estimate = Sophus::SE3d();
    }
    // update SE3d
    virtual void oplusImpl(const double * update) override{
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }
    virtual bool read(std::istream & in) override { return true; }
    virtual bool write(std::ostream & out) const override { return true; }
    
    void SetCamera1Param(double fx, double fy, double cx, double cy){
        focus_length1     = Eigen::Vector2d(fx, fy);
        _principle_point1 = Eigen::Vector2d(cx, cy);
    }
    
    void SetCamera2Param(double fx, double fy, double cx, double cy){
        focus_length2     = Eigen::Vector2d(fx, fy);
        _principle_point2 = Eigen::Vector2d(cx, cy);
    }
    
    Eigen::Vector3d map(Sophus::SE3d se3, const Eigen::Vector3d & mappoint_) const {
        return se3 * mappoint_;
    }
private:
    Eigen::Vector2d focus_length1, focus_length2,
            _principle_point1, _principle_point2;
};

// location only                                     connection Vertex types
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}
    
    virtual void computeError() override {
        const VertexPose * v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d); // --> pixels
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>(); // real pix--> the mappoints
    }
    
    virtual void linearizeOplus() override {
        const VertexPose * v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);
        double x = pos_cam[0], y = pos_cam[1], z = pos_cam[2];
        double x2 = x * x, y2 = y * y, z2 = z * z;
        _jacobianOplusXi << -fx / z, 0, 
        fx * x / z2, fx * x * y / z2, 
        -fx - fx * x2 / z2, fx * y / z,
                        0, -fy / z,
                        fy * y / z2, fy + fy * y2 / z2, 
                        -fy * x * y / z2, -fy * x / z;
    }
    
    virtual bool read(std::istream & in) override { return true; }
    
    virtual bool write(std::ostream & out) const override { return true; }
private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

// edges between 2 vertexs
class SE3dOptimalize : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexPose, VertexPose>{
public:
    virtual bool read(std::istream& is) { return true; }
    virtual bool write(std::ostream& os) { return true; }
    virtual void computeError(){
        const VertexPose* v1 = static_cast<const VertexPose*>(_vertices[0]);
        const VertexPose* v2 = static_cast<const VertexPose*>(_vertices[1]);
        
        Sophus::SE3d C(_measurement);
        Sophus::SE3d error_ = C * v1->estimate() * v2->estimate().inverse(); 
        // V1_cw V2_wc --> C * V_12
        _error = error_.log();
    }
};

/// 路标顶点
class VertexXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void setToOriginImpl() override { _estimate = Eigen::Vector3d::Zero(); }

    virtual void oplusImpl(const double *update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

// optimalize for both posture and mappoint
class EdgeProjectionToPM : public g2o::BaseBinaryEdge <2, Eigen::Vector2d, VertexPose, VertexXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectionToPM(const Eigen::Matrix3d & K, const Sophus::SE3d & cam_ext){
        _K = K;
        _cam_ext = cam_ext;
    }
    
    virtual void computeError() override {
        const VertexPose * v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ * v1 = static_cast<VertexXYZ *>(_vertices[1]);
        Sophus::SE3d T = v0->estimate();
        Eigen::Vector3d pos_pixel = _K * (_cam_ext * (T * v1->estimate()));
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>(); // difference of pixels
    }
    
    virtual void linearizeOplus() override {
        const VertexPose * v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ * v1 = static_cast<VertexXYZ *>(_vertices[1]);
        Sophus::SE3d T = v0->estimate();
        Eigen::Vector3d pw = v1->estimate();
        Eigen::Vector3d pos_cam = _cam_ext * T * pw;
        double fx = _K(0, 0), fy = _K(1, 1), X = pos_cam[0], Y = pos_cam[1], Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1.0e-18);
        double Zinv2 = Zinv * Zinv;
        
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv; // this is estimating postures of 
        
        // 投影点导数 * 内参 * 旋转
        _jacobianOplusXj << _jacobianOplusXi.block<2, 3>(0, 0) *
                           _cam_ext.rotationMatrix() * T.rotationMatrix();
        // this is estimating the vertex of map points
        
    }
    
    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
    
private:
    Eigen::Matrix3d _K;
    Sophus::SE3d _cam_ext;
};


// optimalize for both posture and mappoint
class EdgeProjectionToPMInverse : public g2o::BaseBinaryEdge <2, Eigen::Vector2d, VertexPose, VertexXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectionToPMInverse(const Eigen::Matrix3d & K, const Sophus::SE3d & cam_ext){
        _K = K;
        _cam_ext = cam_ext;
    }
    
    virtual void computeError() override {
        const VertexPose * v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ * v1 = static_cast<VertexXYZ *>(_vertices[1]);
        Sophus::SE3d T = v0->estimate().inverse();
        Eigen::Vector3d pos_pixel = _K * (_cam_ext * (T * v1->estimate()));
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>(); // difference of pixels
    }
    
    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
    
private:
    Eigen::Matrix3d _K;
    Sophus::SE3d _cam_ext;
};


// original 3d position-> new 2d pixel
// optimize posture
void bundleAdjustmentG2O(
    const std::vector<Eigen::Vector3d> &points_3d,
    const std::vector<Eigen::Vector2d> &points_2d,
    Sophus::SE3d &pose);

#endif
