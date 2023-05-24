#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mutex>

class camera{
public:
    Sophus::SE3d leftPose;
    Sophus::SE3d RightPose;
    Sophus::SE3d leftPose_inv;
    Sophus::SE3d RightPose_inv;
    std::mutex data_mutex_; // data mutex to lock down data
    double fx_ = 382.613, fy_ = 382.613, cx_ = 320.183, cy_ = 236.455,
           baseline_ = 0.05;  // Camera intrinsics
    
    camera (){
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_l(0, 0, 0);
        Eigen::Vector3d t_r(0.0499585, 0, 0);
        leftPose = Sophus::SE3d(R, t_l);
        RightPose = Sophus::SE3d(R, t_r);
        leftPose = leftPose.inverse();
        RightPose = RightPose.inverse();
    }
    
    Eigen::Matrix3d K() const{
        Eigen::Matrix3d k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }
    
    Sophus::SE3d getLeftPose(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return leftPose;
    }
    
    Sophus::SE3d getRightPose(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return RightPose;
    }
    
    // coordinate transform: world, camera, pixel
    Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w, bool posture_select=true){
        if (posture_select)
            return leftPose * T_c_w * p_w;
        else
            return RightPose * T_c_w * p_w;
    } // default by using left camera

    Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w, bool posture_select=true){
        if (posture_select)
            return T_c_w.inverse() * leftPose_inv * p_c;
        else
            return T_c_w.inverse() * RightPose_inv * p_c;
    }

    Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c){
        return Eigen::Vector2d(
            fx_ * p_c(0) / p_c(2) + cx_,
            fy_ * p_c(1) / p_c(2) + cy_
        );
    }

    Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1){
        return Eigen::Vector3d(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth
        );
    }

    Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth = 1){
        return camera2world(pixel2camera(p_p, depth), T_c_w);
    }

    Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w){
        return camera2pixel(world2camera(p_w, T_c_w));
    }
};

typedef std::shared_ptr<camera> pCamera;

#endif
