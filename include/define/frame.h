#ifndef _FRAME_H_
#define _FRAME_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>

#include "feature.h"
#include "map.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp> // Posture
#include <DBoW3/DBoW3.h>

using namespace Eigen;

class feature;

class frame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cv::Mat imLeft, imRight, imDepth;
    std::vector <std::shared_ptr<feature>> aFeaturesLeft; // Associated Features
    std::vector <std::shared_ptr<feature>> aFeaturesRight;
    DBoW3::BowVector bv; // compute the dbow3 vector by using descriptor
    cv::Mat descriptors; // descriptors of image
    
    // weights keyframes
    std::vector <std::pair<int, std::weak_ptr<frame>>> coherent_keyframe; // 共视的帧
    unsigned int coherent_keyframe_num = 35;
    
    unsigned int id_ = 0; // id of this frame
    unsigned int kf_id_ = 0; // id of keyframe
    bool keyframe_flag_ = false; // is this keyframe
    Sophus::SE3d pose_; // current posture
    std::mutex frame_mutex_; // data mutex
public:
    frame(){}
    frame(unsigned int id, const Sophus::SE3d & pose, const cv::Mat & imLeft, const cv::Mat & imRight){
        id_ = id;
        pose_ = pose;
        this->imLeft = imLeft.clone();
        this->imRight = imRight.clone();
    }
    
    std::vector<std::shared_ptr<feature>> getFeature(){
        std::unique_lock<std::mutex> lck(frame_mutex_);
        return aFeaturesLeft;
    }
    
    Sophus::SE3d getPose(){
        std::unique_lock<std::mutex> lck(frame_mutex_);
        return pose_;
    }
    
    void setPose(const Sophus::SE3d & pose){
        std::unique_lock<std::mutex> lck(frame_mutex_);
        pose_ = pose;
    }
    
    void setKeyFrame(){
        static unsigned int keyframe_id = 0;
        kf_id_ = keyframe_id++;
        keyframe_flag_ = true;
    }
    
    static std::shared_ptr<frame> createFrame(){
        static int factory_id = 0;
        std::shared_ptr<frame> new_frame(new frame);
        new_frame->id_ = factory_id++;
        return new_frame;
    }
    
    void addCoherentFrame(std::pair<int, std::weak_ptr<frame>> Pairs){
        coherent_keyframe.push_back(Pairs);
    }
    
    std::vector <std::pair<int, std::weak_ptr<frame>>> getCoherentFrames(){
        std::unique_lock<std::mutex> lck(frame_mutex_);
        return coherent_keyframe;
    }
};

#endif
