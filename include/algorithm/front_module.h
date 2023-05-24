#ifndef _FRONT_MODULE_H_
#define _FRONT_MODULE_H_

#include <vector>
// std data structure
#include "../define/map.h"
#include "../define/flags.h"
#include "../define/frame.h"
#include "../define/camera.h"
#include "../define/feature.h"
#include "../define/mappoint.h"
#include "../mapping/posture.h"
#include "../datapath/datapath.h"

#include "../viewer/viewer.h"
#include "../algorithm/back_module.h"
#include "../loop_closure/loop_closing.h"
#include "../loop_closure/dbow.h"
#include "../mapping/pointclouds_map.h"

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

extern std::string timeStampPath;
extern std::string leftImagePath;
extern std::string rightImagePath;
extern std::string depthImagePath;

#define _FRONT_STATUS_TRACK_INIT_ 0
#define _FRONT_STATUS_TRACK_GOOD_ 1
#define _FRONT_STATUS_TRACK_BAD_ 2
#define _FRONT_STATUS_TRACK_LOST_ 3

// glog
#include <glog/logging.h>

class mappoint;
class feature;
class camera;
class frame;
class Map_;

class front_module{
private:
    pPointCloudMap point_cloud_ = nullptr;
    pLoopClosing loop_closure_ = nullptr;
    pBackModule back_module_ = nullptr;
    pFrame current_frame_ = nullptr;
    pFrame last_frame_ = nullptr;
    pCamera camera_ = nullptr;
    pMap map_ = nullptr;
    cv::Mat R_cw; // rotation camera->world
    cv::Mat t_cw; // transition camera->world
    cv::Mat cImRight, cImLeft, cImDepth; // current image
    unsigned int status_ = _FRONT_STATUS_TRACK_INIT_; // 
    Sophus::SE3d relative_motion_; // movements
    int current_frame_index = 0;
    int tracking_inliers_ = 0; // tracked point numbers
    int num_features_tracking_ = 50; //
    int num_features_tracking_bad_ = 30; // 
    int num_features_init_ = 80; // at least need 150 points to init
    int num_features_needed_for_keyframe_ = 80;
    int num_maximum_features_ = 450; // keep only 450 features at max
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    std::vector<std::string> vstrImageLeft;
    std::vector<std::string> vstrImageRight;
    std::vector<std::string> vstrImageDepth;
    std::vector<double> vTimeStamps; // file path
    
    bool use_viewer_ = false;
    bool use_dense_map_ = false;
    pViewer viewer_ = nullptr;
    
    void SetDenseMap(pPointCloudMap point_cloud_, bool use_dense_map_){
        if (point_cloud_ != nullptr){
            this->point_cloud_ = point_cloud_;
            this->use_dense_map_ = use_dense_map_;
        }
    }
    
    void SetLoopClosing(pLoopClosing loop_closing_){
        if (loop_closing_ != nullptr){
            this->loop_closure_ = loop_closing_;
        }
    }
    
    void SetBackModule(pBackModule back_module_){
        if (back_module_ != nullptr){
            this->back_module_ = back_module_;
        }
    }
    
    void SetViewer(pViewer viewer_){
        if (viewer_ != nullptr){
            use_viewer_ = true;
            this->viewer_ = viewer_;
        }
    }
    void SetCamera(pCamera camera_){
        if (camera_ != nullptr){
            this->camera_ = camera_;
        }
    }
    void SetMap(pMap map_){
        if (map_ != nullptr){
            this->map_ = map_;
        }
    }
    
    front_module () {
        R_cw = (cv::Mat_<double>(3, 3) << 
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0);
        t_cw = (cv::Mat_<double>(3, 1) << 
                0.0,
                0.0,
                0.0);
        
        std::cout << "SLAMV1 by Laoxiang" << std::endl;
        initComponents();
        initCameraParameter();
        
        // Load Parameters of RealSense D430
        
        LoadImagesPath(leftImagePath,
                   rightImagePath,
                   depthImagePath,
                   timeStampPath, 
                   vstrImageLeft,
                   vstrImageRight,
                   vstrImageDepth,
                   vTimeStamps);
        std::cout << "Loaded Images Left: " << vstrImageLeft.size() << std::endl;
        std::cout << "Loaded Images Right: " << vstrImageRight.size() << std::endl;
        std::cout << "Loaded Images Depth: " << vstrImageDepth.size() << std::endl;
        std::cout << "Loaded Images TimeStamp: " << vTimeStamps.size() << std::endl;
        
        unsigned int numLeft = vstrImageLeft.size();
        unsigned int numRight = vstrImageRight.size();
        unsigned int numDepth = vstrImageDepth.size();
        unsigned int numTimeStamp = vTimeStamps.size();
        
        if (numLeft != numRight || numLeft != numDepth || numLeft != numTimeStamp){
            std::cerr << "Loaded data: DATA number is not correct" << std::endl;
            return;
        }else{
            
        }
    }
    // Get next frame
    bool GetNextFrame(pFrame & newFrame);
    // Detect Features
    int DetectFeatures();
     /**
     * Set the features in keyframe as new observation of the map points
     */
    void SetObservationsForKeyFrame();
    // estimate pose and optimalize 
    int EstimateCurrentPose(); 
    // Insert key frame
    bool InsertKeyFrame();
    // find features from right images
    int FindFeaturesInRight();
    // get number of last frame
    int TrackLastFrame();
    // StereoInit to track points
    bool StereoInit();
    // building initial mapping
    bool BuildInitMap(int num_features);
    // Triangulate
    bool triangulate(std::vector<Sophus::SE3d> & poses, 
        const std::vector<Eigen::Vector3d> points,
        Eigen::Vector3d & pt_world);
    // track function get relative motion from new frame and last frame
    bool Track();
    // if track is lost
    bool Reset(){
        LOG(INFO) << "Tracking Lost Exiting..." << std::endl;
        StereoInit();
        status_ = _FRONT_STATUS_TRACK_INIT_;
        // exit(0);
        return true;
    }
    // get new points
    bool TriangulateNewMappoints(int num_detected_points);
    // get new frame
    bool AddNewFrame(pFrame frame_){
        current_frame_ = frame_;
        if (status_ == _FRONT_STATUS_TRACK_INIT_){
            StereoInit();
        }else if (status_ == _FRONT_STATUS_TRACK_GOOD_ || status_ == _FRONT_STATUS_TRACK_BAD_){
            Track();
        }else if (status_ == _FRONT_STATUS_TRACK_LOST_){
            Reset();
        }else{
            // UNKOWNN STATUS
            LOG(INFO) << "unknown status is detected" << std::endl;
            return false;
        }
        last_frame_ = current_frame_;
        return true;
    }
    // get coherent keyframe
    void GetCoherentWeights();
};

#endif 
