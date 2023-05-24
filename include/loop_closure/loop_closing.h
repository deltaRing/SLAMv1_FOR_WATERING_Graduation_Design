#ifndef _LOOP_CLOSING_H_
#define _LOOP_CLOSING_H_

#include "../mapping/posture.h"
#include "../define/camera.h"
#include "../define/map.h"
#include "dbow.h"
#include <set>

// 闭环检测原理: 若连续4个关键帧都能在数据库中找到对应的闭环匹配关键帧组,
// 且这些闭环匹配关键帧组间是连续的,则认为实现闭环,
//  KF-3      KF-2      KF-1        KF
//   ||        ||        ||         ||
//   \/        \/        \/         \/
//   KFD       KFD       KFD        KFD


// DetectLopp: Steps:
// 1.  Get vpCandidateKFs 闭环候选关键帧取自于与当前关键帧具有相同的BOW向量但不存在直接连接的关键帧
// 2.  闭环候选关键帧 vpCandidateKFs 和其共视关键帧 (previous KeyFrame) 组合成为关键帧
// 3.  当前关键组和之前的连续关键组间寻找连续关系
//      3.1 若当前关键帧组在之前的连续关键帧组中找到连续关系,则当前的连续关键帧组的连续长度加1
//      3.2 若当前关键帧组在之前的连续关键帧组中没能找到连续关系,则当前关键帧组的连续长度为0
// 4.  若某关键帧组的连续长度达到3,则认为该关键帧实现闭环.

// Simple:
// 1. Get Active KeyFrames Around This Frame
// 2. Check this KeyFrame and each KeyFrame and Select KeyFrame vocab > maxscore * 0.75
// 3. iterate these kfs, compute SE3 between currentKF and candidateKF
// 4. 

class LoopClosing{
public:
    void UpdatingLoop();
    bool ComputeSim3(); // get Sim3->currentKF ---> Sim3->candidateKF
    void CorrectLoop();
    int Optimalize(
                    pFrame kf1,
                    pFrame kf2, 
                    std::vector<pMapPoint> & kf1_matches,
                    std::vector<pMapPoint> & kf2_matches,
                    std::vector<pFeature> & kf1_feature,
                    std::vector<pFeature> & kf2_feature,
                    Sophus::SE3d & se3d12,
                    Sophus::SE3d & se3d21,
                    float th = 5.991
    ); // get Sim3
    bool DetectLoop(); // get loops
    void SetKeyFrame(pFrame _frame_){ 
        std::unique_lock<std::mutex> lck(data_mutex_);
        mpCurrentKF = _frame_;
        new_frame_arrived = true;
    } // insert keyframe
    void SetMap(pMap _map_){
        std::unique_lock<std::mutex> lck(data_mutex_);
        _pMap_ = _map_;
        map_initialized = true;
    }
    void SetCamera(pCamera _camera_){
        std::unique_lock<std::mutex> lck(data_mutex_);
        mpCamera = _camera_;
        camera_initialized = true;
    }
    void StartThread() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        stop_thread = false;
    }
    void StopThread() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        stop_thread = true;
    }
    void KillThread() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        kill_thread = true;
    }
    LoopClosing(){
        LOG(INFO) << "Need Initialize camera and map";
        loopClosingThread = std::thread(&LoopClosing::UpdatingLoop, this);
        loopClosingThread.detach();
    }
    LoopClosing(pMap map_, pCamera camera_){
        SetMap(map_);
        SetCamera(camera_);
        loopClosingThread = std::thread(&LoopClosing::UpdatingLoop, this);
        loopClosingThread.detach();
        camera_initialized = true;
        map_initialized = true;
    }
    
    std::vector<pFrame> vCandidateFrames;
    std::vector<pFrame> GetCandidateFrames(float score);
    
    Sophus::SE3d vCorrectSE3d;
    pFrame vRefinedFrames;
    std::vector <pMapPoint> vRefinedCurrentMapPoints;
    std::vector <pMapPoint> vRefinedSelectMapPoints;
    
private:
    std::mutex data_mutex_;
    std::thread loopClosingThread;
    unsigned int least_continue_frames = 3;
    unsigned int last_closure_kf_id = 0;
    unsigned int minimal_frame_ = 10;  
    unsigned int minimal_map_keyframes_ = 10;
    unsigned int minimal_accept_matches_ = 30;
    
    pFrame mpCurrentKF = nullptr; // 当前关键帧
    pFrame mpMatchedKF = nullptr; // 当前关键帧的闭环匹配关键帧
    pCamera mpCamera = nullptr;   // camera ptr
    pMap _pMap_ = nullptr;
    bool new_frame_arrived = false;
    bool kill_thread = false;
    bool stop_thread = false;
    bool camera_initialized = false;
    bool map_initialized = false;
};

typedef std::shared_ptr<LoopClosing> pLoopClosing;

#endif
