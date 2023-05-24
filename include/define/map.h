#ifndef _MAP_H_
#define _MAP_H_

#include "frame.h"
#include "feature.h"
#include "mappoint.h"
// glog
#include <glog/logging.h>

typedef std::shared_ptr<frame> pFrame;
typedef std::shared_ptr<feature> pFeature;
typedef std::shared_ptr<mappoint> pMapPoint;

class Map_{
public:
    typedef std::unordered_map<unsigned int, pMapPoint> umLandMark;
    typedef std::unordered_map<unsigned int, pFrame> umKeyFrame;
    std::mutex data_mutex_;
    umLandMark landmarks_; // all landmarks 
    umLandMark active_landmarks_; // active landmarks
    umKeyFrame keyframes_; // all keyframes
    umKeyFrame active_keyframes_; // active keyframes
    
    pFrame current_frame_ = nullptr;
    
    // number of active keyframes
    int num_active_keyframes_ = 10;
    
    Map_() {
        
    }
    
    // clear all points
    void CleanMap(){
        int cnt_landmark_removed = 0;
        for (auto iter = active_landmarks_.begin();
            iter != active_landmarks_.end();) {
            if (iter->second->observe_times == 0) {
                iter = active_landmarks_.erase(iter);
                cnt_landmark_removed++;
            } else {
                ++iter;
            }
        }
        LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
    }
private:
    // remove too old key frames
    void RemoveOldKeyFrame(){
        if (current_frame_ == nullptr){
            return;
        }
        // get minimal and maximum distance of kf
        double max_dis = 0.0, min_dis = 9999.9;
        double max_kf_id = 0.0, min_kf_id = 0.0;
        auto Twc = current_frame_->getPose().inverse();
        for (auto & kf : active_keyframes_){
            if (kf.second == current_frame_){
                continue;
            }
            auto dis = (kf.second->getPose() * Twc).log().norm();
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }
        
        const double min_dis_th = 0.2;  // 最近阈值
        pFrame frame_to_remove = nullptr;
        if (min_dis < min_dis_th) {
            // 如果存在很近的帧，优先删掉最近的
            frame_to_remove = keyframes_.at(min_kf_id);
        } else {
            // 删掉最远的
            frame_to_remove = keyframes_.at(max_kf_id);
        }
        LOG(INFO) << "remove keyframe " << frame_to_remove->kf_id_;
        // remove keyframe and landmark observation
        active_keyframes_.erase(frame_to_remove->kf_id_);
        for (auto feat : frame_to_remove->aFeaturesLeft) {
            auto mp = feat->aMappoint.lock();
            if (mp) {
                mp->remove_observe(feat);
            }
        }
        
        for (auto feat : frame_to_remove->aFeaturesRight) {
            if (feat == nullptr) continue;
            auto mp = feat->aMappoint.lock();
            if (mp) {
                mp->remove_observe(feat);
            }
        }

        CleanMap();
    }
    
public:
    // add a keyframe
    void InsertKeyFrame(pFrame frame_){
        current_frame_ = frame_;
        if (keyframes_.find(frame_->kf_id_) == keyframes_.end()){
            keyframes_.insert(make_pair(frame_->kf_id_, frame_));
            active_keyframes_.insert(make_pair(frame_->kf_id_, frame_));
        }else{
            keyframes_[frame_->kf_id_] = frame_;
            active_keyframes_[frame_->kf_id_] = frame_;
        }
        
        if (active_keyframes_.size() > num_active_keyframes_) {
            this->RemoveOldKeyFrame();
        }
    }
    // add map point
    void InsertMapPoint(pMapPoint mappoint_){
        if (landmarks_.find(mappoint_->id_) == landmarks_.end()) {
            landmarks_.insert(make_pair(mappoint_->id_, mappoint_));
            active_landmarks_.insert(make_pair(mappoint_->id_, mappoint_));
        } else {
            landmarks_[mappoint_->id_] = mappoint_;
            active_landmarks_[mappoint_->id_] = mappoint_;
        }
    }
    // get all of mappoints
    umLandMark GetAllMapPoints(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    // active keyframes
    umKeyFrame GetAllKeyFrames(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }
    // get active of mappoints
    umLandMark GetActiveMapPoints(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }
    // get active of keyframes
    umKeyFrame GetActiveKeyFrames(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }
};

typedef std::shared_ptr<Map_> pMap;

#endif
