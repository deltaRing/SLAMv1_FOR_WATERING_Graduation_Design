#ifndef _MAPPOINT_H_
#define _MAPPOINT_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <vector>
#include <memory>
#include <iostream>

#include "feature.h"

class feature;
class frame;

class mappoint{
public:
    unsigned int id_ = 0; // id number of this mappoint
    std::list <std::weak_ptr<feature>> aFeatures; // Associated Features and observed by these frames
    Eigen::Vector3d location; // position of map points
    bool is_outlier = false; // is this healthy?
    unsigned int observe_times = 0; // observe times
    std::mutex data_mutex_; // 
    
    mappoint () {}
    mappoint (unsigned int id, Eigen::Vector3d position) {
        id_ = id;
        location = position;
    }
    
    // set mappoint position
    void setPosition(const Eigen::Vector3d & pos){
        std::unique_lock<std::mutex> lck(data_mutex_);
        location = pos;
    }
    // get position of map point
    // world position
    Eigen::Vector3d getPosition(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return location;
    }
    // add observe time of mappoint
    void add_observe(std::shared_ptr<feature> feat){
        std::unique_lock<std::mutex> lck(data_mutex_);
        aFeatures.push_back(feat);
        observe_times++;
    }
    // check if this mappoint is out lier
    void remove_observe(std::shared_ptr<feature> feat){
        std::unique_lock<std::mutex> lck(data_mutex_);
        for (auto iter = aFeatures.begin(); iter != aFeatures.end(); iter++) {
            if (iter->lock() == feat) {
                aFeatures.erase(iter);
                feat->aMappoint.reset();
                observe_times--;
                break;
            }
        }
    }
    // get features times
    std::list<std::weak_ptr<feature>> get_observe(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return aFeatures;
    }
    // create mappoint
    static std::shared_ptr<mappoint> create_new_mappoint(){
        static long id_ = 0;
        std::shared_ptr<mappoint> new_mappoint(new mappoint);
        new_mappoint->id_ = id_++;
        return new_mappoint;
    }
};

#endif
