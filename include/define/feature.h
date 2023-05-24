#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <thread>

#include "frame.h"

class frame;
class mappoint;

class feature{
public:
    cv::KeyPoint position_; // extract location
    std::weak_ptr<frame> aFrame; // associated frames
    std::weak_ptr<mappoint> aMappoint; // associated mappoints
    
    bool is_outlier = false;
    bool is_left_image = true; // using left image by default
    
public:
    feature(){}
    feature (std::shared_ptr<frame> frame_, const cv::KeyPoint & kp) {
        aFrame = frame_;
        position_ = kp;
    }
};

typedef std::weak_ptr<mappoint> wPmap;
typedef std::weak_ptr<frame> wPframe;

#endif
