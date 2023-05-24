#ifndef _VIEWER_H_
#define _VIEWER_H_

#include <pangolin/pangolin.h>
// dense mapping
#include "../mapping/pointclouds_map.h"
// global mapping
#include "../define/map.h"

#include <vector>
#include <thread>
#include <mutex>

using namespace std;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace Eigen;


class viewer{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    viewer();

    void SetMap(pMap map) { map_ = map; }
 
    void Close();

    // 增加一个当前帧
    void AddCurrentFrame(pFrame current_frame);

    // 更新地图
    void UpdateMap();

private:
    void ThreadLoop();

    void DrawFrame(pFrame frame, const float* color);

    void DrawMapPoints();
    
    // void DrawPointMapPoints();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    pFrame current_frame_ = nullptr;
    pMap map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

    std::unordered_map<unsigned int, pFrame> active_keyframes_;
    std::unordered_map<unsigned int, pMapPoint> active_landmarks_;
    bool map_updated_ = false;
    
    std::vector <pangolin::OpenGlMatrix> Twcs;
    unsigned int _maximum_posture_ = 200;

    std::mutex viewer_data_mutex_;  
};

typedef std::shared_ptr<viewer> pViewer;

#endif
