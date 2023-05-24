#ifndef _BACK_MODULE_H_
#define _BACK_MODULE_H_

#include "../mapping/posture.h"
#include "../define/mappoint.h"
#include "../define/camera.h"
#include "../define/frame.h"
#include "../define/map.h"

#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>

class back_module{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    back_module();
    void SetCamera(pCamera _camera_) { camera_ = _camera_; } // set 
    void SetMap(pMap _map_) { map_ = _map_; } // set map
    void UpdateMap(); // trigger maps
    void StopBackModuleLoop();
    
private:
    void BackModuleLoop();
    // map_update_ is triggered and active keyframes and mappoints are updated 
    void Optimize(Map_::umKeyFrame & kfs, Map_::umLandMark & ldms);
    
    std::thread back_module_thread;
    std::mutex data_mutex;
    std::condition_variable map_update;
    std::atomic<bool> backend_running_; // ?
    
    pCamera camera_ = nullptr;
    pMap map_ = nullptr;
};

typedef std::shared_ptr<back_module> pBackModule;

#endif
