#include <iostream>
#include "include/mapping/pointclouds_map.h"
#include "include/datapath/datapath.h"
#include "include/define/flags.h"
#include "include/algorithm/front_module.h"
#include "include/algorithm/back_module.h"
#include "include/loop_closure/loop_closing.h"
#include "include/loop_closure/dbow.h"

extern std::string timeStampPath;
extern std::string leftImagePath;
extern std::string rightImagePath;
extern std::string depthImagePath;

int main(int argc, char **argv) {
    //Sophus::SE3d tt;
    //Sophus::SE3d rr;
    //(tt * rr).matrix();
    
    load_vocab(true); // load the weight
    // if weight not exists train it
    
    //std::cout << tt.matrix() << std::endl;
    //std::cout << rr.matrix() << std::endl;
    //std::cout << (tt * rr).matrix() << std::endl;
    // Initialize part
    front_module * fm = new front_module(); // load datasets automatically
    pBackModule bm = pBackModule(new back_module());
    pLoopClosing lc = pLoopClosing(new LoopClosing());
    pPointCloudMap pc = pPointCloudMap(new PointCloudMap());
    // back_module * bm = nullptr; TODO 
    pMap map_ = pMap(new Map_());
    pViewer viewer_ = pViewer(new viewer());
    pCamera camera_ = pCamera(new camera());
    
    fm->SetBackModule(bm);
    fm->SetViewer(viewer_);
    fm->SetCamera(camera_);
    fm->SetDenseMap(pc, true);
    fm->SetMap(map_);
    
    bm->SetMap(map_);
    bm->SetCamera(camera_);
    
    viewer_->SetMap(map_);
    
    lc->SetMap(map_);
    lc->SetCamera(camera_);
    fm->SetLoopClosing(lc);
    
    pFrame frames = nullptr;
    // Running Part
    while (fm->GetNextFrame(frames)){
        fm->AddNewFrame(frames);
    }
    
    pc->save_map();
    return 0;
}
