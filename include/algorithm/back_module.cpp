#include "back_module.h"

void back_module::BackModuleLoop(){
    while (backend_running_.load()){
        std::unique_lock<std::mutex> lock(data_mutex);
        map_update.wait(lock);
        
        // optimize the active keyframes and landmarks
        Map_::umKeyFrame act_kfs = map_->GetActiveKeyFrames();
        Map_::umLandMark act_map = map_->GetActiveMapPoints();
        Optimize(act_kfs, act_map);
    }
}

void back_module::Optimize(Map_::umKeyFrame& kfs, Map_::umLandMark& ldms)
{
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    std::map <unsigned int, VertexPose *> vertices;
    unsigned int max_kf_id = 0;
    for (auto & keyframe: kfs){
        auto kf = keyframe.second;
        VertexPose * vertex_pose = new VertexPose(); // camera vertex pose
        vertex_pose->setId(kf->kf_id_);
        vertex_pose->setEstimate(kf->getPose());
        optimizer.addVertex(vertex_pose);
        if (kf->kf_id_ > max_kf_id) max_kf_id = kf->kf_id_;
        vertices.insert({kf->kf_id_, vertex_pose});
    }
    
    // map points
    std::map <unsigned int, VertexXYZ *> vertices_mappoint;
    Eigen::Matrix3d K = camera_->K();
    Sophus::SE3d left_ext = camera_->getLeftPose();
    Sophus::SE3d right_ext = camera_->getRightPose();
    
    // mappoints edges
    double chi2_th = 5.991;
    int index = 1;
    std::map <EdgeProjectionToPM *, pFeature> edges_and_features;
    for (auto & landmark : ldms){
        if (landmark.second->is_outlier) continue; // if this is out lier points
        unsigned int landmark_id = landmark.second->id_;
        auto observations = landmark.second->get_observe(); // observe is include ids and mappoints 
        for (auto & obs : observations){ // keyframes from this mappoints
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock(); // observed by features
            if (feat->is_outlier || feat->aFrame.lock() == nullptr) continue;
            auto frame_ = feat->aFrame.lock();
            
            EdgeProjectionToPM * edgeTpm = nullptr;
            if (feat->is_left_image){
                edgeTpm = new EdgeProjectionToPM(K, left_ext);
            }else{
                edgeTpm = new EdgeProjectionToPM(K, right_ext);
            }
            
            // if landmark is not being added into optimize add new vertex
            if (vertices_mappoint.find(landmark_id) == vertices_mappoint.end()){
                VertexXYZ * v = new VertexXYZ;
                v->setEstimate(landmark.second->getPosition());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_mappoint.insert({landmark_id, v});
                optimizer.addVertex(v);
            }
            
            edgeTpm->setId(index);
            edgeTpm->setVertex(0, vertices.at(frame_->kf_id_)); // postures all according frames id
            edgeTpm->setVertex(1, vertices_mappoint.at(landmark_id)); // mappoints
            edgeTpm->setMeasurement(Eigen::Vector2d(feat->position_.pt.x, feat->position_.pt.y));
            edgeTpm->setInformation(Matrix2d::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edgeTpm->setRobustKernel(rk);
            edges_and_features.insert({edgeTpm, feat});
            
            optimizer.addEdge(edgeTpm);
            index++;
        }
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5){
        cnt_outlier = 0; cnt_inlier = 0;
        for (auto & ef : edges_and_features){
            if (ef.first->chi2() > chi2_th){
                cnt_outlier++;
            }else{
                cnt_inlier++;
            }
        }
        double ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (ratio > 0.5) break;
        else{
            chi2_th *= 2;
            iteration++;
        }
    }
    
    for (auto & ef: edges_and_features){
        if (ef.first->chi2() > chi2_th){
            ef.second->is_outlier = true;
            ef.second->aMappoint.lock()->remove_observe(ef.second);
        }else{
            ef.second->is_outlier = false;
        }
    }
    
    LOG(INFO) << "Outlier/Inlier in Backend optimization: " << cnt_outlier << "/" << cnt_inlier;
    
    for (auto & v: vertices){
        kfs.at(v.first)->setPose(v.second->estimate());
    }
    for (auto & v: vertices_mappoint){
        ldms.at(v.first)->setPosition(v.second->estimate());
    }
}

back_module::back_module(){
    backend_running_.store(true);
    back_module_thread = std::thread(std::bind(&back_module::BackModuleLoop, this));
}

void back_module::StopBackModuleLoop() {
    backend_running_.store(false);
    map_update.notify_one();
    back_module_thread.join();
}

void back_module::UpdateMap()
{
    std::unique_lock<std::mutex> lock(data_mutex);
    map_update.notify_one();
}
