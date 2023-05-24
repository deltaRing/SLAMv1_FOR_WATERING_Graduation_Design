#include "loop_closing.h"

//
void LoopClosing::CorrectLoop(){
    if (mpCurrentKF == nullptr) return;
    Sophus::SE3d cTcw = mpCurrentKF->getPose();
    Sophus::SE3d cTwc = cTcw.inverse();
    
    // Get Twc 
    std::vector <pair<int, std::weak_ptr<frame>>> vConnectedFrame = mpCurrentKF->getCoherentFrames();
    std::vector <Sophus::SE3d> vCorrectedSE3toKeyFrames;
    std::vector <Sophus::SE3d> vNotCorrectedSE3toKeyFrames;
    
    for (auto & vit : vConnectedFrame){
        pFrame pKFi = vit.second.lock();
        Sophus::SE3d Tiw = pKFi->getPose();
        Sophus::SE3d Tic = Tiw * cTwc;
        Sophus::SE3d Tcorrect = Tic * vCorrectSE3d;
        vCorrectedSE3toKeyFrames.push_back(Tcorrect);
        vNotCorrectedSE3toKeyFrames.push_back(Tiw);
    }
    
    int kf_index = 0;
    for (auto & mit : vCorrectedSE3toKeyFrames){
        pFrame pKFi = vConnectedFrame[kf_index].second.lock();
        Sophus::SE3d Tcorrectediw = mit;
        Sophus::SE3d Tcorrectedwi = mit.inverse();
        Sophus::SE3d Tnoniw = vNotCorrectedSE3toKeyFrames[kf_index];
        std::vector<pFeature> features = pKFi->getFeature();
        for (auto & feat : features){
            if (feat->aMappoint.expired()){
                continue;
            }
            pMapPoint map_point = feat->aMappoint.lock();
            Eigen::Vector3d P3Dw = map_point->getPosition();
            Eigen::Vector3d correctedP3Dw = Tcorrectedwi * Tnoniw * P3Dw;
            map_point->setPosition(correctedP3Dw);
        }
        pKFi->setPose(Tcorrectediw);
        kf_index++;
    }
    
    // non LoopClosure is done to this program
    last_closure_kf_id = mpCurrentKF->kf_id_;
}

// 
void LoopClosing::UpdatingLoop(){
    while (true){
        if (!map_initialized){
            continue;
        }
        if (!camera_initialized){
            continue;
        }
        if (stop_thread){
            continue;
        }
        if (kill_thread){
            return;
        }
        if (new_frame_arrived){
            new_frame_arrived = false;
            if (DetectLoop()){
                if (ComputeSim3()){
                    CorrectLoop();
                }
            }
        }
    }
}

bool LoopClosing::DetectLoop(){
    if (mpCurrentKF->kf_id_ - last_closure_kf_id < minimal_frame_){
        LOG(INFO) << "Current Frame is too close to last Last closing";
        return false;
    }
    
    // step1. 取出缓冲队列头部的关键帧,作为当前检测闭环关键帧
    {
        unique_lock<std::mutex> data_mutex_;
        // get data
        if (mpCurrentKF == nullptr){
            LOG(INFO) << "Current keyframe is nullptr!";
            return false;
        }
    }
    
    // step2. 
    const DBoW3::BowVector CurrentBowVec = mpCurrentKF->bv;
    float minScore = 1.0;
    std::vector <pair<int, std::weak_ptr<frame>>> lc_cFrames = mpCurrentKF->getCoherentFrames();
    for (int ii = 0; ii < lc_cFrames.size(); ii++){
         wPframe wpKF = lc_cFrames[ii].second;
         if (wpKF.expired()) continue;
         const DBoW3::BowVector BowVector = wpKF.lock()->bv;
         float score = score_vocab(CurrentBowVec, BowVector);
         if (score < minScore)
             minScore = score;
    }
    
    // step3. get candidate frames 
    this->vCandidateFrames = GetCandidateFrames(minScore);
    
    if (this->vCandidateFrames.size() == 0){
        LOG(INFO) << "No candidate frames are found";
        return false;
    }
    
    LOG(INFO) << "Loop detected!";
    return true;
}

// Get candidate frames in loops
std::vector<pFrame> LoopClosing::GetCandidateFrames(float minscore){
    std::vector<pFrame> vCandidateFrames;
    std::vector<float>  vCandidateFrameScore;
    std::vector<pFrame> vTempFrames;
    std::vector<pFrame> vFrames;
    // get all keyframes
    Map_::umKeyFrame keyframes = _pMap_->GetAllKeyFrames();
    vCandidateFrames.reserve(2500);
    
    // compute the similarity of all keyframes
    for (auto & kps: keyframes){
        pFrame frame_ = kps.second;
        bool _is_keyframe_covisible_ = false;
        for (int ii = 0; ii < mpCurrentKF->coherent_keyframe.size(); ii++){
            if (frame_->kf_id_ == mpCurrentKF->coherent_keyframe[ii].second.lock()->kf_id_ ||
                frame_->kf_id_ == mpCurrentKF->kf_id_
            ){ // we dont compute current kfs
                _is_keyframe_covisible_ = true;
                break;
            }
        }
        if (_is_keyframe_covisible_)
            continue; // this frame is covisible frame continue
        float score = score_vocab(mpCurrentKF, frame_);
        if (score > minscore){
            vTempFrames.push_back(frame_);
        }
    }
    
    double maxscore = -99999.0;
    // select max score of map
    {
        for (int index = 0; index < vTempFrames.size(); index++){
            pFrame __frame__ = vTempFrames[index]; // current frame
            int _search_start_ = index - least_continue_frames / 2;
            int _search_end_ = index + least_continue_frames / 2;
            // 
            if (_search_start_ < 0) _search_start_ = 0;
            if (_search_end_ > vTempFrames.size()) _search_end_ = vTempFrames.size(); // compute offset
            if (_search_end_ - _search_start_ < least_continue_frames) continue; // length is not satisfied 
            int start_index = -1;
            bool find_continue_frames = true;
            for (int ii = _search_start_; ii < _search_end_; ii++){
                pFrame frame_ = vTempFrames[ii];
                if (start_index < -1){ start_index = frame_->kf_id_; continue; }
                else {
                    if (abs(start_index - frame_->kf_id_) != 1){
                        find_continue_frames = false;
                        break;
                    }
                }
            }
            if (!find_continue_frames) { continue; } // 共视帧不满足连续三个
            else{
                // get max score
                float __score__ = 0.0;
                for (int ii = _search_start_; ii < _search_end_; ii++){
                    pFrame frame_ = vTempFrames[ii];
                    float score = score_vocab(mpCurrentKF, __frame__);
                    __score__ += score;
                }
                vFrames.push_back(__frame__);
                vCandidateFrameScore.push_back(__score__);
                maxscore = __score__ / least_continue_frames > maxscore?
                __score__ / least_continue_frames: maxscore;
                index = _search_end_;
            }
        }
    }
    
    // refine keyframes
    {
        double refine_score = maxscore * 0.75;
        if (refine_score < 0){
            return vCandidateFrames;
        }
        for (int ii = 0; ii < vFrames.size(); ii++){
            if (vCandidateFrameScore[ii] > refine_score){
                vCandidateFrames.push_back(vFrames[ii]);
            }
        }
    }
    
    return vCandidateFrames;
}

// optimalize 
int LoopClosing::Optimalize(
                    pFrame kf1,
                    pFrame kf2, 
                    std::vector<pMapPoint> & kf1_matches,
                    std::vector<pMapPoint> & kf2_matches,
                    std::vector<pFeature> & kf1_feature,
                    std::vector<pFeature> & kf2_feature,
                    Sophus::SE3d & se3d12,
                    Sophus::SE3d & se3d21,
                    float th
){
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
   
    Sophus::SE3d currentKFPose = kf1->getPose();
    Sophus::SE3d selectKFPose  = kf2->getPose();
    
    std::vector<EdgeProjectionToPM *> vpEdges12;
    std::vector<EdgeProjectionToPMInverse *> vpEdges21;
    
    VertexPose * vertex_pose = new VertexPose(); // camera vertex pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(se3d12);
    optimizer.addVertex(vertex_pose);
    
    Eigen::Matrix3d K = mpCamera->K();
    Sophus::SE3d left_ext = mpCamera->getLeftPose();
    Sophus::SE3d right_ext = mpCamera->getRightPose();
    
    for (int ii = 0; ii < kf1_matches.size(); ii++){
        unsigned int id1 = ii * 2 + 1;
        unsigned int id2 = 2 * (ii * 1);
        if (kf1_matches[ii] == nullptr || kf2_matches[ii] == nullptr){
            continue; // this edge is not accepctable
        }
        else{
            VertexXYZ * mapPoints_KF1 = new VertexXYZ;
            VertexXYZ * mapPoints_KF2 = new VertexXYZ;
            Eigen::Vector3d P3D1w = kf1_matches[ii]->getPosition();
            Eigen::Vector3d P3D2w = kf2_matches[ii]->getPosition();
            Eigen::Vector3d P3D1c = currentKFPose * P3D1w;
            Eigen::Vector3d P3D2c = selectKFPose * P3D2w;
            
            if (P3D2c[2] < 0 || P3D1c[2] < 0){
                LOG(INFO) << "Z position is negative";
                continue;
            }
            
            mapPoints_KF1->setEstimate(P3D1c);
            mapPoints_KF1->setId(id1);
            mapPoints_KF1->setFixed(true);
    
            mapPoints_KF2->setEstimate(P3D2c);
            mapPoints_KF2->setId(id2);
            mapPoints_KF2->setFixed(true);
            
            optimizer.addVertex(mapPoints_KF1);
            optimizer.addVertex(mapPoints_KF2);
        }
        
        
        Eigen::Vector2d obs1, obs2;
        obs1 << kf1_feature[ii]->position_.pt.x,  kf1_feature[ii]->position_.pt.y;
        obs2 << kf2_feature[ii]->position_.pt.x,  kf2_feature[ii]->position_.pt.y;
        
        EdgeProjectionToPM * e12 = new EdgeProjectionToPM(K, left_ext);
        EdgeProjectionToPMInverse * e21 = new EdgeProjectionToPMInverse(K, left_ext);
        
        e12->setVertex(0, optimizer.vertex(0)); // postures all according frames id
        e12->setVertex(1, optimizer.vertex(id2)); // mappoints
        e12->setMeasurement(obs1);
        e12->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(sqrt(th));
        optimizer.addEdge(e12);
        
        e21->setVertex(0, optimizer.vertex(0)); // postures all according frames id
        e21->setVertex(1, optimizer.vertex(id1)); // mappoints
        e21->setMeasurement(obs1);
        e21->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk1->setDelta(sqrt(th));
        optimizer.addEdge(e21);
        
        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
    }
    // Optimize!
    // 3. 开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        EdgeProjectionToPM * e12 = vpEdges12[i];
        EdgeProjectionToPMInverse * e21 = vpEdges21[i];
        if (e12->chi2() > th || e21->chi2() > th){
            optimizer.removeEdge(e21);
            optimizer.removeEdge(e12);
            vpEdges12[i] = nullptr;
            vpEdges21[i] = nullptr;
            continue;
        }
        
        // Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }
    
    // After removing bad edges
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++){
        EdgeProjectionToPM * e12 = vpEdges12[i];
        EdgeProjectionToPMInverse * e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;
        
        e12->computeError();
        e21->computeError();
         
        if (e12->chi2() > th || e21->chi2() > th){
            
        }else{
            nIn++;
        }
    }
    
    // get estimate
    se3d12 = vertex_pose->estimate();
    return nIn;
}

// 
bool LoopClosing::ComputeSim3(){
    if (vCandidateFrames.size() <= 0) {
        LOG(INFO) << "No candidate frames are selected";
        return false;
    }
    
    unsigned int nCandidates = 0;
    std::vector <pFrame> vSelectFrames;
    std::vector <std::vector <pMapPoint>> vCurrentMapPoints;
    std::vector <std::vector <pMapPoint>> vSelectMapPoints;
    std::vector <std::vector <pFeature>>  vCurrentFeatures;
    std::vector <std::vector <pFeature>>  vSelectFeatures;
    // features 
    cv::Mat currentImage = this->mpCurrentKF->imLeft;
    // get keypoints
    std::vector<cv::KeyPoint> kps_cur;
    for (pFeature feat: this->mpCurrentKF->aFeaturesLeft){
        kps_cur.push_back(feat->position_);
    } 
    cv::Mat brief_cur = GetBRIEF(currentImage, kps_cur);
    // Get brief descriptor of current frame
    
    for (int ii = 0; ii < vCandidateFrames.size(); ii++){
        cv::Mat candidateImage = vCandidateFrames[ii]->imLeft;
        std::vector<cv::KeyPoint> kps_can;
        for (pFeature feat: vCandidateFrames[ii]->aFeaturesLeft){
            kps_can.push_back(feat->position_);
        } 
        // get descriptor
        cv::Mat brief_can = GetBRIEF(candidateImage, kps_can);
        // get matches
        std::vector<cv::DMatch> matches = GetMatches(brief_cur, brief_can);
        // filter matches
        std::vector<cv::DMatch> good_matches = FilterMatches(matches, brief_can);
        // estimate matches
        if (good_matches.size() < minimal_accept_matches_){
            continue; // dont need bad match frames
        }else{
            vSelectFrames.push_back(vCandidateFrames[ii]);
            // find match 3d points
            std::vector <pMapPoint> vcmap;
            std::vector <pMapPoint> vsmap;
            std::vector <pFeature> vcf;
            std::vector <pFeature> vsf;
            for (int jj = 0; jj < good_matches.size(); jj++){
                int orginal_id = good_matches[jj].queryIdx;
                int dest_id = good_matches[jj].trainIdx;
                // push back all matched points 
                if (this->mpCurrentKF->aFeaturesLeft[orginal_id]->aMappoint.expired()){
                    // mappoint is null
                    vcf.push_back(this->mpCurrentKF->aFeaturesLeft[orginal_id]);
                    vcmap.push_back(nullptr);
                }else{
                    vcf.push_back(this->mpCurrentKF->aFeaturesLeft[orginal_id]);
                    vcmap.push_back(this->mpCurrentKF->aFeaturesLeft[orginal_id]->aMappoint.lock());
                }
                // push candidate all matched points
                if (vCandidateFrames[ii]->aFeaturesLeft[dest_id]->aMappoint.expired()){
                    // candidate frame's mappoint is null
                    vsf.push_back(vCandidateFrames[ii]->aFeaturesLeft[dest_id]);
                    vsmap.push_back(nullptr);
                }else{
                    vsf.push_back(vCandidateFrames[ii]->aFeaturesLeft[dest_id]);
                    vsmap.push_back(vCandidateFrames[ii]->aFeaturesLeft[dest_id]->aMappoint.lock());
                }
            }
            vCurrentMapPoints.push_back(vcmap);
            vSelectMapPoints.push_back(vsmap);
            vCurrentFeatures.push_back(vcf);
            vSelectFeatures.push_back(vsf);
        }
    }
    
    if (vSelectFrames.size() == 0)
        return false;
    
    // Optimalize and get se3d from these
    for (int ii = 0; ii < vSelectFrames.size(); ii++){
        Sophus::SE3d se3d12;
        Sophus::SE3d se3d21;
        int inliers = Optimalize(mpCurrentKF, vSelectFrames[ii],
            vCurrentMapPoints[ii], vSelectMapPoints[ii],
            vCurrentFeatures[ii], vSelectFeatures[ii],
            se3d12, se3d21
        );
        if (inliers > 10){
            vCorrectSE3d = se3d12;
            vRefinedFrames = vSelectFrames[ii];
            vRefinedCurrentMapPoints = vCurrentMapPoints[ii];
            vRefinedSelectMapPoints = vSelectMapPoints[ii];
            LOG(INFO) << "Iter/Inliers " << ii << "/" << inliers << "Found Search";
            break;
        }
        LOG(INFO) << "Iter/Inliers " << ii << "/" << inliers << " Continue Search";
    }
    
    return true;
}
