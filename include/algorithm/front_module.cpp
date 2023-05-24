#include "front_module.h"

extern bool use_orb;

bool front_module::GetNextFrame(pFrame & newFrame){
    if (vstrImageLeft.size() <= current_frame_index) {
        LOG(INFO) << "Data is already finished";
        return false;
    }else{
        GetImages(current_frame_index, 
            vstrImageLeft, vstrImageRight, vstrImageDepth, 
            cImLeft, cImRight, cImDepth);
        newFrame = frame::createFrame();
        newFrame->imLeft = cImLeft.clone();
        newFrame->imRight = cImRight.clone();
        newFrame->imDepth = cImDepth.clone();
        current_frame_index++;
    }
    
    return true;
}

int front_module::EstimateCurrentPose(){
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    // optimalize the posture vertex
    VertexPose * vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->getPose()); // optimalize this poseture
    optimizer.addVertex(vertex_pose);
    
    // K
    Eigen::Matrix3d K = camera_->K();
    // egdes
    std::vector <EdgeProjection *> edges;
    std::vector <pFeature> features;
    int index = 1;
    for (int ii = 0; ii < current_frame_->aFeaturesLeft.size(); ii++){
        auto mp = current_frame_->aFeaturesLeft[ii]->aMappoint.lock();
        if (mp){
            features.push_back(current_frame_->aFeaturesLeft[ii]);
            EdgeProjection * edge = new EdgeProjection(mp->location, K);
            edge->setId(index++);
            edge->setVertex(0, vertex_pose);
            Eigen::Vector2d measurement = Eigen::Vector2d(
                current_frame_->aFeaturesLeft[ii]->position_.pt.x, 
                current_frame_->aFeaturesLeft[ii]->position_.pt.y
            );
            edge->setMeasurement(measurement);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
        }
    }
    
    // determine the outliers
    const double chi2_th = 5.991;
    int cnt_outliers = 0; // compute the outlier points
    for (int ii = 0; ii < 4; ii++){
        vertex_pose->setEstimate(current_frame_->getPose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outliers = 0;
        for (int i = 0; i < edges.size(); i++){
            auto e = edges[i];
            if (features[i]->is_outlier){
                e->computeError();
            }
            if (e->chi2() > chi2_th){
                features[i]->is_outlier = true;
                e->setLevel(1);
                cnt_outliers++;
            }else{
                features[i]->is_outlier = false;
                e->setLevel(0);
            }
            if (ii == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }
    
    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outliers << "/" << features.size() - cnt_outliers;
    
    current_frame_->setPose(vertex_pose->estimate());
    
    LOG(INFO) << "Current Pose = \n" << current_frame_->getPose().matrix();
    
    for (auto & feat:features){
        if (feat->is_outlier){
            feat->aMappoint.reset(); // clear this mappoint
            feat->is_outlier = false; // maybe this feature are useful later
        }
    }
    return features.size() - cnt_outliers;
}

// Add observe for mappoints when frame is keyframe
void front_module::SetObservationsForKeyFrame() {
    for (auto & feat : current_frame_->aFeaturesLeft){
        auto mp = feat->aMappoint.lock();
        if (mp){
            mp->add_observe(feat);
        }
    }
}

int front_module::DetectFeatures(){
    std::vector<KeyPoint> kpLeft, fkpLeft;
    kpLeft = GetKeypointsMultiScale(cImLeft);
    // kpLeft = GetKeypoints(cImLeft); 
    // detecting new features
    if (use_orb){
        orb_key_point_filter(kpLeft, fkpLeft, Point2i(0, 0), 
                        Point2i(cImLeft.rows, cImLeft.cols), 0);
    }else{
        fkpLeft = kpLeft;
    }
    
    for (auto &kp : fkpLeft) {
        bool _is_feature_same_ = false;
        for (auto & ckp: current_frame_->aFeaturesLeft){
            if (int(ckp->position_.pt.x) == int(kp.pt.x) && int(ckp->position_.pt.y) == int(kp.pt.y)){
                _is_feature_same_ = true;
                break;
            }
        }
        if (_is_feature_same_)
            continue;
        current_frame_->aFeaturesLeft.push_back(
            pFeature(new feature(current_frame_, kp)));
    }
    
    LOG(INFO) << "Detect " << fkpLeft.size() << " new features";
    
    return kpLeft.size();
}

bool front_module::InsertKeyFrame(){
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    
    // insert current frame as keyframe
    current_frame_->setKeyFrame();
    GetDescriptor(current_frame_); // get descriptors for keyframes
    compute_vocab(current_frame_); // compute the vocabs for keyframes
    
    map_->InsertKeyFrame(current_frame_);
    
    LOG(INFO) << "Frame " << current_frame_->id_ << " is keyframe " << current_frame_->kf_id_ << " now.";
    
    SetObservationsForKeyFrame(); // add map point's observation
    DetectFeatures(); // now if features of int are filtered to ensure no extra features are extracted
    
    int getStereoFeatures = FindFeaturesInRight();
    LOG(INFO) << "Get Stereo Features: " << getStereoFeatures;
    // get new mappoints
    TriangulateNewMappoints(getStereoFeatures);
    // back module update --> to all mappoints
    back_module_->UpdateMap();
    //
    GetCoherentWeights(); 
    
    if (use_viewer_){
        // TODO
        viewer_->UpdateMap();
    }
    
    if (use_dense_map_){
        point_cloud_->insert_frame(current_frame_->imLeft,
                                   current_frame_->imDepth,
                                   current_frame_->getPose().inverse());
    }
    
    if (loop_closure_ != nullptr){
        loop_closure_->SetKeyFrame(current_frame_);
    }
    
    return true;
}

// triangulate and current pose to get new map points
bool front_module::TriangulateNewMappoints(int num_detected_points){
    // build initial map according to features
    std::vector<Sophus::SE3d> poses_{
        camera_->getLeftPose(), camera_->getRightPose()
    }; 
    Sophus::SE3d current_frame_Twc = current_frame_->getPose().inverse();
    int cnt_triangulated_pts = 0;
    
    for (int ii = 0; ii < num_detected_points; ii++){
        if (current_frame_->aFeaturesLeft[ii]->aMappoint.expired() &&
            current_frame_->aFeaturesRight[ii] != nullptr){ // if frame is new frame and aMappoint is null
            // create map point from triangulation
            std::vector<Eigen::Vector3d> points{
                camera_->pixel2camera(
                    Eigen::Vector2d(
                    current_frame_->aFeaturesLeft[ii]->position_.pt.x,
                    current_frame_->aFeaturesLeft[ii]->position_.pt.y
                )),
                camera_->pixel2camera(
                    Eigen::Vector2d(
                    current_frame_->aFeaturesRight[ii]->position_.pt.x,
                    current_frame_->aFeaturesRight[ii]->position_.pt.y
                ))
            };
            
            Eigen::Vector3d pworld; // points location of world
            if (triangulate(poses_, points, pworld) && pworld[2] > 0){
                auto new_map_point = mappoint::create_new_mappoint();
                pworld = current_frame_Twc * pworld; // build new map points
                new_map_point->setPosition(pworld);
                new_map_point->add_observe(current_frame_->aFeaturesLeft[ii]);
                new_map_point->add_observe(current_frame_->aFeaturesRight[ii]);
                current_frame_->aFeaturesLeft[ii]->aMappoint = new_map_point;
                current_frame_->aFeaturesRight[ii]->aMappoint = new_map_point;
                cnt_triangulated_pts++;
                map_->InsertMapPoint(new_map_point);
            }
        }
    }
    return true;
}

int front_module::TrackLastFrame(){
    std::vector<cv::Point2f> kp_last, kp_current;
    // use Lk flow
    for (auto &kp : last_frame_->aFeaturesLeft){
        if (kp->aMappoint.lock()){
            // project point
            auto mp = kp->aMappoint.lock();
            auto px = camera_->world2pixel(mp->location, current_frame_->getPose());
            kp_last.push_back(kp->position_.pt);
            kp_current.push_back(cv::Point2f(px[0], px[1]));
        }else{
            // is this useful?
            kp_last.push_back(kp->position_.pt);
            kp_current.push_back(kp->position_.pt);
        }
    }
    // points are filtered 
    std::vector <uchar> status;
    std::vector <float> error;
    
    cv::calcOpticalFlowPyrLK(last_frame_->imLeft, 
                             current_frame_->imLeft,
                             kp_last,
                             kp_current, 
                             status, error, cv::Size(11, 11), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                              30,
                                              0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kp_current[i], 7.0);
            pFeature feature_(new feature(current_frame_, kp));
            feature_->aMappoint = last_frame_->aFeaturesLeft[i]->aMappoint;
            current_frame_->aFeaturesLeft.push_back(feature_);
            num_good_pts++;
        }
    }
    
    LOG(INFO) << "Find last points " << kp_last.size() << " in the last image.";
    LOG(INFO) << "Find good points " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool front_module::StereoInit(){
    int num_features = DetectFeatures();
    int num_features_right = FindFeaturesInRight();
    
    if (num_features < num_features_init_){
        LOG(INFO) << "NOT enough features " << num_features_right << "/" << num_features_init_;
        return false;
    } 
    
    bool success = BuildInitMap(num_features_right);
    if (success) {
        status_ = _FRONT_STATUS_TRACK_GOOD_;
        if (use_viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        LOG(INFO) << "Initial map is built!";
        return true;
    }
    return false;
}

bool front_module::BuildInitMap(int num_features){
    // build initial map according to features
    std::vector<Sophus::SE3d> poses_{
        camera_->getLeftPose(), camera_->getRightPose()
    };
    int init_landmarks = 0;
    for (int ii = 0; ii < num_features; ii++){
        // create map point from triangulation
        if (current_frame_->aFeaturesRight[ii] == nullptr) continue;
        std::vector<Eigen::Vector3d> points{
            camera_->pixel2camera(
                Eigen::Vector2d(
                current_frame_->aFeaturesLeft[ii]->position_.pt.x,
                current_frame_->aFeaturesLeft[ii]->position_.pt.y
            )),
            camera_->pixel2camera(
                Eigen::Vector2d(
                current_frame_->aFeaturesRight[ii]->position_.pt.x,
                current_frame_->aFeaturesRight[ii]->position_.pt.y
            ))
        };
        
        Eigen::Vector3d pworld; // points location of world
        if (triangulate(poses_, points, pworld) && pworld[2] > 0){
            auto new_map_point = mappoint::create_new_mappoint();
            new_map_point->setPosition(pworld);
            new_map_point->add_observe(current_frame_->aFeaturesLeft[ii]);
            new_map_point->add_observe(current_frame_->aFeaturesRight[ii]);
            current_frame_->aFeaturesLeft[ii]->aMappoint = new_map_point;
            current_frame_->aFeaturesRight[ii]->aMappoint = new_map_point;
            init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    // set key frame
    current_frame_->setKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    // back module update
    back_module_->UpdateMap();
    // -------------=--------------
    
    LOG(INFO) << "Initial map with mappoints of " << init_landmarks << std::endl;
    return true;
}

// using linear triangulate method to get points in world
bool front_module::triangulate(std::vector<Sophus::SE3d> & poses, 
        const std::vector<Eigen::Vector3d> points,
        Eigen::Vector3d & pt_world
    ){
    Eigen::MatrixXd A(2 * poses.size(), 4);
    Eigen::VectorXd b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();
    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;
    }
    return false;
}

// Detect features of left images
// and then find features from right images
int front_module::FindFeaturesInRight(){
    std::vector<cv::Point2f> kpLeft, kpRight, 
                        fkpLeft, fkpRight;
                        
    for (auto &kp : current_frame_->aFeaturesLeft) {
        kpLeft.push_back(kp->position_.pt);
        auto mp = kp->aMappoint.lock();
        if (mp) {
            // use projected points as initial guess
            auto px = camera_->world2pixel(mp->location, current_frame_->getPose());
            kpRight.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            kpRight.push_back(kp->position_.pt);
        }
    }                    
    
    // Optical Flow
    std::vector <uchar> status;
    std::vector <float> error;
    
    cv::calcOpticalFlowPyrLK(cImLeft,
                             cImRight,
                             kpLeft,
                             kpRight, 
                             status,
                             error, 
                             cv::Size(11, 11), 
                             3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                             30,
                                             0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    
    int cnt_detected = 0;
    int index = 0;
    for (auto &kp : kpLeft) {
        if (status[index]){
            cv::KeyPoint kpr(kpRight[index], 7.0);
            pFeature tfeat = pFeature(new feature(current_frame_, kpr));
            tfeat->is_left_image = false;
            current_frame_->aFeaturesRight.push_back(
                tfeat
            );
            cnt_detected++;
        }else{
            current_frame_->aFeaturesRight.push_back(
                nullptr
            );
        }
        index++;
    }
    // detecting new features
    LOG(INFO) << "Right Features Detect " << cnt_detected << " new features";
    return cnt_detected;
}

// assume this is motion 
bool front_module::Track(){
    if (last_frame_ != nullptr){
        // last frame is not null
        Sophus::SE3d new_pose = relative_motion_ * last_frame_->getPose(); 
        current_frame_->setPose(new_pose);
    }
    
    int num_track_last = TrackLastFrame(); // get last frames
    tracking_inliers_ = EstimateCurrentPose(); // get now postures and out_liers
    
    if (tracking_inliers_ > num_features_tracking_)
        status_ = _FRONT_STATUS_TRACK_GOOD_;
    else if (tracking_inliers_ > num_features_tracking_bad_)
        status_ = _FRONT_STATUS_TRACK_BAD_;
    else
        status_ = _FRONT_STATUS_TRACK_LOST_;
    
    InsertKeyFrame(); // check if need to get new keyframe
    relative_motion_ = current_frame_->getPose() * last_frame_->getPose().inverse();
    
    if (viewer_)
        viewer_->AddCurrentFrame(current_frame_);
    return true;
}

// get coherent weights of maps
void front_module::GetCoherentWeights(){
    std::vector<wPmap> pMps;
    std::vector<wPframe> wpFrame;
    std::vector<int> wpFrameCounter;
    
    for (auto & feats : current_frame_->aFeaturesLeft){
        pMps.push_back(feats->aMappoint);
    }
    
    // find these observed keyframes
    for (auto & _mappoint_ : pMps){
        if (_mappoint_.expired()) continue;
        std::list<std::weak_ptr<feature>> aObs = _mappoint_.lock()->get_observe();
        auto obs = aObs.begin();
        while (obs != aObs.end()){
            std::weak_ptr<feature> tObs = *obs;
            if (tObs.expired()){
                obs++;
                continue;
            }
            if (tObs.lock()->aFrame.lock()->kf_id_ == current_frame_->kf_id_){
                obs++;
                continue;
            }
            
            if (wpFrame.size() > 0){
                bool find_ = false;
                for (int ii = 0; ii < wpFrame.size(); ii++){
                    if (wpFrame[ii].lock() == tObs.lock()->aFrame.lock()){
                        wpFrameCounter[ii]++;
                        find_ = true;
                        break;
                    }
                }
                if (!find_){
                    wpFrame.push_back(tObs.lock()->aFrame.lock());
                    wpFrameCounter.push_back(1);
                }
            }else{
                wpFrame.push_back(tObs.lock()->aFrame.lock());
                wpFrameCounter.push_back(1);  
            }
            obs++;
        }
    }
    
    // steps 2
    int th = current_frame_->coherent_keyframe_num;
    for (int ii = 0; ii < wpFrame.size(); ii++) {
        if (wpFrameCounter[ii] >= th) {
            current_frame_->addCoherentFrame(make_pair(wpFrameCounter[ii], wpFrame[ii].lock()));
            wpFrame[ii].lock()->addCoherentFrame(make_pair(wpFrameCounter[ii], current_frame_)); 
            // 对超过阈值的共视边建立连接
        }
    }
}
