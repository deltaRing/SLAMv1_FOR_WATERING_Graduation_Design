#include "posture.h"

bool initialized = false;
cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorExtractor> descriptor;
cv::Ptr<cv::DescriptorMatcher> matcher;
cv::Ptr<cv::GFTTDetector> gftt_;  // feature detector in opencv
bool initialize_intrinsics = false;
bool use_orb = true;
cv::Mat initCameraLeft;
cv::Mat initCameraRight;
cv::Mat K;

void GetDescriptor(pFrame kf){
    if (!initialized){
        LOG(INFO) << "Initialization: Not Initialized";
        return;
    }
    
    if (kf == nullptr){
        LOG(INFO) << "KeyFrame Pointer is NULL";
        return;
    }
    std::vector <cv::KeyPoint> kps;
    
    detector->detectAndCompute(kf->imLeft, cv::Mat(), kps, kf->descriptors);
    return;
}

cv::Point2f pixel2cam(const cv::Point2d &p) {
  return cv::Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void initComponents(){
    if (initialized){
        LOG(INFO) << "Initialization: Already Initialized";
        return;
    }
    gftt_ =
        cv::GFTTDetector::create(_MAXIMUM_POINT_, 0.01, 20);
    detector = cv::ORB::create(_MAXIMUM_POINT_ORB_, _scaleFactor_, _nlevels_, 
                           32, 0, 4, cv::ORB::HARRIS_SCORE, 32, 10);
    descriptor = cv::ORB::create(_MAXIMUM_POINT_ORB_, _scaleFactor_, _nlevels_,
                             32, 0, 4, cv::ORB::HARRIS_SCORE, 32, 10);
    matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    initialized = true;
}

// get ORB keypoints from multiple-zoom
std::vector<cv::KeyPoint> GetKeypointsMultiScale(cv::Mat Images, int scalefactor){
    std::vector <cv::KeyPoint> all_kps;
    
    if (!initialized) {
        LOG(INFO) << "Get multipleScale-FAST Points: Need Initialized";
        return all_kps;
    }
    
    for (int ii = 0; ii < scalefactor; ii++){
        if (int(Images.cols / pow(2, ii + 1)) < 16 || int(Images.rows / pow(2, ii + 1)) < 16){
            return all_kps; // image is too small not confident
        }
        
        std::vector <cv::KeyPoint> kps;
        cv::Mat resized_image;
        cv::resize(Images, resized_image, cv::Size(int(Images.cols / pow(2, ii + 1)), int(Images.rows / pow(2, ii + 1))));
        detector->detect(resized_image, kps);
        for (int iii = 0; iii < kps.size(); iii++){
            cv::KeyPoint temp_kp;
            temp_kp.pt = cv::Point2i(int(kps[iii].pt.x * pow(2, ii + 1)), int(kps[iii].pt.y * pow(2, ii + 1)));
            temp_kp.response = kps[iii].response;
            bool unique = true;
            for (int iiii = 0; iiii < all_kps.size(); iiii++){
                if (int(all_kps[iiii].pt.x) == int(temp_kp.pt.x) && 
                    int(all_kps[iiii].pt.y) == int(temp_kp.pt.y)
                ){
                    unique = false;
                    break;
                }
            }
            if (unique){
                all_kps.push_back(temp_kp);
            }
        }
    }
    
    return all_kps;
}

// compute oriented FAST points
std::vector<cv::KeyPoint> GetKeypoints(cv::Mat images, bool _USE_ORB_){
    std::vector<cv::KeyPoint> points;
    if (!initialized) {
        LOG(INFO) << "Get FAST Points: Need Initialized";
        return points;
    }
    if (use_orb || _USE_ORB_)
        detector->detect(images, points);
    else
        gftt_->detect(images, points);
    return points;
}

// compute Brief
cv::Mat GetBRIEF(cv::Mat images, std::vector<cv::KeyPoint> keypoints){
    cv::Mat descriptors;
    if (!initialized) {
        LOG(INFO) << "Get BRIEF descriptor: Need Initialized";
        return descriptors;
    }
    descriptor->compute(images, keypoints, descriptors);
    return descriptors;
}

// compute matches
std::vector<cv::DMatch> GetMatches(cv::Mat descriptor1, 
                                   cv::Mat descriptor2){
    std::vector<cv::DMatch> matches;
    if (!initialized) {
        LOG(INFO) << "Get Matches: Need Initialized";
        return matches;
    }
    matcher->match(descriptor1, descriptor2, matches);
    return matches;
}

// filter matches
std::vector<cv::DMatch> FilterMatches(std::vector<cv::DMatch> matches, cv::Mat _descriptor_){
    std::vector<cv::DMatch> good_matches;
    if (matches.size() == 0) {
        LOG(INFO) << "Filter Matches: No match points";
        return good_matches;
    }
    // compute min max distances
    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < _descriptor_.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    
    for (int ii = 0; ii < _descriptor_.rows; ii++){
        if (matches[ii].distance <= std::max(2.0 * min_dist, 30.0)){
            good_matches.push_back(matches[ii]);
        }
    }
    
    return good_matches;
}

// init camera models D430
void initCameraParameter(){
    if (initialize_intrinsics) {
        LOG(INFO) << "Initialization d430: Already Initialized";
        return;
    }
    K = (cv::Mat_<double>(3, 3) << 382.613, 0, 320.183,
         0, 382.613, 236.455,
         0, 0, 1);
    initialize_intrinsics = true;
}

// compute posture 2d-2d images
// return true: successfully 
// return false: R t is not computed correctly
bool pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<cv::DMatch> matches,
                          cv::Mat &R, cv::Mat &t) {
    if(!initialize_intrinsics) {
        LOG(INFO) << "2d-2d Posture: Need Initialization";
        return false;
    }
    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    cv::Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    //-- 计算本质矩阵
    cv::Point2d principal_point(320.183, 236.455);  //相机光心
    double focal_length = 382.613;      //相机焦距
    cv::Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

    cv::Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, cv::RANSAC, 3);
    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    
    return true;
} // not useful by directly compute


// Get transition by using LK optical flow
bool pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<uchar> status,
                          std::vector<float> error,
                          cv::Mat &R, cv::Mat &t){
    if(!initialize_intrinsics) {
        LOG(INFO) << "2d-2d Posture: Need Initialization";
        return false;
    }
    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < status.size(); i++) {
        if (status[i] && error[i] <= 2.5){
            points1.push_back(keypoints_1[i].pt);
            points2.push_back(keypoints_2[i].pt);
            //std::cout << keypoints_1[i].pt << " " << keypoints_2[i].pt << std::endl;
        }
    }
    if (points1.size() <= 30){
        LOG(INFO) << "Too less matched points";
        return false;
    }
    //-- 计算本质矩阵
    cv::Point2d principal_point(320.183, 236.455);  //相机光心
    double focal_length = 382.613;      //相机焦距
    cv::Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    
    return true;
}

// orb KeyPoints Filter
// remove keypoints of low response
void orb_key_point_filter(std::vector<cv::KeyPoint> kps, 
                          std::vector<cv::KeyPoint> & best_kp,
                          cv::Point2i node_start,
                          cv::Point2i cell_size,
                          int current_level,
                          int max_level
                         ){
    // Keep lower response ORB_POINTS removed
    if (current_level > max_level){
        return;
    }
    if (node_start.x < 0 || node_start.y < 0){
        return;
    }
    if (cell_size.x <= 0 || cell_size.y <= 0){
        return;
    }
    // check node 
    if (kps.size() < 1){
        return; // disable this sector
    }
    if (current_level == max_level){
        // select maximum response of orb points
        double max_reponse = -1.0;
        int index = -1;
        for (int ii = 0; ii < kps.size(); ii++){
            if (max_reponse < kps[ii].response){
                max_reponse = kps[ii].response;
                index = ii;
            }
        }
        best_kp.push_back(kps[index]);
        return;
    }
    
    int point_start_x[] = {0, 1, 0, 1};
    int point_start_y[] = {0, 0, 1, 1};
    
    for (int ii = 0; ii < 4; ii++){
        // Get range of cells and start location
        int xxCell = cell_size.x / 2, 
            yyCell = cell_size.y / 2;
        int xxStart = node_start.x + cell_size.x / 2 * point_start_x[ii],
            yyStart = node_start.y + cell_size.y / 2 * point_start_y[ii];
        int xxEnd = xxStart + xxCell,
            yyEnd = yyStart + yyCell;
        std::vector<cv::KeyPoint> t_kps;
        for (int jj = 0; jj < kps.size(); jj++){
            if (kps[jj].pt.x >= xxStart && kps[jj].pt.y >= yyStart &&
                kps[jj].pt.x <= xxEnd && kps[jj].pt.y <= yyEnd){
                t_kps.push_back(kps[jj]);
            }
        }
        cv::Point2i StartLoc(xxStart, 
                         yyStart);
        cv::Point2i Cells(xxCell, 
                      yyCell);
        orb_key_point_filter(t_kps,
                             best_kp,
                             StartLoc,
                             Cells,
                             current_level + 1,
                             max_level
                            );
    }
}

// get optical flow and track new points
void calculate_optFlow(
    std::vector<cv::KeyPoint> kp_old, 
    std::vector<cv::KeyPoint> & kp_new,
    cv::Mat img_old,
    cv::Mat img_new,
    std::vector<uchar> & status,
    std::vector<float> & err,
    int & track_num
){
    std::vector <cv::Point2f> pt_old, pt_new;
    for (int ii = 0; ii < kp_old.size(); ii++){
        pt_old.push_back(kp_old[ii].pt);
    }
    cv::calcOpticalFlowPyrLK(img_old, img_new,
                             pt_old, pt_new, 
                             status, err);
    for (int ii = 0; ii < status.size(); ii++){
        cv::KeyPoint kp(pt_new[ii], 7.0f);
        kp_new.push_back(kp);
        track_num += status[ii];
    }
}

// get mappoint's depth
// input: kp_old_frame kp_new_frame
// input: matches position_old position_new
// output: points_3d
void triangulation(
    std::vector<cv::KeyPoint> &keypoint_1,
    std::vector<cv::KeyPoint> &keypoint_2,
    std::vector<cv::DMatch> &matches,
    cv::Mat &R1, const cv::Mat &t1,
    cv::Mat &R2, const cv::Mat &t2,
    std::vector<cv::Point3d> & points) {
    cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
        R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0, 0),
        R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1, 0),
        R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2, 0)
    );
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
        R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0, 0),
        R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1, 0),
        R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2, 0)
    );

    std::vector<cv::Point2f> pts_1, pts_2;
    for (cv::DMatch m:matches) {
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        cv::Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0)
        );
        points.push_back(p);
    }
}

// original 3d position-> new 2d pixel
// optimize posture
void bundleAdjustmentG2O(
    const std::vector<Eigen::Vector3d> &points_3d,
    const std::vector<Eigen::Vector2d> &points_2d,
    Sophus::SE3d &pose) {
    if (!initialize_intrinsics){
        std::cerr << "BA G2O PoseTure 3d->2d: Need Initilize Camera Parameter First" << std::endl;
        return;
    }
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // vertex
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection edge(p3d, K_eigen);
        edge.setId(index);
        edge.setVertex(0, vertex_pose);
        edge.setMeasurement(p2d);
        edge.setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(&edge);
        index++;
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    pose = vertex_pose->estimate();
    
    delete [] vertex_pose;
}
