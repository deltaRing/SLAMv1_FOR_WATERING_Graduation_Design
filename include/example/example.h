#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include "../mapping/pointclouds_map.h"
#include "../mapping/posture.h"
#include "../datapath/datapath.h"
#include "../define/flags.h"

extern std::string timeStampPath;
extern std::string leftImagePath;
extern std::string rightImagePath;
extern std::string depthImagePath;

int latest_code(){
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrImageDepth;
    vector<double> vTimeStamps;
    
    std::cout << "SLAMV1 by Laoxiang" << std::endl;
    initComponents();
    initCameraParameter();
    
    LoadImagesPath(leftImagePath,
                   rightImagePath,
                   depthImagePath,
                   timeStampPath, 
                   vstrImageLeft,
                   vstrImageRight,
                   vstrImageDepth,
                   vTimeStamps);
    std::cout << "Loaded Images Left: " << vstrImageLeft.size() << std::endl;
    std::cout << "Loaded Images Right: " << vstrImageRight.size() << std::endl;
    std::cout << "Loaded Images Depth: " << vstrImageDepth.size() << std::endl;
    std::cout << "Loaded Images TimeStamp: " << vTimeStamps.size() << std::endl;
    
    unsigned int numLeft = vstrImageLeft.size();
    unsigned int numRight = vstrImageRight.size();
    unsigned int numDepth = vstrImageDepth.size();
    unsigned int numTimeStamp = vTimeStamps.size();
    
    if (numLeft != numRight || numLeft != numDepth || numLeft != numTimeStamp){
        std::cerr << "Loaded data: DATA number is not correct" << std::endl;
        return _LOAD_DATA_FAILURE_;
    }else{
        
    }
    
    Mat R_cw, t_cw;
    R_cw = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    t_cw = (Mat_<double>(3, 1) << 0, 0, 0);
    Mat img_prev, img_next;
    for (int ii = 1; ii < vstrImageLeft.size(); ii++){
        Mat image_now;
        Mat image_old;
        Mat imLeft_new, imRight_new, imDepth_new;
        Mat imLeft_old, imRight_old, imDepth_old;
        // load data
        GetImages(ii - 1, 
                  vstrImageLeft, vstrImageRight, vstrImageDepth, 
                  imLeft_old, imRight_old, imDepth_old);
        GetImages(ii, 
                  vstrImageLeft, vstrImageRight, vstrImageDepth, 
                  imLeft_new, imRight_new, imDepth_new);
        image_now = imLeft_new.clone();
        image_old = imLeft_old.clone();
        // check image
        if (image_now.empty()){
            std::cerr << "Main Program: Data is Empty" << std::endl;
            return _DATA_IS_EMPTY_DETECT_;
        }
        // get keypoints
        std::vector<KeyPoint> kp_old, kp_new, kp_new_opt;
        kp_old = GetKeypoints(image_now); 
        kp_new = GetKeypoints(image_old);
        // Get filtered points
        std::vector<KeyPoint> f_kp_old, f_kp_new;
        orb_key_point_filter(kp_old, f_kp_old, Point2i(0, 0), 
                             Point2i(image_old.rows, image_old.cols), 0);
        orb_key_point_filter(kp_new, f_kp_new, Point2i(0, 0),
                             Point2i(image_now.rows, image_now.cols), 0);
        // Optical Flow
        int track_num = 0;
        std::vector <uchar> status;
        std::vector <float> error;
        calculate_optFlow(f_kp_old, kp_new_opt,
                          image_old, image_now,
                          status, error, track_num);
        // imshow kps
        for (int jj = 0; jj < status.size(); jj++){
            // f_kp[jj].response KeyPoint's response
            if (status[jj] && error[jj] <= 2.5){
                circle(image_now, f_kp_new[jj].pt, 1, Scalar(0, 255, 0), -1);
                line(image_now, f_kp_old[jj].pt, kp_new_opt[jj].pt, Scalar(0, 255, 0));
            }
        }
        imshow("optical features", image_now);
        waitKey(10);
        
        Mat R, t;
        bool succ = pose_estimation_2d2d(f_kp_old, kp_new_opt, status, error, R, t);
        if (succ){
            R_cw = R_cw * R;
            t_cw = t_cw + t;
        }
    }
}

int using_single_eye(){
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrImageDepth;
    vector<double> vTimeStamps;
    
    std::cout << "SLAMV1 by Laoxiang" << std::endl;
    initComponents();
    initCameraParameter();
    
    LoadImagesPath(leftImagePath,
                   rightImagePath,
                   depthImagePath,
                   timeStampPath, 
                   vstrImageLeft,
                   vstrImageRight,
                   vstrImageDepth,
                   vTimeStamps);
    std::cout << "Loaded Images Left: " << vstrImageLeft.size() << std::endl;
    std::cout << "Loaded Images Right: " << vstrImageRight.size() << std::endl;
    std::cout << "Loaded Images Depth: " << vstrImageDepth.size() << std::endl;
    std::cout << "Loaded Images TimeStamp: " << vTimeStamps.size() << std::endl;
    
    unsigned int numLeft = vstrImageLeft.size();
    unsigned int numRight = vstrImageRight.size();
    unsigned int numDepth = vstrImageDepth.size();
    unsigned int numTimeStamp = vTimeStamps.size();
    
    if (numLeft != numRight || numLeft != numDepth || numLeft != numTimeStamp){
        std::cerr << "Loaded data: DATA number is not correct" << std::endl;
        return _LOAD_DATA_FAILURE_;
    }else{
        
    }
    
    // Related position
    Mat R_cw = (Mat_<double>(3, 3) << 1, 0, 0,
                                      0, 1, 0, 
                                      0, 0, 1);
    Mat t_cw = (Mat_<double>(3, 1) << 0, 0, 0);
    for (int ii = 1; ii < vstrImageLeft.size(); ii++){
        cout << ii << endl;
        Mat image_now;
        Mat image_old;
        Mat imLeft_new, imRight_new, imDepth_new;
        Mat imLeft_old, imRight_old, imDepth_old;
        // load data
        GetImages(ii - 1, 
                  vstrImageLeft, vstrImageRight, vstrImageDepth, 
                  imLeft_old, imRight_old, imDepth_old);
        GetImages(ii, 
                  vstrImageLeft, vstrImageRight, vstrImageDepth, 
                  imLeft_new, imRight_new, imDepth_new);
        image_now = imLeft_new.clone();
        image_old = imLeft_old.clone();
        // check image
        if (image_now.empty()){
            std::cerr << "Main Program: Data is Empty" << std::endl;
            return _DATA_IS_EMPTY_DETECT_;
        }
        // get keypoints
        std::vector<KeyPoint> kpOld, kpNew;
        Mat briefOld, briefNew;
        std::vector<DMatch> matches;
        std::vector<DMatch> good_matches;
        // get keypoints
        kpOld = GetKeypoints(image_now); 
        kpNew = GetKeypoints(image_old);
        // get descriptor
        briefOld = GetBRIEF(image_now, kpOld);
        briefNew = GetBRIEF(image_old, kpNew);
        // get matches
        matches = GetMatches(briefOld, briefNew);
        // filter matches
        good_matches = FilterMatches(matches, briefNew);
        // estimate movement of cameras
        Mat R, t;
        // get new position
        bool succ = pose_estimation_2d2d(kpOld, kpNew, good_matches, R, t);
        if (!succ) {
            std::cout << "Posture estimate failure" << std::endl;
            continue;
        }
        // draw matches
        Mat img_goodmatch;
        drawMatches(image_old, kpOld, image_now, kpNew, 
                    good_matches, img_goodmatch);
        imshow("matches", img_goodmatch);
        waitKey(10);
        R_cw = R * R_cw;
        t_cw += t;
        std::cout << "Rotation: " << R_cw << std::endl;
        std::cout << "Transition: " << t_cw << std::endl;
    }
    return 0;
}

int filter_orb_features(){
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrImageDepth;
    vector<double> vTimeStamps;
    
    std::cout << "SLAMV1 by Laoxiang" << std::endl;
    initComponents();
    initCameraParameter();
    
    LoadImagesPath(leftImagePath,
                   rightImagePath,
                   depthImagePath,
                   timeStampPath, 
                   vstrImageLeft,
                   vstrImageRight,
                   vstrImageDepth,
                   vTimeStamps);
    std::cout << "Loaded Images Left: " << vstrImageLeft.size() << std::endl;
    std::cout << "Loaded Images Right: " << vstrImageRight.size() << std::endl;
    std::cout << "Loaded Images Depth: " << vstrImageDepth.size() << std::endl;
    std::cout << "Loaded Images TimeStamp: " << vTimeStamps.size() << std::endl;
    
    unsigned int numLeft = vstrImageLeft.size();
    unsigned int numRight = vstrImageRight.size();
    unsigned int numDepth = vstrImageDepth.size();
    unsigned int numTimeStamp = vTimeStamps.size();
    
    if (numLeft != numRight || numLeft != numDepth || numLeft != numTimeStamp){
        std::cerr << "Loaded data: DATA number is not correct" << std::endl;
        return _LOAD_DATA_FAILURE_;
    }else{
        
    }
    
    for (int ii = 0; ii < vstrImageLeft.size(); ii++){
        Mat imLeft, imRight, imDepth;
        // load data
        GetImages(ii, 
                  vstrImageLeft, vstrImageRight, vstrImageDepth, 
                  imLeft, imRight, imDepth);
        // check image
        if (imLeft.empty()){
            std::cerr << "Main Program: Data is Empty" << std::endl;
            return _DATA_IS_EMPTY_DETECT_;
        }
        // get keypoints
        std::vector<KeyPoint> kp;
        Mat brief;
        std::vector<DMatch> matches;
        std::vector<DMatch> good_matches;
        // get keypoints
        kp = GetKeypoints(imLeft); 
        // get descriptor
        brief = GetBRIEF(imLeft, kp);
        // Get filtered points
        std::vector<KeyPoint> f_kp;
        orb_key_point_filter(kp, f_kp, Point2i(0, 0), Point2i(imLeft.rows, imLeft.cols), 0);
        // imshow kps
        for (int jj = 0; jj < f_kp.size(); jj++){
            // f_kp[jj].response KeyPoint's response
            circle(imLeft, f_kp[jj].pt, 1, Scalar(0, 255, 0), -1);
        }
        imshow("features", imLeft);
        std::cout << f_kp.size() << std::endl;
        waitKey(10);
    }
    
    return 0;
}

#endif
