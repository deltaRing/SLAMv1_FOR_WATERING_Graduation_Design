#include "dbow.h"

DBoW3::Vocabulary vocab;
bool _vocab_is_load_ = false;
std::string trainData_timeStampPath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_3/timestamp.txt";
std::string trainData_leftImagePath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_3/cam0";
std::string trainData_rightImagePath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_3/cam1";
std::string trainData_depthImagePath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_3/depth";

void feature_training(){
    // read the images and database  
    std::vector<std::string> vstrImageLeft;
    std::vector<std::string> vstrImageRight;
    std::vector<std::string> vstrImageDepth;
    std::vector<double> vTimeStamps;
    extern bool initialized;
    extern Ptr<FeatureDetector> detector;
    if (!initialized){
        initComponents(); // initialize the feature detector
        LOG(INFO) << "Feature extractor is initialized...";
    }
    
    LoadImagesPath(trainData_leftImagePath,
                   trainData_rightImagePath,
                   trainData_depthImagePath,
                   trainData_timeStampPath,
                   vstrImageLeft,
                   vstrImageRight,
                   vstrImageDepth,
                   vTimeStamps);
    
    unsigned int training_num = 100;
    unsigned int data_index = 0;
    unsigned int data_step = (vstrImageLeft.size() - 1) / training_num;
    std::vector<cv::Mat> descriptors;
    for (int ii = 0; ii < training_num; ii++){
        cv::Mat cLeft, cRight, cDepth;
        GetImages(data_index, 
                  vstrImageLeft,
                  vstrImageRight,
                  vstrImageDepth,
                  cLeft, 
                  cRight,
                  cDepth);
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute(cLeft, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
        data_index += data_step;
    }
    LOG(INFO) << "creating vocabulary, please wait...";
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    LOG(INFO) << "vocabulary info: "<< vocab;
    vocab.save("vocabulary.yml.gz");
    LOG(INFO) <<"done";
}

void load_vocab(bool train)
{
    /*if (!access("vocabulary.yml.gz", F_OK)) {
        LOG(INFO) << "Vocabulary does not exist.";
        if (train){
            LOG(INFO) << "Retraining Features.";
            feature_training();
        }
        else{
            _vocab_is_load_ = false;
            return;
        }
    }*/
    vocab = DBoW3::Vocabulary("./vocabulary.yml.gz");
    _vocab_is_load_ = true;
}


void compute_vocab(pFrame kf1){
    if (!_vocab_is_load_){
        LOG(INFO) << "DBoW3 is not loaded" << std::endl;
        return;
    }
    vocab.transform(kf1->descriptors, kf1->bv);
}

double score_vocab(pFrame kf1, pFrame kf2){
    if (!_vocab_is_load_){
        LOG(INFO) << "DBoW3 is not loaded" << std::endl;
        return 0.0;
    }
    return vocab.score(kf1->bv, kf2->bv);
}

double score_vocab(DBoW3::BowVector bv1, DBoW3::BowVector bv2) {
    if (!_vocab_is_load_){
        LOG(INFO) << "DBoW3 is not loaded" << std::endl;
        return 0.0;
    }
    return vocab.score(bv1, bv2);
}
