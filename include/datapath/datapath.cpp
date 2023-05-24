#include "datapath.h"

std::string timeStampPath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_1/timestamp.txt";
std::string leftImagePath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_1/cam0";
std::string rightImagePath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_1/cam1";
std::string depthImagePath = "/home/e1ysium/projects/SLAMv1/datasets/realsense_d430_1/depth";

void LoadImagesPath(const std::string strPathLeft,
                const std::string strPathRight, 
                const std::string strPathDepth, 
                const std::string strPathTimes,
                std::vector<std::string> & vstrImageLeft, 
                std::vector<std::string> & vstrImageRight, 
                std::vector<std::string> & vstrImageDepth,
                std::vector<double> & vTimeStamps)
{
    std::ifstream fTimes;
    std::cout << strPathLeft << std::endl;
    std::cout << strPathRight << std::endl;
    std::cout << strPathTimes << std::endl;
    std::cout << strPathDepth << std::endl;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    vstrImageDepth.reserve(5000);
    while(!fTimes.eof())
    {
        std::string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            vstrImageDepth.push_back(strPathDepth + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);
        }
    }
}

void GetImages(int index,
               std::vector<std::string> vstrImageLeft, 
               std::vector<std::string> vstrImageRight, 
               std::vector<std::string> vstrImageDepth,
               cv::Mat & LeftImage,
               cv::Mat & RightImage,
               cv::Mat & depthImage
              ){
    LeftImage = cv::imread(vstrImageLeft[index], cv::IMREAD_GRAYSCALE);
    RightImage = cv::imread(vstrImageRight[index], cv::IMREAD_GRAYSCALE);
    depthImage = cv::imread(vstrImageDepth[index], cv::IMREAD_ANYDEPTH);

    if(LeftImage.empty() || RightImage.empty() || depthImage.empty()){
        std::cerr << std::endl << "Failed to load image at: " <<  index << std::endl;
        return;
    }
}
