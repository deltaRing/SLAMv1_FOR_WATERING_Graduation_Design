#ifndef _DATAPATH_H_
#define _DATAPATH_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
// load images
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
// standard library

// Get images
void LoadImagesPath(const std::string strPathLeft,
                const std::string strPathRight, 
                const std::string strPathDepth, 
                const std::string strPathTimes,
                std::vector<std::string> & vstrImageLeft, 
                std::vector<std::string> & vstrImageRight, 
                std::vector<std::string> & vstrImageDepth,
                std::vector<double> & vTimeStamps);

// Load Images to Mat
void GetImages(int index,
               std::vector<std::string> vstrImageLeft, 
               std::vector<std::string> vstrImageRight, 
               std::vector<std::string> vstrImageDepth,
               cv::Mat & LeftImage,
               cv::Mat & RightImage,
               cv::Mat & depthImage
              );

#endif
