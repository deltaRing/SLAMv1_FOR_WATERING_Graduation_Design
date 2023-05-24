#include "pointclouds_map.h"

// 获取三维点云
void PointCloudMap::get_point_clouds_(
                                    cv::Mat image,
                                    cv::Mat depth, 
                                    int step) {
    pointClouds.clear();
    // clear original pointclouds
    for (int v = 0; v < depth.rows; v+=step){
        for (int u = 0; u < depth.cols; u+=step) {
            if (depth.at<ushort>(v, u) <= 0 || depth.at<ushort>(v, u) >= 128 * 256) continue; // 太远的太近的不靠谱
            
             Vector4d point_(0, 0, 0, image.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

			 unsigned short disp = depth.at<ushort>(v, u);
			 double z = double(disp) / deep_scale;//计算中是以毫米为单位的；
			 
			 point_[2] = z ; 
             point_[0] = double(u - cx) * point_[2] / fx;  
			 point_[1] = double(v - cy) * point_[2] / fy;

			 pointClouds.push_back(point_); 
            // end your code here
        }
    }
}

// get filtered point clouds
void PointCloudMap::filter_point_clouds_(
                                    cv::Mat image,
                                    cv::Mat depth, 
                                    int step){
    cPointCloudI::Ptr current = cPointCloudI::Ptr(new cPointCloudI);
    
    for (int v = 0; v < depth.rows; v+=step){
        for (int u = 0; u < depth.cols; u+=step) {
            if (depth.at<ushort>(v, u) <= 0 || depth.at<ushort>(v, u) >= 64 * 256) 
                continue; // 太远的太近的不靠谱
            unsigned short disp = depth.at<ushort>(v, u);
            double z = double(disp) / deep_scale;//计算中是以毫米为单位的；
			 
            Vector3d point_;
            point_[2] = z; 
            point_[0] = double(u - cx) * point_[2] / fx;  
            point_[1] = double(v - cy) * point_[2] / fy;
            
            Vector3d point_world = poses_ * point_;
            pXYZI p;
            p.x = point_world[0];
            p.y = point_world[1];
            p.z = point_world[2];
            p.intensity = image.at<uchar>(v, u) / 256.0;
            current->points.push_back(p);
        }
    }
    
    // depth filter and statistical removal 
    cPointCloudI::Ptr tmp = cPointCloudI::Ptr(new cPointCloudI);
    pcl::StatisticalOutlierRemoval<pXYZI> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(current);
    statistical_filter.filter(*tmp);
    
    (*pointClouds_filter) += *tmp;
}
