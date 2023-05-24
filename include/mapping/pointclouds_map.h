#ifndef _POINTClOUDS_MAP_H_
#define _POINTClOUDS_MAP_H_

// glog
#include <glog/logging.h>

#include <boost/format.hpp> 

// pcl optimalize point_clouds
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/filters/voxel_grid.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/filters/statistical_outlier_removal.h>

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

using namespace Eigen;

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

// 获取三维点云

typedef std::vector<Vector4d, aligned_allocator<Vector4d>> vVector4d;


// 定义点云使用的格式：这里用的是XYZRGB
typedef pcl::PointXYZRGB pXYZRGB;
typedef pcl::PointXYZI pXYZI;
typedef pcl::PointCloud<pXYZRGB> cPointCloudRGB;
typedef pcl::PointCloud<pXYZI> cPointCloudI;
typedef pcl::visualization::PCLVisualizer cPCLViewer;

class PointCloudMap{
public:
    PointCloudMap(){
        point_clouds_thread = std::thread(&PointCloudMap::update_thread, this);
        point_clouds_thread.detach();
    }
    
    PointCloudMap(bool use_pcl, bool use_viewer){
        use_pcl_ = use_pcl;
        use_viewer_ = use_viewer;
        point_clouds_thread = std::thread(&PointCloudMap::update_thread, this);
        point_clouds_thread.detach();
    }
    
    PointCloudMap(
                   double fx_,
                   double fy_, 
                   double cx_, 
                   double cy_, 
                   double deep_scale_,
                   bool use_pcl,
                   bool use_viewer
                  ){
        fx = fx_;
        fy = fy_;
        cx = cx_;
        cy = cy_;
        deep_scale = deep_scale_;
        use_pcl_ = use_pcl;
        use_viewer_ = use_viewer;
        LOG(INFO) << "Dense Mapping Initialized with: Fx, Fy " << fx_ << ", "
                    << fy_ << "\n Cx, Cy " 
                    << cx_ << ", " << cy_;
        point_clouds_thread = std::thread(&PointCloudMap::update_thread, this);
        point_clouds_thread.detach();
    }
    \
    // get raw point clouds
    void get_point_clouds_(cv::Mat image, 
                           cv::Mat depth, 
                           int step=5);

    // get filtered point clouds
    void filter_point_clouds_(cv::Mat image,
                              cv::Mat depth, 
                              int step=1);
    
    void update_thread(){
        while (1){
            if (stop_thread_now) break;
            if (new_keyframe_is_arrive){
                new_keyframe_is_arrive = false;
                if (use_pcl_)
                    filter_point_clouds_(image_, depth_);
                else
                    get_point_clouds_(image_, depth_);
                new_pointcloud_is_generate = true;
                LOG(INFO) << "New Dense map is arrived";
            }
        }
    }
    
    vVector4d get_pointclouds(bool & validate){
        std::unique_lock<std::mutex> lck(data_mutex_);
        validate = new_pointcloud_is_generate;
        new_pointcloud_is_generate = true;
        return pointClouds;
    }
    
    void insert_frame(
        cv::Mat image, 
        cv::Mat depth,
        Sophus::SE3d poses
                     ){
        std::unique_lock<std::mutex> lck(data_mutex_);
        image_ = image.clone();
        depth_ = depth.clone();
        poses_  = poses;
        new_keyframe_is_arrive = true;
    }
    
    void stop_thread(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        stop_thread_now = true;
    }
    
    void save_map(){
        // depth filter and statistical removal 
        cPointCloudI::Ptr tmp = cPointCloudI::Ptr(new cPointCloudI);
        pcl::StatisticalOutlierRemoval<pXYZI> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(pointClouds_filter);
        statistical_filter.filter(*tmp);
        pcl::io::savePCDFileBinary("map.pcd", *tmp);
        LOG(INFO) << "Point CLouds Maps Saved";
    }
    
private:
    double fx = 382.613;
    double fy = 382.613;
    double cx = 320.183;
    double cy = 236.455; // 内参
    double deep_scale = 1000.0; // deep scale --> meter
    
    bool use_pcl_ = true;
    bool use_viewer_ = true;
    bool stop_thread_now = false;
    bool new_keyframe_is_arrive = false;
    bool new_pointcloud_is_generate = false;
    cv::Mat image_;
    cv::Mat depth_;
    Sophus::SE3d poses_;
    
    std::mutex data_mutex_; // lock down thread and data
    std::thread point_clouds_thread;
    vVector4d pointClouds;
    // 新建一个点云
    
    cPointCloudI::Ptr pointClouds_filter = cPointCloudI::Ptr(new cPointCloudI);
    // view Point-clouds
};

typedef std::shared_ptr<PointCloudMap> pPointCloudMap;

#endif
