/*
 * metrics.cpp
 *
 * CS365 19 Spring 
 *
 * Assignment 2: Content-based image retrieval 
 *
 */
#include <cstdio> 
#include <cstdlib>
#include <dirent.h>
#include <cstring> 
#include <string> 
#include <map> 

#include <opencv2/opencv.hpp> 

#include "utilities.cpp"

// Baseline Matching
double ssd_metric(const cv::Mat query,const cv::Mat img){
    double distance = 0.0;

    return distance;
}

// Baseline Histogram Matching
double baseline_hist_metric(const cv::Mat query, const cv::Mat img) { 
    
    // separate images in 3 places (B, R, G)
    std::vector<cv::Mat> brg_planes_query; 
    std::vector<cv::Mat> brg_planes_img; 

    cv::split(query, brg_planes_query);
    cv::split(img, brg_planes_img);

    // parameters for calcHist function 
    int histSize = 256; 
    float range[] = {0, 256};
    const float *histRange = {range};

    // matrices to store histograms 
    cv::Mat b_hist_query, g_hist_query, r_hist_query; 
    cv::Mat b_hist_img, g_hist_img, r_hist_img; 

    cv::Mat mask;

    // calculate histograms 
    cv::calcHist( &brg_planes_query[0], 1, 0, mask, b_hist_query, 1, &histSize, &histRange);
    cv::calcHist( &brg_planes_query[1], 1, 0, mask, g_hist_query, 1, &histSize, &histRange);
    cv::calcHist( &brg_planes_query[2], 1, 0, mask, r_hist_query, 1, &histSize, &histRange);

    cv::calcHist( &brg_planes_img[0], 1, 0, mask, b_hist_img, 1, &histSize, &histRange);
    cv::calcHist( &brg_planes_img[1], 1, 0, mask, g_hist_img, 1, &histSize, &histRange);
    cv::calcHist( &brg_planes_img[2], 1, 0, mask, r_hist_img, 1, &histSize, &histRange);
    
    std::cout << type2str(b_hist_img.type()) << std::endl;

    return 0; 
}

// Multiple Histogram Matching

// Integrating Texture and Color

// Custom Distance Metric
