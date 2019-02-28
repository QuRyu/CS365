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
#include <cmath>

#include <opencv2/opencv.hpp> 

#include "utilities.cpp"

// Baseline Matching
double ssd_metric(const cv::Mat query,const cv::Mat img){
    double distance = 0.0;

    // calculate the start, end of the 5x5 piece in both query and img
    int query_row_start = query.rows/2 - 3; // midpoint-1 is the central index
    int query_col_start = query.cols/2 - 3;

    int img_row_start = img.rows/2 - 3;
    int img_col_start = img.cols/2 - 3;

    // create containers for squares of differences of each color channel
    double b_arr[5][5];
    double g_arr[5][5];
    double r_arr[5][5];

    // loop through the two pieces
    int i,j;
    for(i = 0; i < 5; i++){
        for(j = 0; j < 5; j++){
            // std::cout << "blue channel img value: " << img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[0] << std::endl;
            // replace nan with 0
            auto query_b = query.at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[0];
            auto query_g = query.at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[1];
            auto query_r = query.at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[2];
            auto img_b = img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[0];
            auto img_g = img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[1];
            auto img_r = img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[2];

            if(isnan(query_b)) query_b = 0;
            if(isnan(query_g)) query_g = 0;
            if(isnan(query_r)) query_r = 0;
            if(isnan(img_b)) img_b = 0;
            if(isnan(img_g)) img_g = 0;
            if(isnan(img_r)) img_r = 0;

            b_arr[i][j] = pow(abs(query_b - img_b),2);
            g_arr[i][j] = pow(abs(query_g - img_g),2);
            r_arr[i][j] = pow(abs(query_r - img_r),2);

            // std::cout << "LOOK " << b_arr[i][j] << " "<< g_arr[i][j]<< " " << r_arr[i][j] << std::endl;
        }
    }

    // sum up
    int k,l;
    for(k = 0; k < 5; k++){
        for(l = 0; l < 5; l++){
            distance += (b_arr[k][l]+g_arr[k][l]+r_arr[k][l]);
        }
    }

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
double texture_color_metric(const cv::Mat query, const cv::Mat img){
    // Containers
    cv::Mat query_gray, img_gray, query_x, query_y, img_x, img_y;

    // Convert the images to Gray 
    cv::cvtColor( query, query_gray, CV_BGR2GRAY );
    cv::cvtColor( img, img_gray, CV_BGR2GRAY );

    // Apply Sobel filter (x)
    cv::Sobel(query_gray, query_x, query_gray.depth(), 1, 0);
    cv::Sobel(img_gray, img_x, img_gray.depth(), 1, 0);

    // Apply Sobel filter (y)
    cv::Sobel(query_gray, query_y, query_gray.depth(), 0, 1);
    cv::Sobel(img_gray, img_y, img_gray.depth(), 0, 1);

    
}

// Custom Distance Metric
