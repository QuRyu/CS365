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

    // parameters for calcHist function 
    int histSize = 8; 
    int channels[] = {0, 1, 2};
    float range[] = {0, 256};
    const float *histRange = {range};

    // matrices to store histograms 
    cv::Mat hist_query, hist_image; 

    cv::Mat mask;

    // calculate histograms 
    cv::calcHist(&query, 1, channels, mask, hist_query, 1, &histSize, &histRange);
    cv::calcHist(&img, 1, channels, mask, hist_image, 1, &histSize, &histRange);

    cv::normalize(hist_query, hist_query, 1, query.rows, cv::NORM_MINMAX);
    cv::normalize(hist_image, hist_image, 1, img.rows, cv::NORM_MINMAX);

    return cv::compareHist(hist_query, hist_image, cv::HISTCMP_CORREL);
}

// Multiple Histogram Matching
double multi_hist_metric(const cv::Mat query, const cv::Mat img) { 
    using namespace cv;

    auto query_width = query.cols, query_height = query.rows,
         img_width = img.cols, img_height = img.rows;
    Mat query_left(query, Rect(0, 0, query_width/5, query_height)),
        query_right(query, Rect(query_width*4/5, 0, 
                query_width/5, query_height)), 
        query_middle(query, Rect(query_width/5, 0,
                    query_width*4/5, img_height));

    Mat img_left(img, Rect(0, 0, img_width/5, img_height)),
        img_right(img, Rect(img_width*4/5, 0, img_width/5, img_height)),
        img_middle(img, Rect(img_width/5, 0, img_width*4/5, img_height));


    auto left_cmp = baseline_hist_metric(query_left, img_left);
    auto right_cmp = baseline_hist_metric(query_right, img_right);
    auto middle_cmp = baseline_hist_metric(query_middle, img_middle);
    auto whole_cmp = baseline_hist_metric(query, img);

    return left_cmp * 0.05 + right_cmp * 0.05 +
           middle_cmp * 0.7 + whole_cmp * 0.2;

}

// Integrating Texture and Color

// Custom Distance Metric
