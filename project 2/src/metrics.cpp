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
            b_arr[i][j] = pow(query.at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[0]
                            - img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[0],2);
            g_arr[i][j] = pow(query.at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[1]
                            - img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[1],2);
            r_arr[i][j] = pow(query.at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[2]
                            - img.at<cv::Vec3f>(img_row_start+i,img_col_start+j).val[2],2);
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

// Multiple Histogram Matching

// Integrating Texture and Color

// Custom Distance Metric
