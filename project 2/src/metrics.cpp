/*
 * metrics.cpp
 *
 * CS365 19 Spring
 *
 * Iris Lian and Qingbo Liu
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

using namespace cv;
using namespace std;

// Baseline Matching
double ssd_metric(const vector<Mat> query,const cv::Mat img){
    double distance = 0.0;

    // calculate the start, end of the 5x5 piece in both query and img
    int query_row_start = query[0].rows/2 - 3; // midpoint-1 is the central index
    int query_col_start = query[0].cols/2 - 3;

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
            // replace nan with 0
            auto query_b = query[0].at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[0];
            auto query_g = query[0].at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[1];
            auto query_r = query[0].at<cv::Vec3f>(query_row_start+i,query_col_start+j).val[2];
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

// helper funciton 0: baseline 1: single color
cv::Mat calc_histogram(Mat img, int k){
    Mat hist;
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32; // 30 32; 15 16
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};

    if(k == 0){
        calcHist(&img, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
        normalize(hist, hist, 0, img.rows*img.cols, cv::NORM_MINMAX);
    }
    else if(k == 1){
        calcHist(&img, 1, 0, cv::Mat(), hist, 1, histSize, ranges, true, false);
        normalize(hist, hist, 1, img.rows*img.cols, cv::NORM_MINMAX);
    }

    return hist;
}

// helper function
std::vector<cv::Mat> calc_textColorHists(Mat img){
    vector<Mat> hists;
    // color
    vector<Mat> img_rgb;
    split(img, img_rgb);
    hists.push_back(calc_histogram(img_rgb[0], 1)); // b
    hists.push_back(calc_histogram(img_rgb[1], 1)); // g
    hists.push_back(calc_histogram(img_rgb[2], 1)); // r
    // texture
    Mat img_gray, img_x, img_y, img_mag;
    cv::cvtColor( img, img_gray, cv::COLOR_BGR2GRAY );
    cv::Sobel(img_gray, img_x, CV_32F, 1, 0);
    cv::Sobel(img_gray, img_y, CV_32F, 0, 1);
    cv::magnitude(img_x, img_y, img_mag);
    hists.push_back(calc_histogram(img_mag, 1));
    return hists;
}

// Baseline Histogram Matching
double baseline_hist_metric(const vector<Mat> hist_query, const cv::Mat img) { 
    return cv::compareHist(hist_query[0], calc_histogram(img, 0), cv::HISTCMP_INTERSECT);
}


// Multiple Histogram Matching
double multi_hist_metric(const vector<Mat> query, const cv::Mat img) { 
    using namespace cv;

    auto img_width = img.cols, img_height = img.rows;

    Mat img_left(img, Rect(0, 0, img_width/5, img_height)),
        img_right(img, Rect(img_width*4/5, 0, img_width/5, img_height)),
        img_middle(img, Rect(img_width/5, 0, img_width*3/5, img_height));

    auto left_cmp = compareHist(query[0], calc_histogram(img_left, 0), HISTCMP_INTERSECT);
    auto right_cmp = compareHist(query[1], calc_histogram(img_right, 0), HISTCMP_INTERSECT);
    auto middle_cmp = compareHist(query[2], calc_histogram(img_middle, 0), HISTCMP_INTERSECT);
    auto whole_cmp = compareHist(query[3], calc_histogram(img, 0), HISTCMP_INTERSECT);

    return left_cmp * 0.05 + right_cmp * 0.05 +
           middle_cmp * 0.7 + whole_cmp * 0.2;

}

// Integrating Texture and Color
double texture_color_metric(const vector<Mat> query, const cv::Mat img){
    auto img_hists = calc_textColorHists(img);
    // Color 
    auto b_cmp = compareHist(query[0], img_hists[0], HISTCMP_CORREL);
    auto g_cmp = compareHist(query[1], img_hists[1], HISTCMP_CORREL);
    auto r_cmp = compareHist(query[2], img_hists[1], HISTCMP_CORREL);

    // Texture 
    auto mag_cmp = compareHist(query[3], img_hists[3], HISTCMP_CORREL);

    return b_cmp*0.1667 + g_cmp*0.1667 + r_cmp*0.1667 + mag_cmp * 0.5; 
}

// Custom Distance Metric
double custom_distance_metric(const vector<Mat> query, const cv::Mat img){
    // whole img single hist
    double whole_cmp = compareHist(query[0], calc_histogram(img, 0), cv::HISTCMP_INTERSECT);

    // center 100*100 texture & color
    int img_row_mid = img.rows/2 - 1;
    int img_col_mid = img.cols/2 - 1;

    cv::Mat img_middle(img, cv::Rect(img_col_mid-50, img_row_mid-50,
                    100, 100));
    // remove the first element
    vector<Mat> newq;
    for(int i = 1; i < query.size(); i++){
        newq.push_back(query[i]);
    }
    double texture_color_cmp = texture_color_metric(newq, img_middle);

    return whole_cmp * 0.5 + texture_color_cmp * 0.5;
}

Mat other_matching_helper(Mat img){
    Mat img_gray,
        img_lap,
        img_lap_abs; 

    GaussianBlur(img, img, Size(3, 3), 0);
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    Laplacian(img_gray, img_lap, CV_16S);
    convertScaleAbs(img_lap, img_lap_abs);

    return calc_histogram(img_lap_abs, 1);
}

double other_matching(const vector<Mat> query, const Mat img) { 
    Mat query_lap_abs, img_lap_abs; 

    query_lap_abs = query[0];
    img_lap_abs = other_matching_helper(img);

    return compareHist(query_lap_abs, img_lap_abs, cv::HISTCMP_CORREL);
}
