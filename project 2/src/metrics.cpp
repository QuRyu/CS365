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

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };

    int channels[] = {0, 1};

    // matrices to store histograms 
    cv::Mat hist_query, hist_image; 

    // calculate histograms 
    cv::calcHist(&query, 1, channels, cv::Mat(), hist_query, 2, histSize, ranges, true, false);
    cv::calcHist(&img, 1, channels, cv::Mat(), hist_image, 2, histSize, ranges, true, false);
    

    cv::normalize(hist_query, hist_query, 1, query.rows*query.cols, cv::NORM_MINMAX);
    cv::normalize(hist_image, hist_image, 1, img.rows*img.cols, cv::NORM_MINMAX);

    return cv::compareHist(hist_query, hist_image, cv::HISTCMP_INTERSECT);
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

// helper function for texture_color_metric
double single_color_hist(const cv::Mat query, const cv::Mat img) { 
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };

    // matrices to store histograms 
    cv::Mat hist_query, hist_image; 

    // calculate histograms 
    cv::calcHist(&query, 1, 0, cv::Mat(), hist_query, 1, histSize, ranges, true, false);
    cv::calcHist(&img, 1, 0, cv::Mat(), hist_image, 1, histSize, ranges, true, false);
    

    cv::normalize(hist_query, hist_query, 1, query.rows*query.cols, cv::NORM_MINMAX);
    cv::normalize(hist_image, hist_image, 1, img.rows*img.cols, cv::NORM_MINMAX);

    return cv::compareHist(hist_query, hist_image, cv::HISTCMP_INTERSECT);
}

// Integrating Texture and Color
double texture_color_metric(const cv::Mat query, const cv::Mat img){
    // Color 
    std::vector<cv::Mat> query_rgb, img_rgb;
    cv::split(query, query_rgb);
    cv::split(img, img_rgb);

    auto b_cmp = single_color_hist(query_rgb[0], img_rgb[0]);
    auto g_cmp = single_color_hist(query_rgb[1], img_rgb[1]);
    auto r_cmp = single_color_hist(query_rgb[2], img_rgb[2]);


    // Texture 
    // Containers
    cv::Mat query_gray, img_gray, query_x, query_y, img_x, img_y;
    cv::Mat query_mag, img_mag; 

    // Convert the images to Gray 
    cv::cvtColor( query, query_gray, cv::COLOR_BGR2GRAY );
    cv::cvtColor( img, img_gray, cv::COLOR_BGR2GRAY );

    // Apply Sobel filter (x)
    cv::Sobel(query_gray, query_x, CV_32F, 1, 0);
    cv::Sobel(img_gray, img_x, CV_32F, 1, 0);

    // Apply Sobel filter (y)
    cv::Sobel(query_gray, query_y, CV_32F, 0, 1);
    cv::Sobel(img_gray, img_y, CV_32F, 0, 1);

    // calculate the magnitude of gradient
    cv::magnitude(query_x, query_y, query_mag);
    cv::magnitude(img_x, img_y, img_mag);

    auto mag_cmp = single_color_hist(query_mag, img_mag);

    return b_cmp*0.1667 + g_cmp*0.1667 + r_cmp*0.1667 + mag_cmp * 0.5; 
}

// Custom Distance Metric
double custom_distance_metric(const cv::Mat query, const cv::Mat img){
    // whole img single hist
    double whole_cmp = baseline_hist_metric(query, img);

    // center 100*100 texture & color
    int query_row_mid = query.rows/2 - 1; // midpoint-1 is the central index
    int query_col_mid = query.cols/2 - 1;

    int img_row_mid = img.rows/2 - 1;
    int img_col_mid = img.cols/2 - 1;

    cv::Mat query_middle(query, cv::Rect(query_col_mid-50, query_row_mid-50,
                    query_col_mid+50, query_row_mid+50));
    cv::Mat img_middle(img, cv::Rect(img_col_mid-50, img_row_mid-50,
                    img_col_mid+50, img_row_mid+50));

    double texture_color_cmp = texture_color_metric(query_middle, img_middle);

    return whole_cmp * 0.5 + texture_color_cmp * 0.5;
}
