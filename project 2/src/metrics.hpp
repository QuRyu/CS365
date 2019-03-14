#ifndef METRICS_H
#define METRICS_H 

#include <opencv2/opencv.hpp> 

double ssd_metric(const std::vector<cv::Mat> query,const cv::Mat img);

cv::Mat calc_histogram(cv::Mat img, int k);

std::vector<cv::Mat> calc_textColorHists(cv::Mat img);

double baseline_hist_metric(const std::vector<cv::Mat>, const cv::Mat);

double multi_hist_metric(const std::vector<cv::Mat>, const cv::Mat);

double texture_color_metric(const std::vector<cv::Mat>, const cv::Mat);

double custom_distance_metric(const std::vector<cv::Mat>, const cv::Mat img);

cv::Mat other_matching_helper(cv::Mat img);

double other_matching(const std::vector<cv::Mat>, const cv::Mat);

#endif 
