#ifndef METRICS_H
#define METRICS_H 

#include <opencv2/opencv.hpp> 

double ssd_metric(const cv::Mat,const cv::Mat);

double baseline_hist_metric(const cv::Mat, const cv::Mat, int);

#endif 