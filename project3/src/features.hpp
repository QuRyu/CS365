#ifndef FEATURES_HPP 
#define FEATURES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Moments>
compute_mulitiple_moments(const cv::Mat &src);

void compute_multiple_HuMoments(const cv::Mat &src, std::vector<double *> huMoments);

void compute_single_HuMoments(const cv::Mat &src, double *hu);

double compute_entropy(const cv::Mat &src);

void compute_features(const cv::Mat &src, double *f);

#endif // FEATURES_HPP
