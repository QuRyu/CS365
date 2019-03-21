#ifndef FEATURES_HPP 
#define FEATURES_HPP

#include <fstream> 

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// number of features for each image 
const int NUM_OF_FEATURES = 10;

struct Features {
    std::vector<double> feature; 
    double centroid_x; 
    double centroid_y; 
    double orientation; 
    //std::vector<std::vector<cv::Point>> contours;
    std::string label;

    Features() { }

    //Features(std::vector<double> features, std::vector<std::vector<cv::Point>> contours,
    Features(std::vector<double> features, 
    	double ctd_x, double ctd_y, double ort) : 
		feature(features), centroid_x(ctd_x), 
		centroid_y(ctd_y), orientation(ort) {
    }

    Features(const Features &f) = default;

    inline 
    int num_of_features() const { 
	return feature.size();
    }

    inline 
    double operator[](std::size_t pos) {
	return feature[pos];
    }

    inline 
    double operator[](std::size_t pos) const { 
	return feature[pos];
    }

};

std::ostream& operator<<(std::ostream& os, const Features &f);

std::istream& operator>>(std::istream& is, Features &f);

std::vector<std::vector<cv::Point>> compute_contours(const cv::Mat &src);

std::vector<cv::Moments>
compute_multiple_moments(const cv::Mat &src);

std::vector<double *> compute_multiple_HuMoments(const cv::Mat &src);

void compute_single_HuMoments(const cv::Mat &src, double *hu);

double compute_entropy(const cv::Mat &src);

std::vector<double> compute_multiple_entropy(const cv::Mat &src, std::vector<std::vector<cv::Point>> contours);

std::vector<double> compute_HWRatios(std::vector<std::vector<cv::Point>> contours);

std::vector<double> compute_percentArea(std::vector<std::vector<cv::Point>> contours);

std::vector<double> compute_centriods_ort(const cv::Mat &src);

Features compute_features(const cv::Mat &src);

#endif // FEATURES_HPP
