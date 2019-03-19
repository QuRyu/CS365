#ifndef FEATURES_HPP 
#define FEATURES_HPP

#include <fstream> 

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

struct Features {
    std::vector<double> feature; 
    double centroid_x; 
    double centroid_y; 
    double orientation; 
    std::vector<std::vector<cv::Point>> contours;
    std::string label;

    Features() { }

    Features(std::vector<double> features, std::vector<std::vector<cv::Point>> contours,
    	double ctd_x, double ctd_y, double ort) : 
		feature(features), centroid_x(ctd_x), 
		centroid_y(ctd_y), orientation(ort) {
    }

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

    void write_to_fstream(std::fstream &stream);
};

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
