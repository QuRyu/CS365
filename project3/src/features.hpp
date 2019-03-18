#ifndef FEATURES_HPP 
#define FEATURES_HPP

#include <fstream> 

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Moments>
compute_mulitiple_moments(const cv::Mat &src);

void compute_multiple_HuMoments(const cv::Mat &src, std::vector<double *> huMoments);

void compute_single_HuMoments(const cv::Mat &src, double *hu);

double compute_entropy(const cv::Mat &src);

void compute_features(const cv::Mat &src, double *f);

struct Features {
    std::vector<double> feature; 
    double centroid_x; 
    double centroid_y; 
    double orientation; 

    Features() { }

    Features(std::vector<double> features) : Features(features, 0, 0, 0) { 
    }

    Features(std::vector<double> features, double ctd_x, 
	     double ctd_y, double ort) : 
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

#endif // FEATURES_HPP
