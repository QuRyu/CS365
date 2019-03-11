#ifndef IMGPROC_HPP 
#define IMGPROC_HPP

#include <opencv2/opencv.hpp>

cv::Mat threshold(const cv::Mat &src);

// closing: grow and then shrink 
// closes holes in target objects 
cv::Mat morph_closing(const cv::Mat &src);

// opening: shrink and then grow 
// eliminates noise 
cv::Mat morph_opening(const cv::Mat &src);

cv::Mat morph_dilate(const cv::Mat &src); 

// assume the type is 8UC1 -- unsigned char 
// use 4-connected ways 
cv::Mat morph_shrink(const cv::Mat &src);

#endif // IMGPROC_HPP
