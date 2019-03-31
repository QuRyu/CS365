
#include "imgproc.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

static const unsigned char BACKGROUND = 255;
static const unsigned char FOREGROUND = 0; 

using elem_type = unsigned char; 


Mat threshold(const Mat &src) {
  Mat grayscale, dst; 


  cv::cvtColor(src, grayscale, COLOR_BGR2GRAY);
  cv::threshold(grayscale, dst, 100, 255, THRESH_BINARY);

  return dst;
}


Mat morph_shrink(const Mat &src) {
  Mat shrunk; 
  int sum; 

  shrunk.create(src.rows, src.cols, CV_8UC1);

  for (int i=0; i<src.rows; i++) {
    for (int j=0; j<src.cols; j++) {
      auto A1 = i == src.rows-1 ? BACKGROUND : src.at<elem_type>(i+1, j); 
      auto A3 = j == 0 ? BACKGROUND : src.at<elem_type>(i, j-1);
      auto A5 = i == 0 ? BACKGROUND : src.at<elem_type>(i-1, j);
      auto A7 = j == src.cols-1 ? BACKGROUND : src.at<elem_type>(i, j+1);

      // initialize sum and add neighbors 
      // (convert the values to avoid overflow)
      sum = A1; 
      sum += A3; 
      sum += A5;
      sum += A7;

      if (sum > 0) 
        shrunk.at<elem_type>(i, j) = BACKGROUND; 
      else 
        shrunk.at<elem_type>(i, j) = src.at<elem_type>(i, j);
    }
  }

  return shrunk;
}


Mat morph_dilate(const Mat &src) { 
  Mat dilated; 
  int sum; 

  dilated.create(src.rows, src.cols, CV_8UC1);

  for(int i=0; i<src.rows; i++) {
    for(int j=0; j<src.cols; j++) {
      // find adjacent neighbors 
      auto A1 = i == src.rows-1 ? BACKGROUND : src.at<elem_type>(i+1, j); 
      auto A3 = j == 0 ? BACKGROUND : src.at<elem_type>(i, j-1);
      auto A5 = i == 0 ? BACKGROUND : src.at<elem_type>(i-1, j);
      auto A7 = j == src.cols-1 ? BACKGROUND : src.at<elem_type>(i, j+1);

      // initialize sum and add neighbors 
      // (convert the values to avoid overflow)
      sum = A1; 
      sum += A3; 
      sum += A5;
      sum += A7;

      if (sum < BACKGROUND*4) 
        dilated.at<elem_type>(i, j) = FOREGROUND;
      else 
        dilated.at<elem_type>(i, j) = src.at<elem_type>(i, j);
    }
  }

  return dilated; 
}


Mat morph_opening(const Mat &src) { 
  auto shrunk = morph_shrink(src);
  auto dilated = morph_dilate(shrunk);

  return dilated;
}


Mat morph_closing(const Mat &src) {
  auto dilated = morph_dilate(src);
  auto shrunk = morph_shrink(dilated);

  return dilated;
}

