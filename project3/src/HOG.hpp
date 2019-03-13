#ifndef HOG_HPP 
#define HOG_HPP

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void computeMagAngle(InputArray src, OutputArray mag, OutputArray ang);

void computeHOG(InputArray mag, InputArray ang, OutputArray dst, int dims, bool isWeighted);

#endif // HOG_HPP
