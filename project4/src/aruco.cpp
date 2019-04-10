/*
 * CS365 Spring 2019 
 * Project 4 
 *
 * Iris Lian, Qingbo Liu
*/
#include <cstdio>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <aruco/aruco.h>

using namespace cv; 
using namespace std; 

// chessboard size 
const int CHESSBOARD_SIZE = 54; 
const int CHESSBOARD_WIDTH = 9;

const int CALIB_NUM_PHOTOS = 5;

auto img_folder = "../data/images/";
auto img_ext = ".jpg";

auto coefficient_data_file = "../data/cof.txt";

int main(int argc, char *argv[]) {
  // first read in camera matrix and distortion coefficients 
  FileStorage fs(coefficient_data_file, FileStorage::READ);
  Mat camera_matrix(3, 3, CV_64FC1), 
      dist_coeff(1, 5, CV_64FC1);

  fs["camMat"] >> camera_matrix;
  fs["distMat"] >> dist_coeff;

  cout << "camera_matrix" << camera_matrix << endl 
       << "distortion matrix" << dist_coeff << endl;

  // start video loop 
	cv::VideoCapture *capdev;

	// open the video device
	capdev = new cv::VideoCapture(1);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		             (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	cv::namedWindow("Video", 0); // identifies a window?
	cv::Mat frame;

  *capdev >> frame; 

  cout << "initialized camera matrix: " << camera_matrix << endl << endl;

  aruco::MarkerDetector MDetector; 
  MDetector.setDictionary("ARUCO_MIP_36h12");

	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

    for (auto m : MDetector.detect(frame)) {
      cout << m << endl; 
      m.draw(frame); 
    }

    cv::imshow("Video", frame);

    auto key = cv::waitKey(10); 

		if (key == 'q') {
		  break;
    } 
	}

	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

	return(0);
}
