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
	cv::Mat frame, gray;

  int img_counter = 0;

  Size patternsize(9, 6); 
  vector<Point2f> corner_set; 

  *capdev >> frame; 

  cout << "initialized camera matrix: " << camera_matrix << endl << endl;


	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

    // first convert img to gray
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    int thres = 220;

    Mat dst, dst_norm;
    cornerHarris(gray, dst, 5, 3, 0.005); 

    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1);


    corner_set.clear();
    for (int j=0; j<frame.rows; j++) {
      for (int i=0; i<frame.cols; i++) {
        if ( static_cast<int>(dst_norm.at<float>(j, i)) > thres) {
          corner_set.push_back(Point(i, j));
        }
      }
    }

    if (!corner_set.empty()) {
      cornerSubPix(gray, corner_set, Size(5, 5), Size(-1, -1), 
                      TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.1));

      if (corner_set.size() > 3)  {
          line(frame, corner_set[0], corner_set[1], Scalar(0), 4);
          line(frame, corner_set[0], corner_set[2], Scalar(0), 4);
          line(frame, corner_set[1], corner_set[2], Scalar(0), 4);

      }
    }


    cv::imshow("Video", frame);

    auto key = cv::waitKey(30); 

    if (key == 'q') {
      break;
    } 
	}

	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

	return(0);
}
