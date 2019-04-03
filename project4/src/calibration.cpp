/*
 * CS365 Spring 2019 
 * Project 4 
 *
 * Iris Lian, Qingbo Liu
*/
#include <cstdio>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std; 

// chessboard size 
const int CHESSBOARD_SIZE = 54; 
const int CHESSBOARD_WIDTH = 9;

const int CALIB_NUM_PHOTOS = 5;

auto img_folder = "../data/images/";
auto img_ext = ".jpg";

int main(int argc, char *argv[]) {
	cv::VideoCapture *capdev;

	// open the video device
	capdev = new cv::VideoCapture(0);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		             (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	cv::namedWindow("Video", 0); // identifies a window?
	cv::Mat frame;

  int img_counter = 0;

  Size patternsize(9, 6); 
  vector<Point2f> corner_set; 
  vector<Vec3f> point_set; 
  vector<vector<Point2f>> corner_list; 
  vector<vector<Vec3f>> point_list;

  *capdev >> frame; 

  Mat camera_matrix = Mat::eye(3, 3, CV_64FC1);
  camera_matrix.at<double>(0, 2) = frame.cols/2;
  camera_matrix.at<double>(1, 2) = frame.rows/2;

	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

    // check presence of chessboard  
    bool found = findChessboardCorners(frame, patternsize, corner_set, 
                          CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH
                          + CALIB_CB_NORMALIZE_IMAGE); 

    if (found) { 
      //cout << "chessboard found, corner size " << corner_set.size() << endl; 
      //cout << "first corner (" << corner_set[0].x << ", " 
                        //<< corner_set[0].y << ")" << endl;

      drawChessboardCorners(frame, patternsize, corner_set, found);
    }


    cv::imshow("Video", frame);

    auto key = cv::waitKey(10); 

		if (key == 'q') {
		  break;
    } else if (key == 's' && found) { 
      // save corners and point 
      corner_list.push_back(corner_set);
      point_set.clear();

      for (int i=0; i<CHESSBOARD_SIZE; i++) {
        int y = -(i / CHESSBOARD_WIDTH);
        int x = i + y * CHESSBOARD_WIDTH;
        point_set.emplace_back(x, y, 0);
      }

      point_list.push_back(point_set);

      // save image 
      stringstream ss; 
      ss << img_folder << img_counter << img_ext;
      imwrite(ss.str(), frame);
      img_counter++;

      if (img_counter == CALIB_NUM_PHOTOS) {
        Mat dist_coeff, rvecs, tvecs;
        auto reprojection_error = calibrateCamera(
                    point_list, corner_list, frame.size(), camera_matrix, 
                    dist_coeff, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO); 
      }
    }
	}

	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

	return(0);
}
