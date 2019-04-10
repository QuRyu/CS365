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
	capdev = new cv::VideoCapture(0);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		             (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	cv::namedWindow("Video", 1); // identifies a window?
	cv::Mat frame, gray;

  int img_counter = 0;

  Size patternsize(9, 6); 
  vector<Point2f> corner_set; 
  vector<Vec3f> point_set; 
  vector<vector<Point2f>> corner_list; 
  vector<vector<Vec3f>> point_list;

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

    // check presence of chessboard  
    bool found = findChessboardCorners(gray, patternsize, corner_set, 
                          CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH
                          + CALIB_CB_NORMALIZE_IMAGE); 

    if (found) { 
       //refine the points 
      cornerSubPix(gray, corner_set, Size(10, 10), Size(-1, -1), 
                    TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.1));

      // draw points on the chessboard 
      //drawChessboardCorners(frame, patternsize, corner_set, found);

      // fill in point_set 
      point_set.clear();
      for (int i=0; i<CHESSBOARD_SIZE; i++) {
        int y = -(i / CHESSBOARD_WIDTH);
        int x = i + y * CHESSBOARD_WIDTH;
        point_set.emplace_back(x, y, 0);
      }

      // solve for rotation and translation 
      Mat rvec, tvec;
      solvePnP(point_set, corner_set, camera_matrix, dist_coeff, rvec, tvec);
      cout << "rotations: " << rvec << endl << "translations: " << tvec << endl;

      // 3D coordinate axis
      vector<Vec3f> points{point_set[0], point_set[2], point_set[18], Vec3f(0, 0, 2)};
      vector<Point2f> imgPoints; 
      projectPoints(points, rvec, tvec, camera_matrix, dist_coeff, imgPoints);
      arrowedLine(frame, imgPoints[0], imgPoints[1], Scalar(255, 0, 0)); // x-axis 
      arrowedLine(frame, imgPoints[0], imgPoints[2], Scalar(0, 255, 0)); // y-axis 
      arrowedLine(frame, imgPoints[0], imgPoints[3], Scalar(0, 0, 255)); // z-axis 

      // extension - cover the target
      points.clear(); 
      imgPoints.clear();

      points.emplace_back(-1,-1,0);
      points.emplace_back(9,6,0);
      projectPoints(points, rvec, tvec, camera_matrix, dist_coeff, imgPoints);
      rectangle(frame, imgPoints[0], imgPoints[1], Scalar(0), FILLED);

      // a pyramid 
      points.clear(); 
      imgPoints.clear();

      points.push_back(point_set[45]); // left angle
      points.push_back(point_set[53]); // right angle 
      points.push_back(point_set[4]); // top angle 
      auto middle = point_set[31]; // middle point
      middle[2] = 4;
      points.push_back(middle);
      projectPoints(points, rvec, tvec, camera_matrix, dist_coeff, imgPoints);

      line(frame, imgPoints[0], imgPoints[1], Scalar(100, 100, 100), 4);
      line(frame, imgPoints[0], imgPoints[2], Scalar(100, 100, 100), 4);
      line(frame, imgPoints[1], imgPoints[2], Scalar(100, 100, 100), 4);
      line(frame, imgPoints[0], imgPoints[3], Scalar(100, 100, 100), 4);
      line(frame, imgPoints[1], imgPoints[3], Scalar(100, 100, 100), 4);
      line(frame, imgPoints[2], imgPoints[3], Scalar(100, 100, 100), 4);
    }


    cv::imshow("Video", frame);

    auto key = cv::waitKey(10); 

		if (key == 'q') {
		  break;
    } else if (key == 's' && found) { 
      // save corners and point 
      //corner_list.push_back(corner_set);
      //point_set.clear();


      //point_list.push_back(point_set);

      //// save image 
      //stringstream ss; 
      //ss << img_folder << img_counter << img_ext;
      //imwrite(ss.str(), frame);
      //img_counter++;

    }
	}

	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

	return(0);
}
