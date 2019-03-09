/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

*/

#include <string>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include "utilities.hpp"

using namespace std;
using namespace cv;

Mat threshold(const Mat &src) {
    Mat grayscale, dst; 

    cv::cvtColor(src, grayscale, COLOR_BGR2GRAY);
    cout << grayscale.at<int>(0,0) << endl;
    cv::threshold(grayscale, dst, 120, 255, THRESH_BINARY);

    return dst;
}

Mat process_img(const Mat &src) {
    Mat dst; 

    dst = threshold(src);

    return dst;
}

int main(int argc, char *argv[]) {
    bool camera;
    string img_fp; 


    //switch(argc) {  
	//case 2: // use the camera
	    //camera = true; 
	    //break;
	//case 3: // takes a set of images 
	    //camera = false; 
	    //img_fp = argv[2];
	    //break;
	//default:
	    //cerr << "unsupported number of parameters" << endl;
	    //exit(-1);
    //}


    // assuming we are using a list of images 
    camera = false;
    img_fp = argv[2];
    int num_images_to_show = 10;

    auto images_fp = traverse_dir(img_fp);

    for (auto img_fp : images_fp) {
	auto img = imread(img_fp);
	auto processed = process_img(img);

	namedWindow(img_fp, 1);
	imshow(img_fp, processed);
    }

    cv::waitKey(0);

    for (auto img_fp : images_fp) {
	destroyWindow(img_fp);
    }



    if (camera) {
        cv::VideoCapture *capdev;
        char label[256];
        int quit = 0;
        int frameid = 0;
        char buffer[256];
        std::vector<int> pars;
        
        pars.push_back(5);
        
        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
        	printf("Unable to open video device\n");
        	return(-1);
        }
        
        strcpy(label, argv[1]);
        
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
        	       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        
        printf("Expected size: %d %d\n", refS.width, refS.height);
        
        cv::namedWindow("Video", 1); // identifies a window?
        cv::Mat frame;
        
        
	for(;!quit;) {
	    *capdev >> frame; // get a new frame from the camera, treat as a stream
        
	    if( frame.empty() ) {
		printf("frame is empty\n");
		break;
            }
          
          cv::imshow("Video", frame);
        
          int key = cv::waitKey(10);
        
          switch(key) {
	    case 'q':
              quit = 1;
              break;
              
	    case 'c': // capture a photo if the user hits c
              sprintf(buffer, "%s.%03d.png", label, frameid++);
              cv::imwrite(buffer, frame, pars);
              printf("Image written: %s\n", buffer);
              break;
   
	    default: break;
	   }
   
   
   
	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

    }

	return(0);

    }}
