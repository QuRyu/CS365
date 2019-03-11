/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

*/

#include <string>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include "imgproc.hpp"
#include "utilities.hpp"

using namespace std;
using namespace cv;

static const unsigned char BACKGROUND = 255;
static const unsigned char FOREGROUND = 0; 

using elem_type = unsigned char; 


pair<int, Mat> comp_seg(const Mat &src) { 
    Mat regmap; 

    int label = cv::connectedComponents(src, regmap); 

    return make_pair(label, regmap);
}



Mat process_img(const Mat &src) {
    // threshold 
    auto thresholded = threshold(src);
 
    // morphological processing 
    auto morph_opened = morph_opening(thresholded); 
    auto morph_closed = morph_closing(morph_opened);

    return morph_closed;
}

void process_one_image(const string &img_fp) {
    auto img = imread(img_fp);
    auto processed = process_img(img);

    namedWindow(img_fp, 1);
    imshow(img_fp, processed);

    waitKey(0);

    destroyWindow(img_fp);
}

void process_images(const vector<string> &images_fp) {
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

    process_one_image(images_fp[0]);



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
