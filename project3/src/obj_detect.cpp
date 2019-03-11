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

static const unsigned char BACKGROUND = 255;
static const unsigned char FOREGROUND = 0; 

using elem_type = unsigned char; 

Mat threshold(const Mat &src) {
    Mat grayscale, dst; 


    cv::cvtColor(src, grayscale, COLOR_BGR2GRAY);
    cv::threshold(grayscale, dst, 120, 255, THRESH_BINARY);

    return dst;
}

pair<int, Mat> comp_seg(const Mat &src) { 
    Mat regmap; 

    int label = cv::connectedComponents(src, regmap); 

    return make_pair(label, regmap);
}

// assume the type is 8UC1 -- unsigned char 
// use 4-connected ways 
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

// opening: shrink and then grow 
// eliminates noise 
Mat morph_opening(const Mat &src) { 
    auto shrunk = morph_shrink(src);
    auto dilated = morph_dilate(shrunk);

    return dilated;
}

// closing: grow and then shrink 
// closes holes in target objects 
Mat morph_closing(const Mat &src) {
    auto dilated = morph_dilate(src);
    auto shrunk = morph_shrink(dilated);

    return dilated;
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
