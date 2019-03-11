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


template <typename T> 
void displayRegmap(const Mat &regmap, const string &window_name, int N) {
    Mat regmap_colored(regmap.rows, regmap.cols, CV_8U);
    vector<unsigned short> colors(N);

    // assign colors to each region 
    colors[0] = 0; 
    for(int i=1; i<N; i++) 
	colors[i] = 255/i;

    for(int i=0; i<regmap_colored.rows; i++) 
	for(int j=0; j<regmap_colored.cols; j++) 
	    regmap_colored.at<unsigned char>(i, j) = 
		colors[regmap.at<T>(i, j)];

    namedWindow(window_name, 1);
    imshow(window_name, regmap_colored);
}

int centralMoments(const int x_moment, const int y_moment, 
	           const Mat &stats, const Mat &centroids,
		   int label) {
    //int num_pixels = stats(label, CC_STAT_AREA);
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
    // read in the image 
    auto img = imread(img_fp);

    // process the image for segmentation  
    auto processed = process_img(img);

    // segment the image 
    Mat regmap, stats, centroids; 
    int label = cv::connectedComponentsWithStats(processed, regmap, 
	                                         stats, centroids, 8, CV_32S);

    namedWindow(img_fp, 1);
    imshow(img_fp, processed);

    displayRegmap<int>(regmap, "region map", label);

    waitKey(0);

    destroyWindow(img_fp);
    destroyWindow("region map");
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
