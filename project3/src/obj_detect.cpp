/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

*/

#include <string>
#include <cstdio>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "imgproc.hpp"
#include "utilities.hpp"
#include "features.hpp"

using namespace std;
using namespace cv;


template <typename T> 
void displayRegmap(const Mat &regmap, const string &window_name, int N) {
    Mat regmap_colored(regmap.rows, regmap.cols, CV_8UC1);
    vector<unsigned short> colors(N);

    // assign colors to each region 
    colors[0] = 0; 
    for(int i=1; i<N; i++) 
	colors[i] = 255/i;

    for (int i=0; i<regmap_colored.rows; i++) 
	for (int j=0; j<regmap_colored.cols; j++) 
	    regmap_colored.at<unsigned char>(i, j) = 
		colors[regmap.at<T>(i, j)];

    namedWindow(window_name, 1);
    imshow(window_name, regmap_colored);
}

template <typename T>
double centralMoments(const int x_moment, const int y_moment,  
		   const Mat &img, const Mat &regmap, 
		   const Mat &stats, const Mat &centroids, int label) {
    // extract variables for readability
    double centroid_x = centroids.at<double>(label, 0); 
    double centroid_y = centroids.at<double>(label, 1); 

    int num_pixels = stats.at<int>(label, CC_STAT_AREA);
    int left_x = stats.at<int>(label, CC_STAT_LEFT);
    int top_y = stats.at<int>(label, CC_STAT_TOP);
    int width = stats.at<int>(label, CC_STAT_WIDTH);
    int height = stats.at<int>(label, CC_STAT_HEIGHT);

    int right_x = left_x + width;
    int down_y = top_y + height;

    cout << "object label " << label << endl;
    cout << "number of pixels " << num_pixels << endl;
    cout << "left top corner (" << left_x << ", " << top_y 
	<< "), width " << width << ", height " << height << endl;
    cout << "centroid_x " << centroid_x << ", centroid_y " << centroid_y << endl;
    cout << endl;

    double sum = 0; 

    for (int j=top_y; j<down_y; j++) {
	for (int i=left_x; i<right_x; i++) {
	    //cout << img.at<T>(i, j) << endl;
	    //if (img.at<T>(i, j) != 0) printf("%u\n", img.at<T>(i, j));
	    //cout << pow(i-centroid_x, x_moment) << " " << pow(j-centroid_y, y_moment) << " " << img.at<T>(i, j) << endl;
	    double value = pow(i-centroid_x, x_moment) * pow(j-centroid_y, y_moment) * img.at<T>(j, i);
	    sum += value; 
	    //cout << value << " "; 
	}
	//cout << endl;
    }

    cout << "sum before scaling-invariant division " << sum << endl;
    cout << endl;


    sum = sum / num_pixels; // make it scaling invariant  

    return sum; 

}

template <typename T>
double orientation_alpha(const Mat &img, const Mat &regmap, 
			 const Mat &stats, const Mat &centroids, int label) {
    double u11 = centralMoments<T>(1, 1, img, regmap, stats, centroids, label);
    double u20 = centralMoments<T>(2, 0, img, regmap, stats, centroids, label);
    double u02 = centralMoments<T>(0, 2, img, regmap, stats, centroids, label);

    double alpha = atan2(2*u11, u20-u02);

    return alpha; 
}

template <typename T>
double orientedCentralMoments(const Mat &img, const Mat &regmap, 
			      const Mat &stats, const Mat &centroids, 
			      double alpha, int label) {
    // extract variables for readability
    double centroid_x = centroids.at<double>(label, 0); 
    double centroid_y = centroids.at<double>(label, 1); 

    int num_pixels = stats.at<int>(label, CC_STAT_AREA);
    int left_x = stats.at<int>(label, CC_STAT_LEFT);
    int top_y = stats.at<int>(label, CC_STAT_TOP);
    int width = stats.at<int>(label, CC_STAT_WIDTH);
    int heigth = stats.at<int>(label, CC_STAT_HEIGHT);

    double sum = 0; 
    for (int i=left_x; i<width; i++) 
	for (int j=top_y; j<heigth; j++) 
	    sum += pow((j-centroid_y)*cos(alpha) + (i-centroid_x)*sin(alpha), 2)
		    * img.at<T>(i, j); 

    sum = sum / num_pixels;

    return sum; 
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

    auto alpha = orientation_alpha<unsigned char>(regmap, regmap, stats, centroids, 1);
    auto moment = orientedCentralMoments<unsigned char>(processed, regmap, stats, centroids, alpha, 1);

    cout << "alpha " << alpha << ", moment " << moment << endl;

    double f[9];
    compute_features(img, f);

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

    printf("%s\n", images_fp[5].c_str());
    process_one_image(images_fp[5]);



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
