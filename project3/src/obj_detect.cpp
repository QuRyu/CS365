/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

*/

#include <string>
#include <cstdio>
#include <cmath>
#include <regex>

#include <opencv2/opencv.hpp>

#include "imgproc.hpp"
#include "utilities.hpp"
#include "features.hpp"

using namespace std;
using namespace cv;


// return the DB path 
std::string DB_path() {
#ifdef macOS_Qingbo 
    string path = "/Users/HereWegoR/Document/CS/CS365/project3/data/db.txt";
    return path;
#endif 

#ifdef linux_Iris
    // fill the path 
    return "/personal/ylian/CS365/CS365/project3/data/db.txt";
#endif 
}

Mat process_img(const Mat &src) {
    // threshold 

    auto thresholded = threshold(src);
 
    // morphological processing 
    auto morph_opened = morph_opening(thresholded); 
    auto morph_closed = morph_closing(morph_opened);

    return morph_closed;
}


// extra the label name given a string name 
string extract_label(const string &str) {
    regex label_regex("(\\w+)\\.\\d*\\.\\w*$");
    if (regex_search(str, label_regex)) {
	auto label_begin = sregex_iterator(str.begin(), str.end(), label_regex);
	smatch match = *label_begin;
	return match[1].str();
    } else 
	return str;


}

Features process_one_image(const Mat &img, const string &label) {
    // process the image for segmentation  
    auto processed = process_img(img);

    double f[9];
    compute_features(img, f);

    auto label_ex = extract_label(label);

    // wrap all features in the Feature struct  

    // add the label to the feature 
    return Features();
}


vector<Features> process_multiple_images(const vector<Mat> &images, 
	                                 const vector<string> &labels) {
    assert(images.size() == labels.size());
    vector<Features> features;

    for (int i=0; i<images.size(); i++) {
	features.push_back(process_one_image(images[i], labels[i]));
    }

    return features;
}

int main(int argc, char *argv[]) {
    /* Argument Format: source (path)
     * source: 0 for camera and 1 for directories 
     * path: if we use the directories, provide the path 
     */

    bool camera;
    string img_fp; 
    fstream db_stream(DB_path());

    if (argc < 1) {
	cerr << "no argument provided" << endl;
	exit(-1);
    } else {
	int source = atoi(argv[1]);
	camera = source == 0 ? true : false; 

	if (!camera) 
	    img_fp = argv[2];
    }

    // assuming we are using a list of images 
    
    if (!camera) { // we are using a list of directories 
	auto images_fp = traverse_dir(img_fp);
	vector<Mat> images;

	for (auto img_fp : images_fp) {
	    auto img = imread(img_fp); 
	    images.push_back(img);
	}

	auto features = process_multiple_images(images, images_fp);

	for (auto f : features) {
	    f.write_to_fstream(db_stream);
	}

	while (true) {
	    cout << "the path of photo to compare: " << endl; 

	    string cmp_path; 
	    cin >> cmp_path; 

	    auto img = imread(cmp_path);
	    auto cmp_feature = process_one_image(img, cmp_path); 

	    // use classifier to find which image 
	}

    } else if (camera) {
        cv::VideoCapture *capdev;
        char label[256];
        int quit = 0;
        int frameid = 0;
        char buffer[256];
        std::vector<int> pars;
	bool training_mode = true; 

	vector<Mat> images; // the set of images used as a training set 
	                    // captured from the camera 
	vector<Features> features; // the corresponding features for each image

        
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
              
	    //case 'c': // capture a photo if the user hits c
              //sprintf(buffer, "%s.%03d.png", label, frameid++);
              //cv::imwrite(buffer, frame, pars);
              //printf("Image written: %s\n", buffer);
              //break;

	    case 'N': // store the new image into the database 
	      if (training_mode) { // capture the image only in training mode 
		  // get the image label 
		  string label;
		  cout << "label for the image" << endl; 
		  cin >> label;
		  cout << endl; 

		  // store the image label 
	          images.push_back(frame);
	          auto feature = process_one_image(frame, label);
	          features.push_back(feature);
	          //feature.write_to_fstream(db_stream);
	      }
	      break;

	    case 'x': 
	      training_mode = false; 
	  }
   
   
	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

    }

	return(0);

    }
}
