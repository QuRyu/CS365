/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

    obj_detect.cpp

*/

#include <string>
#include <cstdio>
#include <cmath>
#include <regex>

#include <opencv2/opencv.hpp>

#include "imgproc.hpp"
#include "utilities.hpp"
#include "features.hpp"
#include "classifier.hpp"

using namespace std;
using namespace cv;


// return the DB path 
std::string DB_path() {
#ifdef macOS_Qingbo 
    string path = "/Users/HereWegoR/Documents/CS/CS365/project3/data/db.txt";
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

    //Features f;
    auto f = compute_features(img);

    auto label_ex = extract_label(label);

    // wrap all features in the Feature struct 
    f.label = label_ex; 

    return f;
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

void read_db(fstream &stream, vector<Features> &v) {
    Features f; 
    for (string line; getline(stream, line); ) {
	istringstream iss(line);
	iss >> f; 
	v.push_back(f);
    }
}

void draw_features(const Mat &src, Features f) {
    Mat src_gray;
    int thresh = 100;
    int max_thresh = 255;
    RNG rng(12345);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    /// Convert image to gray and blur it
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );

    Mat canny_output;

    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    findContours( canny_output, contours, hierarchy,
        RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( 255, 255, 255 );
        drawContours( dst, contours, idx, color, FILLED, 8, hierarchy );
    }
    namedWindow( "Components", 1 );
    imshow( "Components", dst );
    waitKey(0);


    // Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
    // Scalar color( rand()&255, rand()&255, rand()&255 );
    // drawContours(dst, f.contours, -1, color, FILLED, 8);
    // imshow("img", dst);
    // waitKey(0);
}

int main(int argc, char *argv[]) {
    /* Argument Format: source (path)
     * source: 0 for camera and 1 for directories 
     * path: if we use the directories, provide the path 
     */

    bool camera;
    string img_fp; 
    fstream db_stream(DB_path(), fstream::in);

    if (argc < 1) {
	cerr << "no argument provided" << endl;
	exit(-1);
    } else {
	int source = atoi(argv[1]);
	camera = source == 0 ? true : false; 
    }

    //vector<Features> features;
    //if (file_exists(db_stream)) { // if db file exists, first read the data
	//int read;
	//cout << "read from file? 1 for yes and 0 for no" << endl;
	//cin >> read; 
	
	//if (read)
	    //read_db(db_stream, features);

	//cout << "features read from database" << endl;
	//for (auto f : features) {
	    //cout << f << endl;
	//}

	//db_stream = fstream(DB_path(), fstream::out | ios_base::app);
    //}


    if (!camera) { // we are using a list of directories 
        img_fp = argv[2];
	string cmp_fp = argv[3]; 

    	auto images_fp = traverse_dir(img_fp);
	auto images_cmp_fp = traverse_dir(cmp_fp);
    	vector<Mat> images;
	
    	for (auto img_fp : images_fp) {
    	    auto img = imread(img_fp); 
    	    images.push_back(img);
    	}

	auto features = process_multiple_images(images, images_fp);
	//features.insert(end(features), begin(dir_features), end(dir_features));

	//for (auto f : features) {
	    //db_stream << f << endl;
	//}

	//while (true) {
                //cout << "the path of photo to compare: " << endl; 

            ////string cmp_path; 
            ////cin >> cmp_path; 
	    //// use fixed path for now 
	    //string cmp_path("/Users/HereWegoR/Documents/CS/CS365/project3/data/training/shovel.002.png");
	    //cout << "cmp_path " << cmp_path << endl;

                //auto img = imread(cmp_path); // path needs to be complete
                //auto cmp_feature = process_one_image(img, cmp_path); 

	    //auto img_processed = process_img(img);


	    //// use classifier to find which image 
	    //auto [_, dist, f] = euclidean(features, cmp_feature);
	    //// TODO: find the value K from the list of directories we read 
	    ////auto [_, label] = k_means(features, cmp_feature, 9);
	    //cout << f.label << endl;
            ////draw_features(img, cmp_feature);
            //// }

	    //waitKey(0);
            //}
	for (auto cmp : images_cmp_fp) { 
	    auto img_cmp = imread(cmp);
	    auto f = process_one_image(img_cmp, cmp); 
	    auto [_, dist, closest] = euclidean(features, f);
	    
	    cout << "true label " << f.label << ", classified label " << closest.label << endl;
	}
    } else if (camera) {
        //cv::VideoCapture *capdev;
        //char label[256];
        //int quit = 0;
        //int frameid = 0;
        //char buffer[256];
        //std::vector<int> pars;
	//bool training_mode = true; 

            ////vector<Mat> images; // the set of images used as a training set 
                                ////// captured from the camera 
        
        //pars.push_back(5);
        
        //// open the video device
        //capdev = new cv::VideoCapture(1);
        //if( !capdev->isOpened() ) {
                //printf("Unable to open video device\n");
                //return(-1);
        //}
        
        //strcpy(label, argv[1]);
        
        //cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       //(int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        
        //printf("Expected size: %d %d\n", refS.width, refS.height);
        
        //cv::namedWindow("Video", 1); // identifies a window?
        //cv::Mat frame;
        
        
            //for(;!quit;) {
                //*capdev >> frame; // get a new frame from the camera, treat as a stream
            
	    //if( frame.empty() ) {
		//printf("frame is empty\n");
		//break;
	    //}
              
            //cv::imshow("Video", frame);
            
	    //if (!training_mode) {
		//auto frame_feature  = process_one_image(frame, "frame");
		//auto [_, dist, f] = euclidean(features, frame_feature);
		//cout << "object " << f.label << " identified" << endl;
	    //}

            //int key = cv::waitKey(10);

            
	    //switch(key) {
		//case 'q':
		    //quit = 1;
		    //break;
		      
		//case 'c': // capture a photo if the user hits c
		    //sprintf(buffer, "%s.%03d.png", label, frameid++);
		    //cv::imwrite(buffer, frame, pars);
		    //printf("Image written: %s\n", buffer);
		    //break;

		//case 'n': // store the new image into the database 
		    //if (training_mode) { // capture the image only in training 
					//// mode
			//string label;
			//cout << "label for the image" << endl; 
			//cin >> label;

			//// store the image label 
			////images.push_back(frame);
			//auto feature = process_one_image(frame, label);
			//features.push_back(feature);
			//db_stream << feature << endl;
			//db_stream.flush();

			//cout << "image with label " << label << " saved" << endl << endl;
		    //}
		    //break;

		//case 'x': 
		    //if (training_mode) {
			//training_mode = false;
			//cout << "training mode off" << endl;
		    //}
		    //break;
	    //}
       
	//}
       
        //// terminate the video capture
        //printf("Terminating\n");
        //delete capdev;

    }
        

    return(0);
    
}
