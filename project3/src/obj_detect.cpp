/*
  2019 Spring CS365

  Qigbo Liu, Iris Lian

  obj_detect.cpp

*/

#include <string>
#include <cstdio>
#include <cmath>
#include <regex>
#include <dirent.h>

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

  auto f = compute_features(img);
  // auto f = compute_features_conn(img);

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

Mat draw_features_connected(Mat &src, Features f){
    Mat processed = threshold(src);
    Mat inverted;
    bitwise_not(processed, inverted);
    
    Mat labelImage, stats, centroids;
    int num_labels = connectedComponentsWithStats(inverted, labelImage, stats, centroids, 8, CV_32S);

    // find the largest component
    int maxSize = 0;
    int maxIndex = 0;
    for(int label=1; label<num_labels; label++){
        if(stats.at<int>(label, CC_STAT_AREA) > maxSize){
            maxSize = stats.at<int>(label, CC_STAT_AREA);
            maxIndex = label;
        }
    }

    Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
    mask = (labelImage == maxIndex);
    Mat r(src.size(), CV_8UC1, Scalar(0));
    src.copyTo(r,mask);

    int* data = stats.ptr<int>(maxIndex);
    rectangle(r, Rect(data[0], data[1], data[2], data[3]), Scalar(0,0,255,255));
    circle( r, Point(f.centroid_x, f.centroid_y), 4, Scalar(0,0,255,255), -1, 8, 0 );

    // write to picture [predicted_label, centroid_x, centriod_y, orientation]
    putText(r, f.label, Point(f.centroid_x-70, f.centroid_y-100), FONT_HERSHEY_PLAIN, 1,  Scalar(255,255,255));
    putText(r, "centroid_y: "+to_string(f.centroid_y), Point(f.centroid_x-70, f.centroid_y-40), FONT_HERSHEY_PLAIN, 1,  Scalar(255,255,255));
    putText(r, "centroid_x: "+to_string(f.centroid_x), Point(f.centroid_x-70, f.centroid_y-70), FONT_HERSHEY_PLAIN, 1,  Scalar(255,255,255));
    putText(r, "orientation: "+to_string(f.orientation), Point(f.centroid_x-70, f.centroid_y-10), FONT_HERSHEY_PLAIN, 1,  Scalar(255,255,255));

    return r;
}

Mat draw_features_contours(Mat &src, Features f) {
    // declare Mat variables, thr, gray and src
    Mat thr, gray;
     
    // convert image to grayscale
    cvtColor( src, gray, COLOR_BGR2GRAY );
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
     
    // detect edges using canny
    Canny( gray, canny_output, 50, 200, 3 );
     
    // find contours
    findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
     
    // get the moments
    vector<Moments> mu(contours.size());
    for( int i = 0; i<contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }
     
    // get the centroid of figures.
    vector<Point2f> mc(contours.size());
    for( int i = 0; i<contours.size(); i++)
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

    // Find the biggest contour
    double maxArea = 0;
    int index = 0;
    for(int i = 0; i < contours.size(); i++){
        double h = minAreaRect( Mat(contours[i])).size.height;
        double w = minAreaRect( Mat(contours[i])).size.width;
        if((h*w) > maxArea){
            maxArea = h*w;
            index = i;
        }
    }
     
    // draw contours, bounding box, centroid
    Mat drawing(canny_output.size(), CV_8UC3, Scalar(0,0,0));
    Scalar color = Scalar(0,0,0); // B G R values
    for( int i = 0; i<contours.size(); i++ )
        drawContours(src, contours, i, color, 2, 8, hierarchy, 0, Point());
    circle( src, mc[index], 4, Scalar(0,0,255,255), -1, 8, 0 );
    // rotated rectangle
    Point2f rect_points[4]; 
    minAreaRect(Mat(contours[index])).points( rect_points );
    for( int i = 0; i < 4; i++ )
        line( src, rect_points[i], rect_points[(i+1)%4], color, 1, 8 );

    // write to picture [predicted_label, centroid_x, centriod_y, orientation]
    putText(src, f.label, Point(f.centroid_x-70, f.centroid_y-100), FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255,255));
    putText(src, "centroid_y: "+to_string(f.centroid_y), Point(f.centroid_x-70, f.centroid_y-40), FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255,255));
    putText(src, "centroid_x: "+to_string(f.centroid_x), Point(f.centroid_x-70, f.centroid_y-70), FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255,255));
    putText(src, "orientation: "+to_string(f.orientation), Point(f.centroid_x-70, f.centroid_y-10), FONT_HERSHEY_PLAIN, 1,  Scalar(0,0,255,255));

    return src;
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

  vector<Features> features;
  if (file_exists(db_stream)) { // if db file exists, first read the data
    int read;
    cout << "read from file? 1 for yes and 0 for no" << endl;
    cin >> read; 
  
    if (read)
      read_db(db_stream, features);

    cout << "features read from database" << endl;
    for (auto f : features) {
      cout << f << endl;
    }
    db_stream = fstream(DB_path(), fstream::out | ios_base::app);
  }


  if (!camera) { // we are using a list of directories 
    img_fp = argv[2];
    auto images_fp = traverse_dir(img_fp);
    vector<Mat> images;
  
    for (auto img_fp : images_fp) {
      auto img = imread(img_fp); 
      images.push_back(img);
    }

    auto dir_features = process_multiple_images(images, images_fp);
    features.insert(end(features), begin(dir_features), end(dir_features));

    for (auto f : features) {
      db_stream << f << endl;
    }
    db_stream.flush();

    while (true) {
      cout << "the path of photo to compare: " << endl; 

      string cmp_path; 
      cin << cmp_path;

      auto img = imread(cmp_path); // path needs to be complete
      auto cmp_feature = process_one_image(img, cmp_path); 

      auto img_processed = process_img(img);


      // use classifier to find which image 
      auto [_, dist, f] = euclidean(features, cmp_feature);
      cout << f.label << endl;
      // cout << label << endl;
      Mat new_img = draw_features_contours(img, cmp_feature);
      // Mat new_img = draw_features_connected(img, cmp_feature);

      // show the resultant image
      namedWindow( "Contours", WINDOW_AUTOSIZE );
      imshow( "Contours", new_img );
      waitKey(0);
    }
  } else if (camera) {
    cv::VideoCapture *capdev;
    char label[256];
    int quit = 0;
    int frameid = 0;
    char buffer[256];
    std::vector<int> pars;
    bool training_mode = true; 

    //vector<Mat> images; // the set of images used as a training set 
                //// captured from the camera 
    
    pars.push_back(5);
    
    // open the video device
    capdev = new cv::VideoCapture(1);
    if( !capdev->isOpened() ) {
      printf("Unable to open video device\n");
      return(-1);
    }
    
    strcpy(label, argv[1]);
    
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
           (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    
    printf("Expected size: %d %d\n", refS.width, refS.height);
    
    cv::namedWindow("Video", 1);
    cv::Mat frame;
    Mat new_frame;
    
    for(;!quit;) {
      *capdev >> frame; // get a new frame from the camera, treat as a stream
      
      if( frame.empty() ) {
        printf("frame is empty\n");
        break;
      }

      if (!training_mode) {
        auto frame_feature  = process_one_image(frame, "frame");
        auto [_, dist, f] = euclidean(features, frame_feature);
        //if (dist > 2) { // new object found 
          //string name; 
          //cout << "new object identified, dist " << dist << endl;
          //cout << "enter name" << endl;
          //cin >> name; 

          //frame_feature.label = name; 
          //db_stream << frame_feature << endl;
          //features.push_back(frame_feature);
        //} else {
          cout << "object " << f.label << " identified, dist " << dist << endl;
          Mat new_frame = draw_features_contours(frame, frame_feature);
          // Mat new_frame = draw_features_connected(frame, frame_feature);
        //}

      }    
    
      cv::imshow("Video", new_frame);

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

        case 'n': // store the new image into the database 
          if (training_mode) { // capture the image only in training mode
            string label;
            cout << "label for the image" << endl; 
            cin >> label;

            // store the image label 
            //images.push_back(frame);
            auto feature = process_one_image(frame, label);
            features.push_back(feature);
            db_stream << feature << endl;
            db_stream.flush();

            cout << "image with label " << label << " saved" << endl << endl;
          }
          break;

        case 'x': 
          if (training_mode) {
            training_mode = false;
            cout << "training mode off" << endl;
          }
          break;
      }
     
    }
     
    // terminate the video capture
    printf("Terminating\n");
    delete capdev;

  }
    

  return(0);
  
}
