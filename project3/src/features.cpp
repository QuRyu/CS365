/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

    features.cpp

*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "imgproc.hpp"
#include "features.hpp"

using namespace cv;
using namespace std;

void Features::write_to_fstream(fstream &stream) {
    stream << " " << endl;
}

std::vector<std::vector<Point>> compute_contours(const Mat &src){
	Mat src_gray;
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);

	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	/// Convert image to gray and blur it
	cvtColor( src, src_gray, COLOR_BGR2GRAY );
	blur( src_gray, src_gray, Size(3,3) );

	Mat canny_output;

	/// Detect edges using canny
	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	/// Find contours
	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	return contours;
}

std::vector<Moments>
compute_mulitiple_moments(const Mat &src){
	// Get the contours
	std::vector<std::vector<Point>> contours;
	contours = compute_contours(src);

	/// Get the moments
	std::vector<Moments> mu(contours.size() );
	for (int i = 0; i < contours.size(); i++)
	    mu[i] = moments( contours[i], false ); 

	return mu;
}

std::vector<double *> compute_multiple_HuMoments(const Mat &src){
	std::vector<Moments> moments = compute_mulitiple_moments(src);

	std::vector<double *> huMoments;
	for( int i = 0; i < moments.size(); i++ ){ 
		double *hu = new double[7];
		HuMoments(moments[i], hu); 
		// Log scale hu moments
		for(int j = 0; j < 7; j++){
			hu[j] = -1 * copysign(1.0, hu[j]) * log10(abs(hu[j]));  
		}
		huMoments.push_back(hu);
	}
	return huMoments;
}

void compute_single_HuMoments(const Mat &src, double *hu) {
	Mat src_gray, src_thresh;
	// convert to grayscale
	cvtColor( src, src_gray, COLOR_BGR2GRAY );
	// threshold image
	threshold( src_gray, src_thresh, 128, 255, THRESH_BINARY);
	// calculate moments
	Moments mom = moments(src_thresh, false);
	// calculate hu moments
	HuMoments(mom, hu);
	// Log scale hu moments
	for(int i = 0; i < 7; i++){
		hu[i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));  
	}
}

double compute_entropy(const Mat &src) {
	Mat src_gray;
	if(src.channels()==3) cvtColor(src, src_gray, COLOR_BGR2GRAY);
	// establish the number of bins
	int histSize = 256;
	// set the ranges (for B,G,R)
	float range[] = {0, 256};
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	// compute the histograms
	Mat hist;
	calcHist( &src_gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	hist /= src_gray.total();
	hist += 1e-4; // prevent 0

	Mat logP;
	cv::log(hist, logP);

	double entropy = -1*sum(hist.mul(logP)).val[0];
	return entropy;
}

std::vector<double> compute_multiple_entropy(const Mat &src){
	// Get the contours
	std::vector<std::vector<Point>> contours;
	std::vector<double> es;
	contours = compute_contours(src);

	for(int i = 0; i < contours.size(); i++){
		// Get bounding box for contour
        Rect roi = boundingRect(contours[i]);

        // Create a mask for each contour to mask out that region from image.
        Mat mask = Mat::zeros(src.size(), CV_8UC1);
        drawContours(mask, contours, i, Scalar(255), CV_FILLED);

        // At this point, mask has value of 255 for pixels within the contour and value of 0 for those not in contour.

        // Extract region using mask for region
        Mat contourRegion;
        Mat imageROI;
        src.copyTo(imageROI, mask);
        contourRegion = imageROI(roi);

        es.push_back(compute_entropy(contourRegion));
	}

	return es;
}

// find bounding boxes and calculate height-width ratios
std::vector<double> compute_HWRatios(const Mat &src){
	// Get the contours
	std::vector<std::vector<Point>> contours;
	contours = compute_contours(src);

	std::vector<double> ratios;
	for(int i = 0; i < contours.size(); i++){
		double h = minAreaRect( Mat(contours[i])).size.height;
		double w = minAreaRect( Mat(contours[i])).size.width;
		ratios.push_back(h/w);
	}
	return ratios;
}

// find the percentage of the object in its bounding box
std::vector<double> compute_percentArea(const Mat &src){
	// Get the contours
	std::vector<std::vector<Point>> contours;
	contours = compute_contours(src);

	std::vector<double> percent;
	for(int i = 0; i < contours.size(); i++){
		double h = minAreaRect( Mat(contours[i])).size.height;
		double w = minAreaRect( Mat(contours[i])).size.width;
		percent.push_back(contourArea(contours[i])/(h*w));
	}
	return percent;
}

void compute_features(const Mat &src, double *f){

	// compute huMoments
	std::vector<double *> hu;
	hu = compute_multiple_HuMoments(src);
	for(int i=0; i<hu.size(); i++){
		for(int j=0; j<7; j++){
			printf("hu[%d][%d] %f\n", i, j, hu[i][j]);
		}
	}

	// compute entropy
	std::vector<double> es;
	es = compute_multiple_entropy(src);
	for(int i=0; i<es.size(); i++){
		printf("entropy[%d] %f\n", i, es[i]);
	}

	// compute h/w
	std::vector<double> ratios;
	ratios = compute_HWRatios(src);
	for(int i=0; i<ratios.size(); i++){
		printf("ratios[%d] %f\n", i, ratios[i]);
	}

	// compute % in bounding box
	std::vector<double> ps;
	ps = compute_percentArea(src);
	for(int i=0; i<ps.size(); i++){
		printf("percent[%d] %f\n", i, ps[i]);
	}


	// compute HOG
	// Mat grad, angleOfs;

	// HOGDescriptor hog;
 //    hog.winSize = src.size()/8*8;
 //    Mat gray;
 //    vector< float > descriptors;

 //    cout << (src.size()/8*8) << endl;

 //    Rect r = Rect(( src.cols - (src.size()/8*8).width ) / 2,
 //                  ( src.rows - (src.size()/8*8).height ) / 2,
 //                  (src.size()/8*8).width,
 //                  (src.size()/8*8).height);
 //    cvtColor( src(r), gray, COLOR_BGR2GRAY );
 //    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
 //    grad =  Mat( descriptors ).clone() ;

	// cout << "grad " << grad.size() << endl;
	// cout << "grad[0] " << descriptors[0] << endl;
	// cout << "grad[1] " << descriptors[1] << endl;


}

