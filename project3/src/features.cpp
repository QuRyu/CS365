/*
    2019 Spring CS365

    Qigbo Liu, Iris Lian

    features.cpp

*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "HOG.hpp"
#include "features.hpp"

using namespace cv;
using namespace std;

void Features::write_to_fstream(fstream &stream) {
    stream << " " << endl;
}



std::vector<Moments>
compute_mulitiple_moments(const Mat &src) {
	Mat src_gray;
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);

	/// Convert image to gray and blur it
	cvtColor( src, src_gray, COLOR_BGR2GRAY );
	blur( src_gray, src_gray, Size(3,3) );

	Mat canny_output;
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	/// Find contours
	findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Get the moments
	std::vector<Moments> mu(contours.size() );
	for (int i = 0; i < contours.size(); i++)
	    mu[i] = moments( contours[i], false ); 

	return mu;
}

void compute_multiple_HuMoments(const Mat &src, std::vector<double *> huMoments) {
	std::vector<Moments> moments = compute_mulitiple_moments(src);
	for (int i = 0; i < moments.size(); i++)
	    HuMoments(moments[i], huMoments[i]); 
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

void compute_features(const Mat &src, double *f) {
	/// for single object (for now)
	// compute hu
	double hu[7];
	compute_single_HuMoments(src, hu);
	for(int i=0; i<7; i++){
		printf("hu[%d] %f\n", i, hu[i]);
	}
	// compute entropy
	double e = compute_entropy(src);
	cout << "entropy feature: " << e << endl;

	cout << src.size() << endl;
	// // compute HOG
	// Mat mag, ang;
	// computeMagAngle(src, mag, ang);
	// Mat wHogFeature;
	// computeHOG(mag, ang, wHogFeature, src.size(), true);
	// cout << endl << endl;
	// cout << "Magnitude: " << mag << endl;
	// cout << "Angle: " << ang << endl;
	// cout << "Weighted HOG feature: " << wHogFeature << endl;

}

