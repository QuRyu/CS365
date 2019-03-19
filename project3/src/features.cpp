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

vector<vector<Point>> compute_contours(const Mat &src){
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
	findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	// Only return the biggest contour
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

	vector<vector<Point>> obj_contour;
	obj_contour.push_back(contours[index]);

	return obj_contour;
}

vector<Moments>
compute_multiple_moments(const Mat &src){
	// Get the contours
	vector<vector<Point>> contours;
	contours = compute_contours(src);

	/// Get the moments
	vector<Moments> mu(contours.size() );
	for (int i = 0; i < contours.size(); i++)
	    mu[i] = moments( contours[i], false ); 

	return mu;
}

vector<double *> compute_multiple_HuMoments(const Mat &src){
	vector<Moments> moments = compute_multiple_moments(src);

	vector<double *> huMoments;
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
	log(hist, logP);

	double entropy = -1*sum(hist.mul(logP)).val[0];
	return entropy;
}

vector<double> compute_multiple_entropy(const Mat &src, vector<vector<Point>> contours){
	vector<double> es;

	for(int i = 0; i < contours.size(); i++){
		// Get bounding box for contour
        Rect roi = boundingRect(contours[i]);

        // Create a mask for each contour to mask out that region from image.
        Mat mask = Mat::zeros(src.size(), CV_8UC1);
        drawContours(mask, contours, i, Scalar(255), FILLED);

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
vector<double> compute_HWRatios(vector<vector<Point>> contours){
	vector<double> ratios;
	for(int i = 0; i < contours.size(); i++){
		double h = minAreaRect( Mat(contours[i])).size.height;
		double w = minAreaRect( Mat(contours[i])).size.width;
		ratios.push_back(h/w);
	}
	return ratios;
}

// find the percentage of the object in its bounding box
vector<double> compute_percentArea(vector<vector<Point>> contours){
	vector<double> percent;
	for(int i = 0; i < contours.size(); i++){
		double h = minAreaRect( Mat(contours[i])).size.height;
		double w = minAreaRect( Mat(contours[i])).size.width;
		percent.push_back(contourArea(contours[i])/(h*w));
	}
	return percent;
}

vector<double> compute_centriods_ort(const Mat &src){
	vector<double> res;

	Moments m = compute_multiple_moments(src)[0];
	double centriod_x = m.m10/m.m00;
	double centriod_y = m.m01/m.m00;
	res.push_back(centriod_x);
	res.push_back(centriod_y);

	double ort = atan2(2*m.mu11,(m.mu20-m.mu02));
	res.push_back(ort);

	return res;
}

Features compute_features(const Mat &src){
	// Get the contours
	vector<vector<Point>> contours;
	contours = compute_contours(src);

	// compute huMoments
	vector<double *> hu;
	hu = compute_multiple_HuMoments(src);
	// for(int i=0; i<hu.size(); i++){
	// 	for(int j=0; j<7; j++){
	// 		printf("hu[%d][%d] %f\n", i, j, hu[i][j]);
	// 	}
	// }

	// compute entropy
	vector<double> es;
	es = compute_multiple_entropy(src, contours);
	// for(int i=0; i<es.size(); i++){
	// 	printf("entropy[%d] %f\n", i, es[i]);
	// }

	// compute h/w
	vector<double> ratios;
	ratios = compute_HWRatios(contours);
	// for(int i=0; i<ratios.size(); i++){
	// 	printf("ratios[%d] %f\n", i, ratios[i]);
	// }

	// compute % in bounding box
	vector<double> ps;
	ps = compute_percentArea(contours);
	// for(int i=0; i<ps.size(); i++){
	// 	printf("percent[%d] %f\n", i, ps[i]);
	// }

	vector<double> centriods_ort;
	centriods_ort = compute_centriods_ort(src);
	// for(int i=0; i<centriods_ort.size(); i++){
	// 	printf("centriods_ort[%d] %f\n", i, centriods_ort[i]);
	// }


	vector<double> features;
	for(int i=0; i<7; i++){
		features.push_back(hu[0][i]);
	}
	features.push_back(es[0]);
	features.push_back(ratios[0]);
	features.push_back(ps[0]);
	
	Features f(features, contours, centriods_ort[0], centriods_ort[1], centriods_ort[2]);

	return f;
}

