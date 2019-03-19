/*
 * CS365 Spring 2019 Project 3 
 *
 * Qingbo Liu, Iris Lian 
 */

#include "classifier.hpp"

#include <tuple> 

using namespace std; 
using namespace cv;


// helper function to calculate standard deviation for feature data 
// Returns the stdev for each kind of feature data 
vector<double> standard_deviation(const vector<Features> &data) { 
    int N = data.size();
    int n = data[0].num_of_features();

    vector<double> means;
    vector<double> stdev;

    // first calculate the mean of each set of features 
    for (int i=0; i<n; i++) {
	double sum = 0; 
	for (int j=0; j<N; j++) 
	    sum += data[j][i];

	means[i] = sum / N;
    }

    // then the standard deviation 
    for (int i=0; i<n; i++) {
	double sum = 0; 
	double mean = means[i];
	for (int j=0; j<N; j++) 
	    sum += pow((data[j][i]-means[i]), 2); 

	stdev[i] = sqrt(sum / (N-1));
    }

    return stdev;
}

/*
 * Euclidean classifier 
 * returns a tuple indicating 
 * 1. if the object is a new object  
 * 2. the distance 
 * 3. the feature in database that is closest to the one we are matching 
*/
tuple<bool, double, Features> euclidean(const std::vector<Features> &db, const Features &cmp) {
    double distance = numeric_limits<double>::max();
    int index = -1; 
    int N = db.size();
    int n = cmp.num_of_features();

    // find the standard deviation for each kind of features  
    const auto stdev = standard_deviation(db);

    // calculate the euclidean distance for image in database and 
    // find the closest one to cmp 
    for (int i=0; i<N; i++) {
	double dist = 0; 
	for (int j=0; j<n; j++) {
	    dist += abs(db[i][j] - cmp[j]) / stdev[j];
	}

	if (dist < distance) {
	    index = i; 
	    distance = dist;
	}
    }

    // for now, don't implement new object detection feature 
    return make_tuple(true, distance, db[index]);

}



/**
 * K-means classifier 
 * Parameters: 
 *  1. db: set of training images  
 *  2. cmp: new data to compare 
 *  3. K: number of labels 
 *
 * Returns a tuple indicating 
 * 1. bool: if the object is a new object 
 * 2. Features: the feature in database that is closet to the one we are
 * matching 
 */
tuple<bool, Features> k_means(
	const std::vector<Features> &db, const Features &cmp, int K) {
    // prepare the data for the algorithms 
    Mat data(db.size() + 1, cmp.num_of_features(), CV_64FC1); 
    int N = db.size();
    int n = cmp.num_of_features();

    for (int i=0; i<N; i++) {
	const auto &f = db[i];
	for (int j=0; j<n; j++) {
	    data.at<double>(i, j) = f[j];
	}
    }

    // run the algorithm to classify data 
    Mat labels, centers; 
    TermCriteria criteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 1);
    auto flag = KMEANS_RANDOM_CENTERS;
    auto compactness = kmeans(data, K, labels, criteria, 10, flag, centers);

    // find which cluster cmp falls into 
    cout << "label matrix size " << labels.size() << endl; 
    cout << "centers matrix size " << centers.size() << endl; 

}










