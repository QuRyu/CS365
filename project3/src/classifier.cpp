/*
 * CS365 Spring 2019 Project 3 
 *
 * Qingbo Liu, Iris Lian 
 */

#include "classifier.hpp"

#include <tuple> 

#include <opencv2/ml.hpp>

#include "utilities.hpp"


using namespace std; 
using namespace cv;


// helper function to calculate standard deviation for feature data 
// Returns the stdev for each kind of feature data 
vector<double> standard_deviation(const vector<Features> &data) { 
  int N = data.size();
  int n = data[0].num_of_features();

  vector<double> means(n, 0);
  vector<double> stdev(n, 0);


    // first calculate the mean of each set of features 
  for (int i=0; i<n; i++) {
  double sum = 0; 
  for (int j=0; j<N; j++) {
    sum += (data[j])[i];
  }

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
      auto dif = db[i][j] - cmp[j];
      dist += (dif * dif) / (stdev[j] * stdev[j]);
      // dist += abs(dif) / stdev[j];
    }

    if (dist < distance) {
      index = i; 
      distance = dist;
    }
  }

  auto f = db[index];

  if (distance > 2) // new object found 
    return make_tuple(true, distance, f);
  else 
    return make_tuple(false, distance, f);
}

tuple<bool, double, Features> manhattan(const std::vector<Features> &db, const Features &cmp) {
  double distance = numeric_limits<double>::max(); 
  int index = -1; 
  int N = db.size();
  int n = cmp.num_of_features();

  for (int i=0; i<N; i++) {
    double dist = 0; 
    for (int j=0; j<n; j++) {
      dist += abs(db[i][j] - cmp[j]);
    }

    if (dist < distance) {
      index = i; 
      distance = dist;
    }
    
  }

  if (distance > 8) // new object identified 
    return make_tuple(true, distance, db[index]);
  else 
    return make_tuple(false, distance, db[index]);
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
tuple<bool, string> k_means(const std::vector<Features> &db, 
                        const Features &cmp) {
  // count how many groups there are -- the value K 
  set<string> label_set; 
  for (auto f : db) {
    label_set.insert(f.label); 
  }

  int K = label_set.size();

  // convert the set of labels into numbers 
  map<string, float> label_converter; 
  float index = 0; 
  for (auto f : db) {
    if (label_converter.find(f.label) == label_converter.end()) {
     label_converter[f.label] = index++;
    }
  }

  // prepare the data for the algorithms 
  Mat data(db.size(), cmp.num_of_features(), CV_32F),
      response(db.size(), 1, CV_32F), 
      sample(1, cmp.num_of_features(), CV_32F);
  int N = db.size();
  int n = cmp.num_of_features();

  // move data to the train model 
  for (int i=0; i<N; i++) {
    const auto &f = db[i];
    for (int j=0; j<n; j++) {
        data.at<float>(i, j) = f[j];
    }
  }

  for (int i=0; i<n; i++) 
    sample.at<float>(0, i) = cmp[i];

  for (int i=0; i<index; i++) 
    response.at<float>(i, 0) = label_converter[db[i].label]; 

  // run the algorithm to classify data 
  Mat results, neighbors;
  auto kn = cv::ml::KNearest::create();
  kn->train(data, cv::ml::ROW_SAMPLE, response);
  kn->findNearest(sample, K, results, neighbors); 

  // check if the object is the new one 
  bool new_obj = true; 
  map<float, int> neighbor_counter; 

  for (int i=0; i<K; i++) {
    auto neighbor_response = neighbors.at<float>(i);
    if (neighbor_counter.find(neighbor_response) == neighbor_counter.end())
      neighbor_counter[neighbor_response] = 1; 
    else 
      neighbor_counter[neighbor_response] += 1; 
  }

  for (auto p : neighbor_counter) { 
    if (p.second >= 3) 
      new_obj = false; 
  }

   // now find the label the response corresponds to 
  auto findResult = std::find_if( begin(label_converter), 
    end(label_converter), [&](const pair<string, int> &elem) {
    return elem.second == static_cast<int>(results.at<float>(0));
  });


  return make_tuple(new_obj, findResult->first);

}








