/*
 * CS365 Spring 2019 Project 3 
 *
 * Qingbo Liu, Iris Lian 
 */

#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP 

#include "features.hpp"

/*
 * Euclidean classifier 
 * returns a tuple indicating 
 * 1. bool: if the object is a new object  
 * 2. double: the distance 
 * 3. Features: the feature in database that is closest to the one we are matching 
*/
std::tuple<bool, double, Features> euclidean(const std::vector<Features> &db, const Features &cmp);


/*
 * Mahattan distance classifier  
 *
 * returns a tuple indicating 
 * 1. bool: if the object is a new object  
 * 2. double: the distance 
 * 3. Features: the feature in database that is closest to the one we are matching 
 */
std::tuple<bool, double, Features> mahattan(const std::vector<Features> &db, const Features &cmp);

/**
 * K-means classifier 
 * Returns a tuple indicating 
 * 1. bool: if the object is a new object 
 * 2. Features: the feature in database that is closet to the one we are
 * matching 
 */
std::tuple<bool, std::string> k_means(const std::vector<Features> &db, const Features &cmp, int K);



#endif // CLASSIFIER_HPP
