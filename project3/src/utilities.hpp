#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <string> 

// convert the numerical type in opencv to 
// human-readable string
std::string type2str(int type); 

// traverse a directory and return the 
// path of all images
std::vector<std::string> 
traverse_dir(const std::string &dir_fp);

#endif 
