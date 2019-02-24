/*
 *
 * CS365 19 Spring 
 *
 * Assignment 2: Content-based image retrieval 
 *
 */
#include <cstdio> 
#include <cstdlib>
#include <dirent.h>
#include <cstring> 
#include <string> 
#include <map> 

#include <opencv2/opencv.hpp> 

const int QUERY_IMAGE_FP = 1; 
const int DATABASE_FP = 2; 
const int METRIC = 3; 
const int N = 4; 

std::vector<std::string> 
traverse_dir(char *dir_fp) { 
    std::vector<std::string> images_fp; 
    DIR *dirp; 
    struct dirent *dp; 
        
    dirp = opendir(dir_fp);

    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dir_fp);
        exit(-1);
    }

    while ((dp = readdir(dirp)) != NULL) {
        if (strstr(dp->d_name, ".jpg") ||
                strstr(dp->d_name, ".png") || 
                strstr(dp->d_name, ".arw") || 
                strstr(dp->d_name, ".ppm") || 
                strstr(dp->d_name, ".tif")) {
            images_fp.push_back(dp->d_name);
        }
    }

    closedir(dirp); 

    return images_fp;
}

/**
 * Given the file path, return the image 
 */
cv::Mat read_image(const std::string &fp) {
    return cv::imread(fp);
}

std::map<std::string, double> 
compare(const std::string &query_fp,  
        const std::vector<std::string> &database, 
        std::function<double(const cv::Mat,const cv::Mat)> func) { 
    std::map<std::string, double> result; 
    auto query = read_image(query_fp);

    for (auto img_fp : database) {
        double distance; 

        auto img = read_image(img_fp);
        distance = func(query, img);
        result[img_fp] = distance;
    }

    return result; 
}

int main(int argc, char *argv[]) { 
    // command line argument format 
    // 1: query image file path 
    // 2: database file path 
    // 3: which metrics to use 
    // 4: N -- number of results to show 

    if (argc != 5) 
        std::cerr << "expects four arguments" << std::endl;

    auto images = traverse_dir(argv[DATABASE_FP]);


    return 0; 
}


