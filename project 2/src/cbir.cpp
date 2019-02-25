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
const int ARG_N = 4; 

std::vector<std::string> 
traverse_dir(const std::string &dir_fp) { 
    std::string fp_prefix = dir_fp + "/";
    std::vector<std::string> images_fp; 
    DIR *dirp; 
    struct dirent *dp; 

    dirp = opendir(dir_fp.c_str());
    if (dirp == NULL) { 
        std::cerr << "Cannot open directory " << dir_fp << std::endl;
        exit(-1);
    }

    while ((dp = readdir(dirp)) != NULL) {
        if (dp->d_type == DT_DIR) {
           // avoid recursive traversal 
           if ((strcmp(dp->d_name, ".") != 0) 
                   && (strcmp(dp->d_name, "..") != 0)) {  
                auto subdir_fp = fp_prefix + std::string(dp->d_name);
                auto images_subdir = traverse_dir(subdir_fp);
                images_fp.insert(images_fp.end(), 
                        images_subdir.begin(), images_subdir.end());
            }
        }
        else if (strstr(dp->d_name, ".jpg") ||
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

    auto query_img = read_image(argv[QUERY_IMAGE_FP]);
    auto images_fp = traverse_dir(argv[DATABASE_FP]);
    auto N = std::atoi(argv[ARG_N]);
    auto metrics = std::atoi(argv[METRIC]);

    for (auto fp : images_fp) {
        std::cout << fp << std::endl;
    }

    return 0; 
}


