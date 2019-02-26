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
#include <set>
#include <functional>

#include <opencv2/opencv.hpp> 

#include "metrics.hpp"
#include "utilities.hpp"


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
            auto fp = fp_prefix + std::string(dp->d_name);
            images_fp.push_back(fp);
        }
    }

    closedir(dirp);

    return images_fp;
}

std::map<std::string, double> 
compare(const cv::Mat &query,  
        const std::vector<std::string> &database, 
        std::function<double(const cv::Mat,const cv::Mat)> &func) { 
    std::map<std::string, double> result; 

    std::cout << "here?" << std::endl;
    for (auto img_fp : database) {
        double distance; 

        auto img = cv::imread(img_fp);
        if (img.data == NULL) {
            std::cerr << "Unable to read images" << std::endl; 
            exit(-1);
        }

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

    using namespace std; 

    if (argc != 5) 
        std::cerr << "expects four arguments" << std::endl;

    auto query_img = cv::imread(argv[QUERY_IMAGE_FP]);
    auto images_fp = traverse_dir(argv[DATABASE_FP]);
    auto N = std::atoi(argv[ARG_N]);
    auto metrics = std::atoi(argv[METRIC]);


    // read and compare the database images against query image  
    function<double(const cv::Mat, const cv::Mat)> const_metrics = baseline_hist_metric; 
    auto database = compare(query_img, images_fp, const_metrics);

    // sort the distances 
    typedef function<bool(pair<string, int>, pair<string, int>)> Comparator; 

    Comparator comp = [](pair<string, int> elem1, pair<string, int> elem2) { 
        return elem1.second < elem2.second; 
    };

    set<pair<string, int>, Comparator> 
            database_sorted( database.begin(), database.end(), comp);

    // print images in ascending order of distances 
    //auto iter = database_sorted.begin();
    //for (int i=0; i<N; i++) {
        //cout << "Image " << i << endl; 
        //cout << "Address: " << iter->first << endl; 
        //cout << "distance: " << iter->second << endl;
        //++iter;
    //}

    return 0; 
}


