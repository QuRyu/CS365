/*
 * cbir.cpp
 *
 * CS365 19 Spring 
 *
 * Iris Lian and Qingbo Liu
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
#include <tuple>

#include <opencv2/opencv.hpp> 

#include "metrics.hpp"
#include "utilities.hpp"

using namespace std;
using namespace cv;

typedef function<bool(pair<string, double>, pair<string, double>)> Comparator; 

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
compare(const vector<Mat> &query,  
        const std::vector<std::string> &database, 
        std::function<double(const vector<Mat>,const cv::Mat)> &func) { 
    std::map<std::string, double> result; 

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

auto which_metrics(int i, Mat query_img) {
    std::function<double(const vector<Mat>, const cv::Mat)> func;
    Comparator comp = 
        [](pair<string, double> elem1, pair<string, double> elem2) -> bool {
                       return elem1.second > elem2.second; 
        };
    vector<Mat> query_list;

    switch(i) {
        case 0: 
            func = ssd_metric;
            comp = [](pair<string, double> elem1, pair<string, double> elem2) { 
                       return elem1.second < elem2.second; 
                   };
            query_list.push_back(query_img);
            break;
        case 1: 
            func = baseline_hist_metric;
            query_list.push_back(calc_histogram(query_img, 0)); // whole
            break; 
        case 2: 
            {
            func = multi_hist_metric;
            int query_width = query_img.cols, query_height = query_img.rows;
            Mat query_left(query_img, Rect(0, 0, query_width/5, query_height)),
                query_right(query_img, Rect(query_width*4/5, 0, 
                            query_width/5, query_height)), 
                query_middle(query_img, Rect(query_width/5, 0,
                            query_width*3/5, query_height));
            query_list.push_back(calc_histogram(query_left, 0)); // left
            query_list.push_back(calc_histogram(query_right, 0)); // right
            query_list.push_back(calc_histogram(query_middle, 0)); // middle
            query_list.push_back(calc_histogram(query_img, 0)); // whole
            }
            break; 
        case 3: 
            func = texture_color_metric;
            query_list = calc_textColorHists(query_img);
            break;
        case 4:
            {
            func = custom_distance_metric;
            query_list.push_back(calc_histogram(query_img, 0)); // whole
            int query_row_mid = query_img.rows/2 - 1; // midpoint-1 is the central index
            int query_col_mid = query_img.cols/2 - 1;
            Mat query_middle(query_img, cv::Rect(query_col_mid-50, query_row_mid-50,
                    100, 100));
            vector<Mat> tempHists = calc_textColorHists(query_middle);
            for(int i = 0; i < tempHists.size(); i++){
                query_list.push_back(tempHists[i]);
            }
            }
            break;
        case 5:
            func = other_matching;
            comp = [](pair<string, double> elem1, pair<string, double> elem2) { 
                       return elem1.second < elem2.second; 
                   };
            query_list.push_back(other_matching_helper(query_img));
            break; 
        default: 
            std::cerr << "unexpected metrics argument " << i << std::endl;
            exit(-1);
    }

    return std::make_tuple(func, comp, query_list); 
}

int main(int argc, char *argv[]) { 
    // command line argument format 
    // 1: query image file path 
    // 2: database file path 
    // 3: which metrics to use 
    // 4: N -- number of results to show 

    if (argc != 5) 
        std::cerr << "expects four arguments" << std::endl;

    auto query_img = cv::imread(argv[QUERY_IMAGE_FP]);
    auto images_fp = traverse_dir(argv[DATABASE_FP]);
    auto metrics = std::atoi(argv[METRIC]);
    auto N = std::atoi(argv[ARG_N]);

    if (query_img.data == NULL) {
        std::cerr << "Unable to read the query image" << std::endl; 
        exit(-1);
    }

    auto tup = which_metrics(metrics, query_img);
    auto metrics_pick = get<0>(tup);
    auto comp = get<1>(tup);
    auto query = get<2>(tup);

    // read and compare the database images against query image  
    auto database = compare(query, images_fp, metrics_pick);

    // sort the distances 
    set<pair<string, double>, Comparator> 
            database_sorted( database.begin(), database.end(), comp);

    // print images in ascending order of distances 
    auto iter = database_sorted.begin();
    for (int i=0; i<N; i++) {
        cout << "Image " << i << endl;
        cout << "Address: " << iter->first << endl; 
        cout << "distance: " << iter->second << endl;
        cout << endl;
        ++iter;
    }

    return 0; 
}


