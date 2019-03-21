#include <string>
#include <dirent.h>

#include <opencv2/opencv.hpp> 

#include "utilities.hpp"


// convert the numerical type in opencv to 
// human-readable string
std::string type2str(int type) {
  using namespace cv;
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


// traverse a directory and return the 
// path of all images
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

bool file_exists(const std::fstream &stream) {
    return stream.good();
}

