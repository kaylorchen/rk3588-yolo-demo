//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "string"
#include "opencv2/opencv.hpp"
#include "image_process.h"
class VideoFile {
 public:
  VideoFile(const std::string &&filename);
  ~VideoFile();
  void Display(const float framerate = 25.0, const int target_size = 640);
  std::shared_ptr<cv::Mat> GetNextFrame(const int target_size = 640);
 private:
  std::string filename_;
  cv::VideoCapture* capture_{nullptr};
};
