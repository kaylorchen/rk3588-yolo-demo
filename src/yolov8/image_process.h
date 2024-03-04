//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "opencv2/opencv.hpp"

class ImageProcess {
 public:
  ImageProcess(int width, int height, int target_size);
  std::shared_ptr<cv::Mat> Convert(const cv::Mat &src);
 private:
  double scale_;
  int padding_x_;
  int padding_y_;
  cv::Size new_size_;
  int target_size_;
};
