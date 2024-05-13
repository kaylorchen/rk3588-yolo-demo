//
// Created by kaylor on 3/9/24.
//

#pragma once
#include "opencv2/opencv.hpp"

class Camera {
 public:
  Camera(uint16_t index, cv::Size size, double framerate);
  ~Camera();
  std::unique_ptr<cv::Mat> GetNextFrame();

 private:
  cv::Size size_;
  cv::VideoCapture capture_;
};
