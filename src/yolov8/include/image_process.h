//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "BYTETracker.h"
#include "mutex"
#include "opencv2/opencv.hpp"
#include "postprocess.h"

class ImageProcess {
 public:
  ImageProcess(int width, int height, int target_size, bool is_track = false,
               int framerate = 30);
  std::unique_ptr<cv::Mat> Convert(const cv::Mat &src);
  const letterbox_t &get_letter_box();
  void ImagePostProcess(cv::Mat &image, object_detect_result_list &od_results);

 private:
  double scale_;
  int padding_x_;
  int padding_y_;
  cv::Size new_size_;
  int target_size_;
  letterbox_t letterbox_;
  bool is_track_;
  std::unique_ptr<BYTETracker> tracker_;
  std::mutex tracker_mutex_;
  void ProcessDetectionImage(cv::Mat &image,
                             object_detect_result_list &od_results) const;
  void ProcessTrackImage(cv::Mat &image, object_detect_result_list &od_results);
  void ProcessPoseImage(cv::Mat &image,
                        object_detect_result_list &od_results) const;
  void ProcessOBBImage(cv::Mat &image,
                       const object_detect_result_list &od_results) const;
};
