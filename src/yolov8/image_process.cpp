//
// Created by kaylor on 3/4/24.
//

#include "image_process.h"

ImageProcess::ImageProcess(int width, int height, int target_size) {
  scale_ = static_cast<double>(target_size) / std::max(height, width);
  padding_x_ = target_size - static_cast<int>(width * scale_);
  padding_y_ = target_size - static_cast<int>(height * scale_);
  new_size_ = cv::Size(static_cast<int>(width * scale_), static_cast<int>(height * scale_));
  target_size_ = target_size;
}

std::shared_ptr<cv::Mat> ImageProcess::Convert(const cv::Mat &src) {
  cv::Mat resize_img;
  cv::resize(src, resize_img, new_size_);
  auto square_img = std::make_shared<cv::Mat>(target_size_, target_size_, src.type(), cv::Scalar(0, 0, 0));
  cv::Point position(padding_x_ / 2, padding_y_ / 2);
  resize_img.copyTo((*square_img)(cv::Rect(position.x, position.y, resize_img.cols, resize_img.rows)));
  return std::move(square_img);
}