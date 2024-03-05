//
// Created by kaylor on 3/4/24.
//

#include "image_process.h"
#include "kaylordut/log/logger.h"

ImageProcess::ImageProcess(int width, int height, int target_size) {
  scale_ = static_cast<double>(target_size) / std::max(height, width);
  padding_x_ = target_size - static_cast<int>(width * scale_);
  padding_y_ = target_size - static_cast<int>(height * scale_);
  new_size_ = cv::Size(static_cast<int>(width * scale_), static_cast<int>(height * scale_));
  target_size_ = target_size;
  letterbox_.scale = scale_;
  letterbox_.x_pad = padding_x_ / 2;
  letterbox_.y_pad = padding_y_ / 2;
}

std::shared_ptr<cv::Mat> ImageProcess::Convert(const cv::Mat &src) {
  if (&src == nullptr) { return nullptr; }
  cv::Mat resize_img;
  cv::resize(src, resize_img, new_size_);
  auto square_img = std::make_shared<cv::Mat>(target_size_, target_size_, src.type(), cv::Scalar(0, 0, 0));
  cv::Point position(padding_x_ / 2, padding_y_ / 2);
  resize_img.copyTo((*square_img)(cv::Rect(position.x, position.y, resize_img.cols, resize_img.rows)));
  return std::move(square_img);
}

const letterbox_t &ImageProcess::get_letter_box() { return letterbox_; }

void ImageProcess::ImagePostProcess(cv::Mat &image, object_detect_result_list &od_results) {
  for (int i = 0; i < od_results.count; ++i) {
    object_detect_result *detect_result = &(od_results.results[i]);
//    if (strcmp(coco_cls_to_name(detect_result->cls_id), "person") == 0){ continue;}
    KAYLORDUT_LOG_INFO("{} @ ({} {} {} {}) {}",
                       coco_cls_to_name(detect_result->cls_id),
                       detect_result->box.left,
                       detect_result->box.top,
                       detect_result->box.right,
                       detect_result->box.bottom,
                       detect_result->prop);
    cv::rectangle(image,
                  cv::Point(detect_result->box.left, detect_result->box.top),
                  cv::Point(detect_result->box.right, detect_result->box.bottom),
                  cv::Scalar(0, 0, 255), 2);
    char text[256];
    sprintf(text, "%s %.1f%%", coco_cls_to_name(detect_result->cls_id),
            detect_result->prop * 100);
    cv::putText(image,
                text,
                cv::Point(detect_result->box.left, detect_result->box.top + 20),
                cv::FONT_HERSHEY_COMPLEX,
                1,
                cv::Scalar(255, 0, 0),
                2,
                cv::LINE_8);
  }
}