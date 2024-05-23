//
// Created by kaylor on 3/4/24.
//

#include "videofile.h"

#include "image_process.h"
#include "kaylordut/log/logger.h"

VideoFile::VideoFile(const std::string &&filename) : filename_(filename) {
  capture_ = new cv::VideoCapture(filename_);
  if (!capture_->isOpened()) {
    KAYLORDUT_LOG_ERROR("Error opening video file");
    exit(EXIT_FAILURE);
  }
}

VideoFile::~VideoFile() {
  if (capture_ != nullptr) {
    KAYLORDUT_LOG_INFO("Release capture")
    capture_->release();
    delete capture_;
  }
}

void VideoFile::Display(const float framerate, const int target_size) {
  const int delay = 1000 / framerate;
  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
  cv::Mat frame;
  ImageProcess image_process(capture_->get(cv::CAP_PROP_FRAME_WIDTH),
                             capture_->get(cv::CAP_PROP_FRAME_HEIGHT),
                             target_size, false, 30);
  while (true) {
    *capture_ >> frame;
    if (frame.empty()) {
      break;
    }
    cv::imshow("Video", *(image_process.Convert(frame)));
    //    cv::imshow("Video", frame);
    if (cv::waitKey(delay) >= 0) {
      break;
    }
  }
  cv::destroyAllWindows();
}

std::unique_ptr<cv::Mat> VideoFile::GetNextFrame() {
  auto frame = std::make_unique<cv::Mat>();
  *capture_ >> *frame;
  if (frame->empty()) {
    return nullptr;
  }
  return std::move(frame);
}

cv::Mat VideoFile::test() {
  cv::Mat frame;
  *capture_ >> frame;
  cv::waitKey(125);
  return frame;
}

int VideoFile::get_frame_width() {
  return capture_->get(cv::CAP_PROP_FRAME_WIDTH);
}

int VideoFile::get_frame_height() {
  return capture_->get(cv::CAP_PROP_FRAME_HEIGHT);
}