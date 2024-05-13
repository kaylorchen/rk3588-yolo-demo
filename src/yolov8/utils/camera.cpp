//
// Created by kaylor on 3/9/24.
//

#include "camera.h"

#include "kaylordut/log/logger.h"
#include "thread"

Camera::Camera(uint16_t index, cv::Size size, double framerate)
    : capture_(index, cv::CAP_V4L2), size_(size) {
  KAYLORDUT_LOG_INFO("Instantiate a Camera object");
  // 这里使用V4L2捕获，因为使用默认的捕获不可以设置捕获的模式和帧率
  if (!capture_.isOpened()) {
    KAYLORDUT_LOG_ERROR("Error opening video stream or file");
    exit(EXIT_FAILURE);
  }
  capture_.set(cv::CAP_PROP_FOURCC,
               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  // 检查是否成功设置格式
  int fourcc = capture_.get(cv::CAP_PROP_FOURCC);
  if (fourcc != cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) {
    KAYLORDUT_LOG_WARN("Set video format failed");
  }
  capture_.set(cv::CAP_PROP_FRAME_WIDTH, size_.width);
  capture_.set(cv::CAP_PROP_FRAME_HEIGHT, size_.height);
  if (!capture_.set(cv::CAP_PROP_FPS, framerate)) {
    KAYLORDUT_LOG_WARN("set framerate failed!!");
  }
  std::this_thread::sleep_for(std::chrono::seconds(1));
  KAYLORDUT_LOG_INFO("camera width: {}, height: {}, fps: {}",
                     capture_.get(cv::CAP_PROP_FRAME_WIDTH),
                     capture_.get(cv::CAP_PROP_FRAME_HEIGHT),
                     capture_.get(cv::CAP_PROP_FPS));
}

Camera::~Camera() {
  if (capture_.isOpened()) {
    KAYLORDUT_LOG_INFO("Release camera");
    capture_.release();
  }
}

std::unique_ptr<cv::Mat> Camera::GetNextFrame() {
  auto frame = std::make_unique<cv::Mat>();
  capture_ >> *frame;
  if (frame->empty()) {
    KAYLORDUT_LOG_ERROR("Get frame error");
    return nullptr;
  }
  return std::move(frame);
}