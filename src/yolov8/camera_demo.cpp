//
// Created by kaylor on 3/9/24.
//
#include "camera.h"
#include "kaylordut/log/logger.h"
#include "kaylordut/time/time_duration.h"
#include "kaylordut/time/timeout.h"
#include "rknn_pool.h"

const int width = 1280;
const int height = 720;
const int fps = 30;
std::string getCurrentTimeStr() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
  return ss.str();
}

int main(int argc, char *argv[]) {
  auto rknn_pool = std::make_unique<RknnPool>(argv[1], std::atoi(argv[2]));
  auto camera = std::make_unique<Camera>(0, cv::Size(width, height), fps);
  ImageProcess image_process(width, height, 640);
  cv::VideoWriter video_writer(getCurrentTimeStr() + ".mkv",
                               cv::VideoWriter::fourcc('X', '2', '6', '4'), fps,
                               cv::Size(width, height), true);
  if (!video_writer.isOpened()) {
    KAYLORDUT_LOG_ERROR("Open the output video file error");
    return -1;
  }
  std::unique_ptr<cv::Mat> image;
  std::shared_ptr<cv::Mat> image_res;
  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
  static int image_count = 0;
  static int image_res_count = 0;
  TimeDuration time_duration;
  Timeout timeout(std::chrono::seconds(30));
  TimeDuration total_time;
  while ((!timeout.isTimeout()) || (image_count != image_res_count)) {
    auto func = [&] {
      if (!timeout.isTimeout()) {
        image = camera->GetNextFrame();
      }
      if (image != nullptr) {
        rknn_pool->AddInferenceTask(std::move(image), image_process);
        image_count++;
      }
      image_res = rknn_pool->GetImageResultFromQueue();
      if (image_res != nullptr) {
        image_res_count++;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_duration.DurationSinceLastTime());
        KAYLORDUT_LOG_INFO(
            "image count = {}, image res count = {}, delta = {}, duration = "
            "{}ms",
            image_count, image_res_count, image_count - image_res_count,
            duration.count());
        cv::imshow("Video", *image_res);
//        video_writer.write(*image_res);
        cv::waitKey(1);
      }
    };
    func();
  }
  auto duration_total_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          total_time.DurationSinceLastTime());
  KAYLORDUT_LOG_INFO(
      "Process {} frames, total time is {}ms, average frame rate is {}",
      image_res_count, duration_total_time.count(),
      image_res_count * 1000.0 / duration_total_time.count());
  video_writer.release();
  rknn_pool.reset();
  cv::destroyAllWindows();
  return 0;
}