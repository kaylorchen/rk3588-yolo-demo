#include "image_process.h"
#include "kaylordut/log/logger.h"
#include "kaylordut/time/time_duration.h"
#include "rknn_pool.h"
#include "videofile.h"
#include "kaylordut/time/run_once.h"

int main(int argc, char *argv[]) {
  auto rknn_pool = std::make_unique<RknnPool>(argv[1], std::atoi(argv[3]));
  VideoFile video_file(argv[2]);
  int frame_rate = std::atoi(argv[4]);
  int delay = 1000 / frame_rate;
  ImageProcess image_process(video_file.get_frame_width(),
                             video_file.get_frame_height(), 640);
  std::unique_ptr<cv::Mat> image;
  std::shared_ptr<cv::Mat> image_res;
  uint8_t running_flag = 0;
  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
  static int image_count = 0;
  static int image_res_count = 0;
  TimeDuration time_duration;
  do {
    auto func = [&] {
      running_flag = 0;
      image = video_file.GetNextFrame();
      if (image != nullptr) {
        rknn_pool->AddInferenceTask(std::move(image), image_process);
        running_flag |= 0x01;
        image_count++;
      }
      image_res = rknn_pool->GetImageResultFromQueue();
      if (image_res != nullptr) {
        cv::imshow("Video", *image_res);
        image_res_count++;
        KAYLORDUT_LOG_INFO("image count = {}, image res count = {}, delta = {}", image_count,
                           image_res_count, image_count - image_res_count);
        cv::waitKey(1);
        running_flag |= 0x10;
      }
    };
    run_once_with_delay(func, std::chrono::milliseconds(delay));
  }
    // 读取图像或者获取结果非空 线程池里还有任务 推理的结果没有等于输入图像数
  while (running_flag || rknn_pool->GetTasksSize() ||
      image_count > image_res_count);
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
      time_duration.DurationSinceLastTime());
  double fps = image_res_count * 1000.0 / time.count();
  KAYLORDUT_LOG_INFO("Total time is {}ms, and average frame rate is {}fps",
                     time.count(), fps);
  rknn_pool.reset();
  KAYLORDUT_LOG_INFO("exit loop");
  cv::destroyAllWindows();
  return 0;
}