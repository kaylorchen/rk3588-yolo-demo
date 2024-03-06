#include "videofile.h"
#include "image_process.h"
#include "kaylordut/log/logger.h"
#include "rknn_pool.h"

int main(int argc, char *argv[]) {
  auto rknn_pool = std::make_unique<RknnPool>(argv[1], 8);
  VideoFile video_file(argv[2]);
  ImageProcess image_process(video_file.get_frame_width(), video_file.get_frame_height(), 640);
  std::unique_ptr<cv::Mat> image;
  std::shared_ptr<cv::Mat> image_res;
  uint8_t running_flag = 0;
  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
  static int image_count = 0;
  static int image_res_count = 0;
  do {
    running_flag = 0;
    image = video_file.GetNextFrame();
    if (image != nullptr) {
      rknn_pool->AddInferenceTask(std::move(image), image_process);
      running_flag |=0x01;
      image_count++;
    }
    image_res = rknn_pool->GetImageResultFromQueue();
    if (image_res != nullptr) {
      cv::imshow("Video", *image_res);
      image_res_count++;
      KAYLORDUT_LOG_INFO("image count = {}, image res count = {}", image_count, image_res_count);
      cv::waitKey(1);
      running_flag |=0x10;
    }
    KAYLORDUT_LOG_INFO("running flag is 0x{:02X}", running_flag);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while (running_flag || rknn_pool->GetTasksSize());
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  rknn_pool.reset();
  KAYLORDUT_LOG_INFO("exit loop");
  cv::destroyAllWindows();
  return 0;
}