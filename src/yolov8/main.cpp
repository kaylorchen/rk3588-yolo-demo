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

//
//void run(Yolov8 &yolov_8,
//         ImageProcess &image_process,
//         std::shared_ptr<cv::Mat> origin,
//         cv::Mat rgb,
//         const letterbox_t &letter_box) {
////  object_detect_result_list od_results;
////  yolov_8.Inference(rgb.ptr(), &od_results, letter_box);
////  image_process.ImagePostProcess(*origin, od_results);
//  cv::imshow("Video", *origin);
//  cv::waitKey(1);
//}
//
//int main(int argc, char *agrv[]) {
//  Yolov8 yolov_8(agrv[1]);
////  Yolov8 yolov_8("./yolov8s-seg.rknn");
//  VideoFile video_file(agrv[2]);
////    video_file.Display(125, 640);
//  ImageProcess image_process(video_file.get_frame_width(), video_file.get_frame_height(), 640);
//  auto original_img = video_file.GetNextFrame();
//  auto convert_img = image_process.Convert(*original_img);
//  cv::Mat rgb_img = cv::Mat::zeros(640, 640, convert_img->type());
//  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
//  ThreadPool pool(1);
//  while (convert_img != nullptr) {
//    cv::cvtColor(*convert_img, rgb_img, cv::COLOR_BGR2RGB);
////    pool.enqueue([&](std::shared_ptr<cv::Mat> image) {
////      run(yolov_8,
////          image_process,
////          std::move(image),
////          rgb_img,
////          image_process.get_letter_box());
////    }, std::move(original_img));
//    original_img = video_file.GetNextFrame();
//    convert_img = image_process.Convert(*original_img);
//  }
//  KAYLORDUT_LOG_INFO("exit loop");
//
//  std::this_thread::sleep_for(std::chrono::milliseconds(4000));
//  cv::destroyAllWindows();
//  return 0;
//}
