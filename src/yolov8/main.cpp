#include "videofile.h"
#include "yolov8.h"
#include "image_process.h"
#include "kaylordut/log/logger.h"
#include "threadpool.h"

void run(Yolov8 &yolov_8, ImageProcess &image_process, std::shared_ptr<cv::Mat> origin, cv::Mat rgb, const letterbox_t &letter_box){
  object_detect_result_list od_results;
  yolov_8.Inference(rgb.ptr(),  &od_results, letter_box); image_process.ImagePostProcess(*origin, od_results);
  cv::imshow("Video", *origin);
  cv::waitKey(1);
}

int main(int argc, char *agrv[]) {
  Yolov8 yolov_8(agrv[1]);
//  Yolov8 yolov_8("./yolov8s-seg.rknn");
  VideoFile video_file(agrv[2]);
//    video_file.Display(125, 640);
  ImageProcess image_process(video_file.get_frame_width(), video_file.get_frame_height(), 640);
  auto original_img = video_file.GetNextFrame();
  auto convert_img = image_process.Convert(*original_img);
  cv::Mat rgb_img = cv::Mat::zeros(640, 640, convert_img->type());
  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
  ThreadPool pool(1);
  while (convert_img != nullptr) {
    cv::cvtColor(*convert_img, rgb_img, cv::COLOR_BGR2RGB);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    pool.enqueue([&]{ run(yolov_8, image_process, std::move(original_img), rgb_img, image_process.get_letter_box());});
//    run(yolov_8, image_process, std::move(original_img), rgb_img, image_process.get_letter_box());

    original_img = video_file.GetNextFrame();
    convert_img = image_process.Convert(*original_img);
  }
  KAYLORDUT_LOG_INFO("exit loop");

  std::this_thread::sleep_for(std::chrono::milliseconds(4000));
  cv::destroyAllWindows();
  return 0;
}
