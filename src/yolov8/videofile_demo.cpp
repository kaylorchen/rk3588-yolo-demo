#include <cctype>  // 用于检查字符是否为数字
#include <cerrno>  // 用于检查 strtol 和 strtod 的错误

#include "getopt.h"
#include "image_process.h"
#include "kaylordut/log/logger.h"
#include "kaylordut/time/run_once.h"
#include "kaylordut/time/time_duration.h"
#include "rknn_pool.h"
#include "videofile.h"

struct ProgramOptions {
  std::string model_path;
  std::string label_path;
  std::string input_filename;
  int thread_count;
  double framerate;
  bool is_track = false;
};

// 检查字符串是否表示有效的数字
bool isNumber(const std::string &str) {
  char *end;
  errno = 0;
  std::strtod(str.c_str(), &end);
  return errno == 0 && *end == '\0' && end != str.c_str();
}

// 这个函数将解析命令行参数并返回一个 ProgramOptions 结构体
bool parseCommandLine(int argc, char *argv[], ProgramOptions &options) {
  options.is_track = false;
  static struct option longOpts[] = {
      {"model_path", required_argument, nullptr, 'm'},
      {"label_path", required_argument, nullptr, 'l'},
      {"threads", required_argument, nullptr, 't'},
      {"framerate", required_argument, nullptr, 'f'},
      {"input_filename", required_argument, nullptr, 'i'},
      {"help", no_argument, nullptr, 'h'},
      {"track", no_argument, nullptr, 'T'},
      {nullptr, 0, nullptr, 0}};

  int c, optionIndex = 0;
  while ((c = getopt_long(argc, argv, "m:l:t:f:i:hT", longOpts,
                          &optionIndex)) != -1) {
    switch (c) {
      case 'm':
        options.model_path = optarg;
        break;
      case 'l':
        options.label_path = optarg;
        break;
      case 't':
        if (isNumber(optarg)) {
          options.thread_count = std::atoi(optarg);
          if (options.thread_count <= 0) {
            KAYLORDUT_LOG_ERROR("Invalid number of threads: {}", optarg);
            return false;
          }
        } else {
          KAYLORDUT_LOG_ERROR("Thread count must be a number: {}", optarg);
          return false;
        }
        break;
      case 'f':
        if (isNumber(optarg)) {
          options.framerate = std::atof(optarg);
          if (options.framerate <= 0) {
            KAYLORDUT_LOG_ERROR("Invalid frame rate: {}", optarg);
            return false;
          }
        } else {
          KAYLORDUT_LOG_ERROR("Frame rate must be a number: {}", optarg);
          return false;
        }
        break;
      case 'i':
        options.input_filename = optarg;
        break;
      case 'h':
        std::cout << "Usage: " << argv[0]
                  << " [--model_path|-m model_path] [--input_filename|-i "
                     "input_filename] "
                     "[--threads|-t thread_count] [--framerate|-f framerate] "
                     "[--label_path|-l label_path]\n";
        exit(EXIT_SUCCESS);
      case 'T':
        options.is_track = true;
        break;
      case '?':
        // 错误消息由getopt_long自动处理
        return false;
      default:
        std::cout << "Usage: " << argv[0]
                  << " [--model_path|-d model_path] [--input_filename|-i "
                     "input_filename] "
                     "[--threads|-t thread_count] [--framerate|-f framerate] "
                     "[--label_path|-l label_path]\n";
        abort();
    }
  }

  return true;
}

int main(int argc, char *argv[]) {
  KAYLORDUT_LOG_INFO("Yolov8 demo for rk3588");
  ProgramOptions options = {"", "", "", 0, 0.0};
  if (!parseCommandLine(argc, argv, options)) {
    KAYLORDUT_LOG_ERROR("Parse command failed.");
    return 1;
  }
  if (options.framerate == 0.0 || options.thread_count == 0 ||
      options.label_path.empty() || options.input_filename.empty() ||
      options.model_path.empty()) {
    KAYLORDUT_LOG_ERROR("Missing required options. Use --help for help.");
    return 1;
  }
  auto rknn_pool = std::make_unique<RknnPool>(
      options.model_path, options.thread_count, options.label_path);
  VideoFile video_file(options.input_filename.c_str());
  int delay = 1000 / options.framerate;
  ImageProcess image_process(video_file.get_frame_width(),
                             video_file.get_frame_height(), 640,
                             options.is_track, options.framerate);
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
        KAYLORDUT_LOG_INFO("image count = {}, image res count = {}, delta = {}",
                           image_count, image_res_count,
                           image_count - image_res_count);
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