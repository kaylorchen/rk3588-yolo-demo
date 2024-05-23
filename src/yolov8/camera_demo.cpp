//
// Created by kaylor on 3/9/24.
//
#include <cctype>  // 用于检查字符是否为数字
#include <cerrno>  // 用于检查 strtol 和 strtod 的错误

#include "camera.h"
#include "getopt.h"
#include "kaylordut/log/logger.h"
#include "kaylordut/time/time_duration.h"
#include "kaylordut/time/timeout.h"
#include "rknn_pool.h"

struct ProgramOptions {
  std::string model_path;
  std::string label_path;
  int thread_count;
  int camera_index;
  int width;
  int height;
  double fps;
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
      {"camera_index", required_argument, nullptr, 'i'},
      {"width", required_argument, nullptr, 'w'},
      {"height", required_argument, nullptr, 'h'},
      {"fps", required_argument, nullptr, 'f'},
      {"help", no_argument, nullptr, '?'},
      {"track", no_argument, nullptr, 'T'},
      {nullptr, 0, nullptr, 0}};

  int c, optionIndex = 0;
  while ((c = getopt_long(argc, argv, "m:l:t:i:w:h:f:?T", longOpts,
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
      case 'i':
        if (isNumber(optarg)) {
          options.camera_index = std::atoi(optarg);
          if (options.camera_index < 0) {
            KAYLORDUT_LOG_ERROR("Invalid index of camera: {}", optarg);
            return false;
          }
        } else {
          KAYLORDUT_LOG_ERROR("Camera index must be a number: {}", optarg);
          return false;
        }
        break;
      case 'w':
        if (isNumber(optarg)) {
          options.width = std::atoi(optarg);
          if (options.width < 0) {
            KAYLORDUT_LOG_ERROR("Invalid width: {}", optarg);
            return false;
          }
        } else {
          KAYLORDUT_LOG_ERROR("Width must be a number: {}", optarg);
          return false;
        }
        break;
      case 'h':
        if (isNumber(optarg)) {
          options.height = std::atoi(optarg);
          if (options.height < 0) {
            KAYLORDUT_LOG_ERROR("Invalid height: {}", optarg);
            return false;
          }
        } else {
          KAYLORDUT_LOG_ERROR("Height must be a number: {}", optarg);
          return false;
        }
        break;
      case 'f':
        if (isNumber(optarg)) {
          options.fps = std::atof(optarg);
          if (options.fps <= 0) {
            KAYLORDUT_LOG_ERROR("Invalid frame rate: {}", optarg);
            return false;
          }
        } else {
          KAYLORDUT_LOG_ERROR("Frame rate must be a number: {}", optarg);
          return false;
        }
        break;
      case 'T':
        options.is_track = true;
        break;
      case '?':
        std::cout << "Usage: " << argv[0]
                  << " [--model_path|-m model_path] [--camera_index|-i index] "
                     "[--width|-w width] [--height|-h height]"
                     "[--threads|-t thread_count] [--fps|-f framerate] "
                     "[--label_path|-l label_path]\n";
        exit(EXIT_SUCCESS);
      default:
        std::cout << "Usage: " << argv[0]
                  << " [--model_path|-m model_path] [--camera_index|-i index] "
                     "[--width|-w width] [--height|-h height]"
                     "[--threads|-t thread_count] [--fps|-f framerate] "
                     "[--label_path|-l label_path]\n";
        abort();
    }
  }

  return true;
}

std::string getCurrentTimeStr() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
  return ss.str();
}

int main(int argc, char *argv[]) {
  ProgramOptions options = {"", "", 0, 0, 0, 0, 0.0};
  if (!parseCommandLine(argc, argv, options)) {
    KAYLORDUT_LOG_ERROR("Parse command failed.");
    return 1;
  }
  if (options.fps == 0.0 || options.thread_count == 0 || options.height == 0 ||
      options.width == 0 || options.label_path.empty() ||
      options.model_path.empty()) {
    KAYLORDUT_LOG_ERROR("Missing required options. Use --help for help.");
    return 1;
  }
  auto rknn_pool = std::make_unique<RknnPool>(
      options.model_path, options.thread_count, options.label_path);
  auto camera = std::make_unique<Camera>(
      options.camera_index, cv::Size(options.width, options.height),
      options.fps);
  ImageProcess image_process(options.width, options.height, 640,
                             options.is_track, options.fps);
  //  cv::VideoWriter video_writer(
  //      getCurrentTimeStr() + ".mkv", cv::VideoWriter::fourcc('X', '2', '6',
  //      '4'), options.fps, cv::Size(options.width, options.height), true);
  //  if (!video_writer.isOpened()) {
  //    KAYLORDUT_LOG_ERROR("Open the output video file error");
  //    return -1;
  //  }
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
  //  video_writer.release();
  rknn_pool.reset();
  cv::destroyAllWindows();
  return 0;
}