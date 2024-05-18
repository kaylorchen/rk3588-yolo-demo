#include <cctype>  // 用于检查字符是否为数字
#include <cerrno>  // 用于检查 strtol 和 strtod 的错误

#include "getopt.h"
#include "image_process.h"
#include "kaylordut/log/logger.h"
#include "kaylordut/time/run_once.h"
#include "kaylordut/time/time_duration.h"
#include "rknn_pool.h"

struct ProgramOptions {
  std::string model_path;
  std::string label_path;
  std::string input_filename;
  int thread_count;
  double framerate;
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
  static struct option longOpts[] = {
      {"model_path", required_argument, nullptr, 'm'},
      {"label_path", required_argument, nullptr, 'l'},
      {"input_filename", required_argument, nullptr, 'i'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int c, optionIndex = 0;
  while ((c = getopt_long(argc, argv, "m:l:t:f:i:h", longOpts, &optionIndex)) !=
         -1) {
    switch (c) {
      case 'm':
        options.model_path = optarg;
        break;
      case 'l':
        options.label_path = optarg;
        break;
      case 'i':
        options.input_filename = optarg;
        break;
      case 'h':
        std::cout << "Usage: " << argv[0]
                  << " [--model_path|-m model_path] [--input_filename|-i "
                     "input_filename] "
                     "[--label_path|-l label_path]\n";
        exit(EXIT_SUCCESS);
      case '?':
        // 错误消息由getopt_long自动处理
        return false;
      default:
        std::cout << "Usage: " << argv[0]
                  << " [--model_path|-d model_path] [--input_filename|-i "
                     "input_filename] "
                     "[--label_path|-l label_path]\n";
        abort();
    }
  }

  return true;
}

int main(int argc, char *argv[]) {
  KAYLORDUT_LOG_INFO("Yolov8 demo for rk3588");
  ProgramOptions options = {"", "", "", 1, 1.0};
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
  std::unique_ptr<cv::Mat> image = std::make_unique<cv::Mat>();
  *image = cv::imread(options.input_filename);
  if (image->empty()) {
    KAYLORDUT_LOG_ERROR("read image error");
    return -1;
  }
  ImageProcess image_process(image->cols, image->rows, 640);

  std::shared_ptr<cv::Mat> image_res;
  uint8_t running_flag = 0;
  cv::namedWindow("Image demo", cv::WINDOW_AUTOSIZE);
  static int image_count = 0;
  static int image_res_count = 0;
  rknn_pool->AddInferenceTask(std::move(image), image_process);
  while (image_res == nullptr) {
    image_res = rknn_pool->GetImageResultFromQueue();
  }
  cv::imshow("Image demo", *image_res);
  cv::waitKey(0);
  rknn_pool.reset();
  cv::destroyAllWindows();
  cv::imwrite("result_" + options.input_filename, *image_res);
  return 0;
}