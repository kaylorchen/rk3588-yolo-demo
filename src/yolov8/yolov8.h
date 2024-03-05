//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "common.h"
#include "memory"
#include "rknn_api.h"
#include "string"

class Yolov8 {
 public:
  Yolov8(std::string && model_path);
  ~Yolov8();
  int Inference(void *image_buf, object_detect_result_list *od_results, letterbox_t letter_box);

private:
  rknn_app_context_t app_ctx_;
};
