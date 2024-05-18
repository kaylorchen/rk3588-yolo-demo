//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "common.h"
#include "memory"
#include "mutex"
#include "rknn_api.h"
#include "string"

class Yolov8 {
 public:
  Yolov8(std::string &&model_path);
  ~Yolov8();
  int Inference(void *image_buf, object_detect_result_list *od_results,
                letterbox_t letter_box);
  rknn_context *get_rknn_context();
  int Init(rknn_context *ctx_in, bool copy_weight);
  int DeInit();
  int get_model_width();
  int get_model_height();

 private:
  rknn_app_context_t app_ctx_;
  rknn_context ctx_{0};
  std::string model_path_;
  std::unique_ptr<rknn_input[]> inputs_;
  std::unique_ptr<rknn_output[]> outputs_;
  std::mutex outputs_lock_;
  ModelType model_type_;
};
