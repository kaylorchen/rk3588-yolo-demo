//
// Created by kaylor on 3/4/24.
//

#pragma once
#include "string"
#include "rknn_api.h"

typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr* input_attrs;
  rknn_tensor_attr* output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  bool is_quant;
} rknn_app_context_t;

class Yolov8 {
 public:
  Yolov8(std::string && model_path);
 private:
  rknn_app_context_t rknn_app_context_;
};
