//
// Created by kaylor on 3/5/24.
//

#pragma once
#include "rknn_api.h"
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROTO_CHANNEL (32)
#define PROTO_HEIGHT  (160)
#define PROTO_WEIGHT (160)
/**
 * @brief LetterBox
 *
 */
typedef struct {
  int x_pad;
  int y_pad;
  float scale;
} letterbox_t;
/**
 * @brief Image rectangle
 *
 */
typedef struct {
  int left;
  int top;
  int right;
  int bottom;
} image_rect_t;

typedef struct {
  image_rect_t box;
  float prop;
  int cls_id;
} object_detect_result;

typedef struct
{
  uint8_t *seg_mask;
} object_segment_result;

typedef struct {
  int id;
  int count;
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
  object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  bool is_quant;
} rknn_app_context_t;
