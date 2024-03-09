//
// Created by kaylor on 3/5/24.
//

#pragma once
#include <stdint.h>

#include "common.h"
#include "rknn_api.h"
#include <vector>

int init_post_process();
void deinit_post_process();
const char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results);
