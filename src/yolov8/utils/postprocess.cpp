//
// Created by kaylor on 3/5/24.
//

#include "postprocess.h"

#include <set>

#include "Float16.h"
#include "filesystem"
#include "kaylordut/log/logger.h"
#include "opencv2/opencv.hpp"
#include "rknn_matmul_api.h"
static char *labels[OBJ_CLASS_NUM];
inline int clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}
static char *readLine(FILE *fp, char *buffer, int *len) {
  int ch;
  int i = 0;
  size_t buff_len = 0;

  buffer = (char *)malloc(buff_len + 1);
  if (!buffer) return NULL;  // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void *tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL;  // Out of memory
    }
    buffer = (char *)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line) {
  FILE *file = fopen(fileName, "r");
  char *s;
  int i = 0;
  int n = 0;

  if (file == NULL) {
    KAYLORDUT_LOG_ERROR("Open {} fail!", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line) break;
  }
  fclose(file);
  return i;
}

static int loadLabelName(const char *locationFilename, char *label[]) {
  KAYLORDUT_LOG_INFO("load lable {}", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

int init_post_process(std::string &label_path) {
  int ret = 0;
  ret = loadLabelName(label_path.c_str(), labels);
  if (ret < 0) {
    KAYLORDUT_LOG_ERROR("Load {} failed!", label_path);
    return -1;
  }
  return 0;
}

void deinit_post_process() {
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}

const char *coco_cls_to_name(int cls_id) {
  if (cls_id >= OBJ_CLASS_NUM) {
    return "null";
  }
  if (labels[cls_id]) {
    return labels[cls_id];
  }
  return "null";
}
static float CalculateOverlap(float xmin0, float ymin0, float xmax0,
                              float ymax0, float xmin1, float ymin1,
                              float xmax1, float ymax1) {
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
            (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}
static int nms(int validCount, std::vector<float> &outputLocations,
               std::vector<int> classIds, std::vector<int> &order, int filterId,
               float threshold) {
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static void crop_mask(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes,
                      int boxes_num, int *cls_id, int height, int width) {
  for (int b = 0; b < boxes_num; b++) {
    float x1 = boxes[b * 4 + 0];
    float y1 = boxes[b * 4 + 1];
    float x2 = boxes[b * 4 + 2];
    float y2 = boxes[b * 4 + 3];

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // 判断该点在矩形框内
        if (j >= x1 && j < x2 && i >= y1 && i < y2) {
          if (all_mask_in_one[i * width + j] == 0) {
            // seg_mask只有 0或者1 ， cls_id因为可能存在0值，所以 +1
            // 避免结果都为0
            all_mask_in_one[i * width + j] =
                seg_mask[b * width * height + i * width + j] * (cls_id[b] + 1);
          }
        }
      }
    }
  }
}

static void matmul_by_npu_i8(std::vector<float> &A_input, float *B_input,
                             uint8_t *C_input, int ROWS_A, int COLS_A,
                             int COLS_B, rknn_app_context_t *app_ctx) {
  int B_layout = 0;
  int AC_layout = 0;
  int32_t M = 1;
  int32_t K = COLS_A;
  int32_t N = COLS_B;

  rknn_matmul_ctx ctx;
  rknn_matmul_info info;
  memset(&info, 0, sizeof(rknn_matmul_info));
  info.M = M;
  info.K = K;
  info.N = N;
  info.type = RKNN_INT8_MM_INT8_TO_INT32;
  info.B_layout = B_layout;
  info.AC_layout = AC_layout;

  rknn_matmul_io_attr io_attr;
  memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

  int8_t int8Vector_A[ROWS_A * COLS_A];
  for (int i = 0; i < ROWS_A * COLS_A; ++i) {
    int8Vector_A[i] = (int8_t)A_input[i];
  }

  int8_t int8Vector_B[COLS_A * COLS_B];
  for (int i = 0; i < COLS_A * COLS_B; ++i) {
    int8Vector_B[i] = (int8_t)B_input[i];
  }

  int ret = rknn_matmul_create(&ctx, &info, &io_attr);
  // Create A
  rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);
  // Create B
  rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
  // Create C
  rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);

  memcpy(B->virt_addr, int8Vector_B, B->size);
  // Set A
  ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
  // Set B
  ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
  // Set C
  ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);

  for (int i = 0; i < ROWS_A; ++i) {
    memcpy(A->virt_addr, int8Vector_A + i * A->size, A->size);

    // Run
    ret = rknn_matmul_run(ctx);

    for (int j = 0; j < COLS_B; ++j) {
      if (((int32_t *)C->virt_addr)[j] > 0) {
        C_input[i * COLS_B + j] = 1;
      } else {
        C_input[i * COLS_B + j] = 0;
      }
    }
  }

  // destroy
  rknn_destroy_mem(ctx, A);
  rknn_destroy_mem(ctx, B);
  rknn_destroy_mem(ctx, C);
  rknn_matmul_destroy(ctx);
}

static void matmul_by_npu_fp16(std::vector<float> &A_input, float *B_input,
                               uint8_t *C_input, int ROWS_A, int COLS_A,
                               int COLS_B, rknn_app_context_t *app_ctx) {
  int B_layout = 0;
  int AC_layout = 0;
  int32_t M = ROWS_A;
  int32_t K = COLS_A;
  int32_t N = COLS_B;

  rknn_matmul_ctx ctx;
  rknn_matmul_info info;
  memset(&info, 0, sizeof(rknn_matmul_info));
  info.M = M;
  info.K = K;
  info.N = N;
  info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
  info.B_layout = B_layout;
  info.AC_layout = AC_layout;

  rknn_matmul_io_attr io_attr;
  memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

  rknpu2::float16 int8Vector_A[ROWS_A * COLS_A];
  for (int i = 0; i < ROWS_A * COLS_A; ++i) {
    int8Vector_A[i] = (rknpu2::float16)A_input[i];
  }

  rknpu2::float16 int8Vector_B[COLS_A * COLS_B];
  for (int i = 0; i < COLS_A * COLS_B; ++i) {
    int8Vector_B[i] = (rknpu2::float16)B_input[i];
  }

  int ret = rknn_matmul_create(&ctx, &info, &io_attr);
  // Create A
  rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);
  // Create B
  rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
  // Create C
  rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);

  memcpy(A->virt_addr, int8Vector_A, A->size);
  memcpy(B->virt_addr, int8Vector_B, B->size);

  // Set A
  ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
  // Set B
  ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
  // Set C
  ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);

  // Run
  ret = rknn_matmul_run(ctx);
  for (int i = 0; i < ROWS_A * COLS_B; ++i) {
    if (((float *)C->virt_addr)[i] > 0) {
      C_input[i] = 1;
    } else {
      C_input[i] = 0;
    }
  }

  // destroy
  rknn_destroy_mem(ctx, A);
  rknn_destroy_mem(ctx, B);
  rknn_destroy_mem(ctx, C);
  rknn_matmul_destroy(ctx);
}
static void resize_by_opencv(uint8_t *input_image, int input_width,
                             int input_height, uint8_t *output_image,
                             int target_width, int target_height) {
  cv::Mat src_image(input_height, input_width, CV_8U, input_image);
  cv::Mat dst_image;
  cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0,
             cv::INTER_LINEAR);
  memcpy(output_image, dst_image.data, target_width * target_height);
}

static void seg_reverse(uint8_t *seg_mask, uint8_t *cropped_seg,
                        uint8_t *seg_mask_real, int model_in_height,
                        int model_in_width, int proto_height, int proto_width,
                        int cropped_height, int cropped_width,
                        int ori_in_height, int ori_in_width, int y_pad,
                        int x_pad) {
  int cropped_index = 0;
  for (int i = 0; i < proto_height; i++) {
    for (int j = 0; j < proto_width; j++) {
      if (i >= y_pad && i < proto_height - y_pad && j >= x_pad &&
          j < proto_width - x_pad) {
        int seg_index = i * proto_width + j;
        cropped_seg[cropped_index] = seg_mask[seg_index];
        cropped_index++;
      }
    }
  }

  // Note: Here are different methods provided for implementing single-channel
  // image scaling
  resize_by_opencv(cropped_seg, cropped_width, cropped_height, seg_mask_real,
                   ori_in_width, ori_in_height);
  // resize_by_rga_rk356x(cropped_seg, cropped_width, cropped_height,
  // seg_mask_real, ori_in_width, ori_in_height);
  // resize_by_rga_rk3588(cropped_seg, cropped_width, cropped_height,
  // seg_mask_real, ori_in_width, ori_in_height);
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left,
                                     int right, std::vector<int> &indices) {
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

inline static int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

static int box_reverse(int position, int boundary, int pad, float scale) {
  return (int)((clamp(position, 0, boundary) - pad) / scale);
}

void compute_dfl(float *tensor, int dfl_len, float *box) {
  for (int b = 0; b < 4; b++) {
    float exp_t[dfl_len];
    float exp_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i < dfl_len; i++) {
      exp_t[i] = exp(tensor[i + b * dfl_len]);
      exp_sum += exp_t[i];
    }

    for (int i = 0; i < dfl_len; i++) {
      acc_sum += exp_t[i] / exp_sum * i;
    }
    box[b] = acc_sum;
  }
}

static int process_i8(rknn_output *all_input, int input_id, int grid_h,
                      int grid_w, int height, int width, int stride,
                      int dfl_len, std::vector<float> &boxes,
                      std::vector<float> &segments, float *proto,
                      std::vector<float> &objProbs, std::vector<int> &classId,
                      float threshold, rknn_app_context_t *app_ctx) {
  int validCount = 0;
  // 这个张量的H*W，用来记录每一个通道占用内存的长度
  int grid_len = grid_h * grid_w;

  // Skip if input_id is not 0, 4, 8, or 12
  if (input_id % 4 != 0) {
    return validCount;
  }
  // 最后的这一层是跟掩膜相关，处理方式与其他不一样
  if (input_id == 12) {
    /**
     * 获取第12层的输出数据，然后把量化后的数据还原到为原来的浮点型
     * 需要注意的是，这里没有使用比例关系，因为程序需要时INT的数据，不需要0~1的float数据
     */
    int8_t *input_proto = (int8_t *)all_input[input_id].buf;
    int32_t zp_proto = app_ctx->output_attrs[input_id].zp;
    for (int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++) {
      proto[i] = input_proto[i] - zp_proto;
    }
    return validCount;
  }

  int8_t *box_tensor = (int8_t *)all_input[input_id].buf;
  int32_t box_zp = app_ctx->output_attrs[input_id].zp;
  float box_scale = app_ctx->output_attrs[input_id].scale;

  int8_t *score_tensor = (int8_t *)all_input[input_id + 1].buf;
  int32_t score_zp = app_ctx->output_attrs[input_id + 1].zp;
  float score_scale = app_ctx->output_attrs[input_id + 1].scale;

  int8_t *score_sum_tensor = nullptr;
  int32_t score_sum_zp = 0;
  float score_sum_scale = 1.0;
  score_sum_tensor = (int8_t *)all_input[input_id + 2].buf;
  score_sum_zp = app_ctx->output_attrs[input_id + 2].zp;
  score_sum_scale = app_ctx->output_attrs[input_id + 2].scale;

  int8_t *seg_tensor = (int8_t *)all_input[input_id + 3].buf;
  int32_t seg_zp = app_ctx->output_attrs[input_id + 3].zp;
  float seg_scale = app_ctx->output_attrs[input_id + 3].scale;
  // 计算得分阈值的量化之后的数值
  int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
  int8_t score_sum_thres_i8 =
      qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;
      int max_class_id = -1;

      int offset_seg = i * grid_w + j;
      int8_t *in_ptr_seg = seg_tensor + offset_seg;

      // for quick filtering through "score sum"
      if (score_sum_tensor != nullptr) {
        // 如果得分总和少于设定阈值，直接放弃本次的目标
        if (score_sum_tensor[offset] < score_sum_thres_i8) {
          continue;
        }
      }

      int8_t max_score = -score_zp;
      for (int c = 0; c < OBJ_CLASS_NUM; c++) {  // 这里是为了保留最大的概率类别
        if ((score_tensor[offset] > score_thres_i8) &&
            (score_tensor[offset] > max_score)) {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset +=
            grid_len;  // 计算 i * grid_w + j
                       // 这个像素点下，下一个类别偏移量，所以加上一个通道的长度即可
      }

      // compute box， 只有最大概率大于阈值，才判定有目标存在
      if (max_score > score_thres_i8) {
        for (int k = 0; k < PROTO_CHANNEL; k++) {
          // 数据转换为int8， 因为数据是NCHW，所以步长需要grid_len才能拿到数据
          int8_t seg_element_i8 = in_ptr_seg[(k)*grid_len] - seg_zp;
          segments.push_back(seg_element_i8);
        }

        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len * 4];
        for (int k = 0; k < dfl_len * 4; k++) {  // 反量化拿到浮点型数据
          before_dfl[k] =
              deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
          offset += grid_len;
        }
        // 通过dfl计算出box的坐标
        compute_dfl(before_dfl, dfl_len, box);

        float x1, y1, x2, y2, w, h;
        x1 = (-box[0] + j + 0.5) * stride;
        y1 = (-box[1] + i + 0.5) * stride;
        x2 = (box[2] + j + 0.5) * stride;
        y2 = (box[3] + i + 0.5) * stride;
        w = x2 - x1;
        h = y2 - y1;
        boxes.push_back(x1);
        boxes.push_back(y1);
        boxes.push_back(w);
        boxes.push_back(h);
        // 保存该类的概率和类id
        objProbs.push_back(
            deqnt_affine_to_f32(max_score, score_zp, score_scale));
        classId.push_back(max_class_id);
        validCount++;
      }
    }
  }
  return validCount;
}

static int process_fp32(rknn_output *all_input, int input_id, int grid_h,
                        int grid_w, int height, int width, int stride,
                        int dfl_len, std::vector<float> &boxes,
                        std::vector<float> &segments, float *proto,
                        std::vector<float> &objProbs, std::vector<int> &classId,
                        float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;

  // Skip if input_id is not 0, 4, 8, or 12
  if (input_id % 4 != 0) {
    return validCount;
  }

  if (input_id == 12) {
    float *input_proto = (float *)all_input[input_id].buf;
    for (int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++) {
      proto[i] = input_proto[i];
    }
    return validCount;
  }

  float *box_tensor = (float *)all_input[input_id].buf;
  float *score_tensor = (float *)all_input[input_id + 1].buf;
  float *score_sum_tensor = (float *)all_input[input_id + 2].buf;
  float *seg_tensor = (float *)all_input[input_id + 3].buf;

  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;
      int max_class_id = -1;

      int offset_seg = i * grid_w + j;
      float *in_ptr_seg = seg_tensor + offset_seg;

      // for quick filtering through "score sum"
      if (score_sum_tensor != nullptr) {
        if (score_sum_tensor[offset] < threshold) {
          continue;
        }
      }

      float max_score = 0;
      for (int c = 0; c < OBJ_CLASS_NUM; c++) {
        if ((score_tensor[offset] > threshold) &&
            (score_tensor[offset] > max_score)) {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }

      // compute box
      if (max_score > threshold) {
        for (int k = 0; k < PROTO_CHANNEL; k++) {
          float seg_element_f32 = in_ptr_seg[(k)*grid_len];
          segments.push_back(seg_element_f32);
        }

        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len * 4];
        for (int k = 0; k < dfl_len * 4; k++) {
          before_dfl[k] = box_tensor[offset];
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len, box);

        float x1, y1, x2, y2, w, h;
        x1 = (-box[0] + j + 0.5) * stride;
        y1 = (-box[1] + i + 0.5) * stride;
        x2 = (box[2] + j + 0.5) * stride;
        y2 = (box[3] + i + 0.5) * stride;
        w = x2 - x1;
        h = y2 - y1;
        boxes.push_back(x1);
        boxes.push_back(y1);
        boxes.push_back(w);
        boxes.push_back(h);

        objProbs.push_back(max_score);
        classId.push_back(max_class_id);
        validCount++;
      }
    }
  }
  return validCount;
}

int post_process_seg(rknn_app_context_t *app_ctx, rknn_output *outputs,
                     letterbox_t *letter_box, float conf_threshold,
                     float nms_threshold,
                     object_detect_result_list *od_results) {
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  std::vector<float> filterSegments;
  float proto[PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT];
  std::vector<float> filterSegments_by_nms;

  int model_in_w = app_ctx->model_width;   // 获取模型的width
  int model_in_h = app_ctx->model_height;  // 获取模型的height

  int validCount = 0;  // 记录有效的输出
  int stride = 0;      // 活动窗口步长
  int grid_h = 0;
  int grid_w = 0;

  memset(od_results, 0,
         sizeof(object_detect_result_list));  // 把输出结果列表置零

  int dfl_len = app_ctx->output_attrs[0].dims[1] / 4;
  /***
   * 结果输出有三种输出80x80 40x40
   * 20x20，这里取模是为了确定每种输出拥有多少层输出
   */
  int output_per_branch = app_ctx->io_num.n_output / 3;  // default 3 branch

  // process the outputs of rknn
  for (int i = 0; i < 13; i++) {
    grid_h = app_ctx->output_attrs[i].dims[2];  //  这一层输出的高度
    grid_w = app_ctx->output_attrs[i].dims[3];  // 这一层输出的宽度
    stride = model_in_h / grid_h;  // 模型边长对输出层取模等于滑动步长
    // 如果量化了，使用i8的处理方式
    if (app_ctx->is_quant) {
      validCount +=
          process_i8(outputs, i, grid_h, grid_w, model_in_h, model_in_w, stride,
                     dfl_len, filterBoxes, filterSegments, proto, objProbs,
                     classId, conf_threshold, app_ctx);
    } else {
      validCount +=
          process_fp32(outputs, i, grid_h, grid_w, model_in_h, model_in_w,
                       stride, dfl_len, filterBoxes, filterSegments, proto,
                       objProbs, classId, conf_threshold);
    }
  }

  // nms
  if (validCount <= 0) {
    return 0;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }
  // 这里快排把识别的种类概率从大到小排列，对应的index也跟着排列
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {  // 把重合度大于设定阈值的框给标记去掉
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  od_results->count = 0;

  for (int i = 0; i < validCount; ++i) {
    // 上一步中已经标记了无效的重叠框的下标为-1
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];
    // 获取box， 种类id， 概率分数
    float x1 = filterBoxes[n * 4 + 0];
    float y1 = filterBoxes[n * 4 + 1];
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    for (int k = 0; k < PROTO_CHANNEL; k++) {
      // 获取相对应的分割的向量
      filterSegments_by_nms.push_back(filterSegments[n * PROTO_CHANNEL + k]);
    }

    od_results->results[last_count].box.left = x1;
    od_results->results[last_count].box.top = y1;
    od_results->results[last_count].box.right = x2;
    od_results->results[last_count].box.bottom = y2;

    od_results->results[last_count].prop = obj_conf;
    od_results->results[last_count].cls_id = id;
    last_count++;
  }
  od_results->count = last_count;

  int boxes_num = od_results->count;

  // compute the mask (binary matrix) through Matmul
  // 计算掩膜矩阵，最后结果得到 boxes_num 个掩膜矩阵
  int ROWS_A = boxes_num;
  int COLS_A = PROTO_CHANNEL;
  int COLS_B = PROTO_HEIGHT * PROTO_WEIGHT;
  uint8_t matmul_out[boxes_num * PROTO_HEIGHT * PROTO_WEIGHT];
  if (app_ctx->is_quant) {
    matmul_by_npu_i8(filterSegments_by_nms, proto, matmul_out, ROWS_A, COLS_A,
                     COLS_B, app_ctx);
  } else {
    matmul_by_npu_fp16(filterSegments_by_nms, proto, matmul_out, ROWS_A, COLS_A,
                       COLS_B, app_ctx);
  }

  float filterBoxes_by_nms[boxes_num * 4];  // 记录每个box的坐标
  int cls_id[boxes_num];
  for (int i = 0; i < boxes_num; i++) {
    // for crop_mask
    // 掩膜层的分辨率是 160x160
    // 640 / 160 = 4.0
    filterBoxes_by_nms[i * 4 + 0] =
        od_results->results[i].box.left / 4.0;  // x1;
    filterBoxes_by_nms[i * 4 + 1] =
        od_results->results[i].box.top / 4.0;  // y1;
    filterBoxes_by_nms[i * 4 + 2] =
        od_results->results[i].box.right / 4.0;  // x2;
    filterBoxes_by_nms[i * 4 + 3] =
        od_results->results[i].box.bottom / 4.0;  // y2;
    cls_id[i] = od_results->results[i].cls_id;

    // get real box
    // 这里是把640x640的坐标映射返回到原始输入图像的坐标
    od_results->results[i].box.left =
        box_reverse(od_results->results[i].box.left, model_in_w,
                    letter_box->x_pad, letter_box->scale);
    od_results->results[i].box.top =
        box_reverse(od_results->results[i].box.top, model_in_h,
                    letter_box->y_pad, letter_box->scale);
    od_results->results[i].box.right =
        box_reverse(od_results->results[i].box.right, model_in_w,
                    letter_box->x_pad, letter_box->scale);
    od_results->results[i].box.bottom =
        box_reverse(od_results->results[i].box.bottom, model_in_h,
                    letter_box->y_pad, letter_box->scale);
  }

  // crop seg outside box
  uint8_t all_mask_in_one[PROTO_HEIGHT * PROTO_WEIGHT] = {0};
  // 把所有的掩膜数据写到一张图上
  crop_mask(matmul_out, all_mask_in_one, filterBoxes_by_nms, boxes_num, cls_id,
            PROTO_HEIGHT, PROTO_WEIGHT);

  // get real mask
  int cropped_height = PROTO_HEIGHT - letter_box->y_pad / 4 * 2;
  int cropped_width = PROTO_WEIGHT - letter_box->x_pad / 4 * 2;
  int y_pad = letter_box->y_pad / 4;  // 640 / 160 = 4
  int x_pad = letter_box->x_pad / 4;
  int ori_in_height = (model_in_h - letter_box->y_pad * 2) / letter_box->scale;
  int ori_in_width = (model_in_w - letter_box->x_pad * 2) / letter_box->scale;
  uint8_t *cropped_seg_mask =
      (uint8_t *)malloc(cropped_height * cropped_width * sizeof(uint8_t));
  uint8_t *real_seg_mask =
      (uint8_t *)malloc(ori_in_height * ori_in_width * sizeof(uint8_t));
  // 还原到原来的图像分辨率，结果保存到real_seg_mask中，使用od_results->results_seg[0].seg_mask记录
  seg_reverse(all_mask_in_one, cropped_seg_mask, real_seg_mask, model_in_h,
              model_in_w, PROTO_HEIGHT, PROTO_WEIGHT, cropped_height,
              cropped_width, ori_in_height, ori_in_width, y_pad, x_pad);
  od_results->results_seg[0].seg_mask = real_seg_mask;
  free(cropped_seg_mask);

  return 0;
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp,
                      float score_sum_scale, int grid_h, int grid_w, int stride,
                      int dfl_len, std::vector<float> &boxes,
                      std::vector<float> &objProbs, std::vector<int> &classId,
                      float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
  int8_t score_sum_thres_i8 =
      qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;
      int max_class_id = -1;

      // 通过 score sum 起到快速过滤的作用
      if (score_sum_tensor != nullptr) {
        if (score_sum_tensor[offset] < score_sum_thres_i8) {
          continue;
        }
      }

      int8_t max_score = -score_zp;
      for (int c = 0; c < OBJ_CLASS_NUM; c++) {
        if ((score_tensor[offset] > score_thres_i8) &&
            (score_tensor[offset] > max_score)) {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }

      // compute box
      if (max_score > score_thres_i8) {
        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len * 4];
        for (int k = 0; k < dfl_len * 4; k++) {
          before_dfl[k] =
              deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len, box);

        float x1, y1, x2, y2, w, h;
        x1 = (-box[0] + j + 0.5) * stride;
        y1 = (-box[1] + i + 0.5) * stride;
        x2 = (box[2] + j + 0.5) * stride;
        y2 = (box[3] + i + 0.5) * stride;
        w = x2 - x1;
        h = y2 - y1;
        boxes.push_back(x1);
        boxes.push_back(y1);
        boxes.push_back(w);
        boxes.push_back(h);

        objProbs.push_back(
            deqnt_affine_to_f32(max_score, score_zp, score_scale));
        classId.push_back(max_class_id);
        validCount++;
      }
    }
  }
  return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor,
                        float *score_sum_tensor, int grid_h, int grid_w,
                        int stride, int dfl_len, std::vector<float> &boxes,
                        std::vector<float> &objProbs, std::vector<int> &classId,
                        float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;
      int max_class_id = -1;

      // 通过 score sum 起到快速过滤的作用
      if (score_sum_tensor != nullptr) {
        if (score_sum_tensor[offset] < threshold) {
          continue;
        }
      }

      float max_score = 0;
      for (int c = 0; c < OBJ_CLASS_NUM; c++) {
        if ((score_tensor[offset] > threshold) &&
            (score_tensor[offset] > max_score)) {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }

      // compute box
      if (max_score > threshold) {
        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len * 4];
        for (int k = 0; k < dfl_len * 4; k++) {
          before_dfl[k] = box_tensor[offset];
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len, box);

        float x1, y1, x2, y2, w, h;
        x1 = (-box[0] + j + 0.5) * stride;
        y1 = (-box[1] + i + 0.5) * stride;
        x2 = (box[2] + j + 0.5) * stride;
        y2 = (box[3] + i + 0.5) * stride;
        w = x2 - x1;
        h = y2 - y1;
        boxes.push_back(x1);
        boxes.push_back(y1);
        boxes.push_back(w);
        boxes.push_back(h);

        objProbs.push_back(max_score);
        classId.push_back(max_class_id);
        validCount++;
      }
    }
  }
  return validCount;
}

int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results) {
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int validCount = 0;
  int stride = 0;
  int grid_h = 0;
  int grid_w = 0;
  int model_in_w = app_ctx->model_width;
  int model_in_h = app_ctx->model_height;

  memset(od_results, 0, sizeof(object_detect_result_list));

  // default 3 branch
  int dfl_len = app_ctx->output_attrs[0].dims[1] / 4;
  int output_per_branch = app_ctx->io_num.n_output / 3;
  for (int i = 0; i < 3; i++) {
    void *score_sum = nullptr;
    int32_t score_sum_zp = 0;
    float score_sum_scale = 1.0;
    if (output_per_branch == 3) {
      score_sum = outputs[i * output_per_branch + 2].buf;
      score_sum_zp = app_ctx->output_attrs[i * output_per_branch + 2].zp;
      score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
    }
    int box_idx = i * output_per_branch;
    int score_idx = i * output_per_branch + 1;

    grid_h = app_ctx->output_attrs[box_idx].dims[2];
    grid_w = app_ctx->output_attrs[box_idx].dims[3];
    stride = model_in_h / grid_h;

    if (app_ctx->is_quant) {
      validCount += process_i8(
          (int8_t *)outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp,
          app_ctx->output_attrs[box_idx].scale,
          (int8_t *)outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp,
          app_ctx->output_attrs[score_idx].scale, (int8_t *)score_sum,
          score_sum_zp, score_sum_scale, grid_h, grid_w, stride, dfl_len,
          filterBoxes, objProbs, classId, conf_threshold);
    } else {
      validCount += process_fp32(
          (float *)outputs[box_idx].buf, (float *)outputs[score_idx].buf,
          (float *)score_sum, grid_h, grid_w, stride, dfl_len, filterBoxes,
          objProbs, classId, conf_threshold);
    }
  }

  // no object detect
  if (validCount <= 0) {
    return 0;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  od_results->count = 0;

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
    float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    od_results->results[last_count].box.left =
        (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.top =
        (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].box.right =
        (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.bottom =
        (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].prop = obj_conf;
    od_results->results[last_count].cls_id = id;
    last_count++;
  }
  od_results->count = last_count;
  return 0;
}