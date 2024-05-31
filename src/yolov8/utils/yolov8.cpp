//
// Created by kaylor on 3/4/24.
//

#include "yolov8.h"

#include "kaylordut/log/logger.h"
#include "kaylordut/time/time_duration.h"
#include "postprocess.h"

const int RK3588 = 3;

// 设置模型需要绑定的核心
// Set the core of the model that needs to be bound
int get_core_num() {
  static int core_num = 0;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  int temp = core_num % RK3588;
  core_num++;
  return temp;
}

int read_data_from_file(const char *path, char **out_data) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    KAYLORDUT_LOG_INFO("fopen {} failed!", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  char *data = (char *)malloc(file_size + 1);
  data[file_size] = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != fread(data, 1, file_size, fp)) {
    KAYLORDUT_LOG_INFO("fread {} failed!", path);
    free(data);
    fclose(fp);
    return -1;
  }
  if (fp) {
    fclose(fp);
  }
  *out_data = data;
  return file_size;
}
static void dump_tensor_attr(rknn_tensor_attr *attr) {
  KAYLORDUT_LOG_INFO(
      "index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], n_elems={}, "
      "size={}, fmt={}, type={}, qnt_type={}, "
      "zp={}, scale={}",
      attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
      attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
      get_format_string(attr->fmt), get_type_string(attr->type),
      get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

Yolov8::Yolov8(std::string &&model_path) : model_path_(model_path) {}

int Yolov8::Init(rknn_context *ctx_in, bool copy_weight) {
  int model_len = 0;
  char *model;
  int ret = 0;
  model_len = read_data_from_file(model_path_.c_str(), &model);
  if (model == nullptr) {
    KAYLORDUT_LOG_ERROR("Load model failed");
    return -1;
  }
  if (copy_weight) {
    KAYLORDUT_LOG_INFO("rknn_dup_context() is called");
    // 复用模型参数
    ret = rknn_dup_context(ctx_in, &ctx_);
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("rknn_dup_context failed! error code = {}", ret);
      return -1;
    }
  } else {
    KAYLORDUT_LOG_INFO("rknn_init() is called");
    ret = rknn_init(&ctx_, model, model_len, 0, NULL);
    free(model);
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("rknn_init failed! error code = {}", ret);
      return -1;
    }
  }
  rknn_core_mask core_mask;
  switch (get_core_num()) {
    case 0:
      core_mask = RKNN_NPU_CORE_0;
      break;
    case 1:
      core_mask = RKNN_NPU_CORE_1;
      break;
    case 2:
      core_mask = RKNN_NPU_CORE_2;
      break;
  }
  ret = rknn_set_core_mask(ctx_, core_mask);
  if (ret < 0) {
    KAYLORDUT_LOG_ERROR("rknn_set_core_mask failed! error code = {}", ret);
    return -1;
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version,
                   sizeof(rknn_sdk_version));
  if (ret < 0) {
    return -1;
  }
  KAYLORDUT_LOG_INFO("sdk version: {} driver version: {}", version.api_version,
                     version.drv_version);

  // Get Model Input Output Number
  rknn_input_output_num io_num;
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_query failed! error code = {}", ret);
    return -1;
  }
  KAYLORDUT_LOG_INFO("model input num: {}, and output num: {}", io_num.n_input,
                     io_num.n_output);
  // Get Model Input Info
  KAYLORDUT_LOG_INFO("input tensors:");
  rknn_tensor_attr
      input_attrs[io_num.n_input];  //这里使用的是变长数组，不建议这么使用
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("input rknn_query failed! error code = {}", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  // Get Model Output Info
  KAYLORDUT_LOG_INFO("output tensors:");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  // rk官方的segment模型有13个输出
  if (io_num.n_output == 13) {
    KAYLORDUT_LOG_INFO("this is a segment model")
    model_type_ = ModelType::SEGMENT;
  } else {
    model_type_ = ModelType::DETECTION;
  }
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("output rknn_query fail! error code = {}", ret);
      return -1;
    }
    dump_tensor_attr(&(output_attrs[i]));
    if (i == 2) {
      char *found = strstr(output_attrs[i].name, "angle");
      if (found != NULL) {
        KAYLORDUT_LOG_INFO("this is a OBB model");
        model_type_ = ModelType::OBB;
      }
      found = strstr(output_attrs[i].name, "kpt");
      if (found != NULL) {
        KAYLORDUT_LOG_INFO("this is a POSE model");
        model_type_ = ModelType::POSE;
      }
      found = strstr(output_attrs[i].name, "yolov10");
      if (found != NULL) {
        KAYLORDUT_LOG_INFO("this is a Yolov10 detection model");
        model_type_ = ModelType::V10_DETECTION;
      }
    }
  }
  // Set to context
  app_ctx_.rknn_ctx = ctx_;
  if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
      output_attrs[0].type == RKNN_TENSOR_INT8) {
    app_ctx_.is_quant = true;
  } else {
    app_ctx_.is_quant = false;
  }
  app_ctx_.io_num = io_num;
  app_ctx_.input_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(app_ctx_.input_attrs, input_attrs,
         io_num.n_input * sizeof(rknn_tensor_attr));
  app_ctx_.output_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(app_ctx_.output_attrs, output_attrs,
         io_num.n_output * sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    KAYLORDUT_LOG_INFO("model is NCHW input fmt");
    app_ctx_.model_channel = input_attrs[0].dims[1];
    app_ctx_.model_height = input_attrs[0].dims[2];
    app_ctx_.model_width = input_attrs[0].dims[3];
  } else {
    KAYLORDUT_LOG_INFO("model is NHWC input fmt");
    app_ctx_.model_height = input_attrs[0].dims[1];
    app_ctx_.model_width = input_attrs[0].dims[2];
    app_ctx_.model_channel = input_attrs[0].dims[3];
  }
  KAYLORDUT_LOG_INFO("model input height={}, width={}, channel={}",
                     app_ctx_.model_height, app_ctx_.model_width,
                     app_ctx_.model_channel);
  // 初始化输入输出参数
  inputs_ = std::make_unique<rknn_input[]>(app_ctx_.io_num.n_input);
  outputs_ = std::make_unique<rknn_output[]>(app_ctx_.io_num.n_output);
  inputs_[0].index = 0;
  inputs_[0].type = RKNN_TENSOR_UINT8;
  inputs_[0].fmt = RKNN_TENSOR_NHWC;
  inputs_[0].size =
      app_ctx_.model_width * app_ctx_.model_height * app_ctx_.model_channel;
  inputs_[0].buf = nullptr;
  return 0;
}

Yolov8::~Yolov8() { DeInit(); }

int Yolov8::DeInit() {
  if (app_ctx_.rknn_ctx != 0) {
    KAYLORDUT_LOG_INFO("rknn_destroy")
    rknn_destroy(app_ctx_.rknn_ctx);
    app_ctx_.rknn_ctx = 0;
  }
  if (app_ctx_.input_attrs != nullptr) {
    KAYLORDUT_LOG_INFO("free input_attrs");
    free(app_ctx_.input_attrs);
  }
  if (app_ctx_.output_attrs != nullptr) {
    KAYLORDUT_LOG_INFO("free output_attrs");
    free(app_ctx_.output_attrs);
  }
  return 0;
}

rknn_context *Yolov8::get_rknn_context() { return &(this->ctx_); }

int Yolov8::Inference(void *image_buf, object_detect_result_list *od_results,
                      letterbox_t letter_box) {
  TimeDuration total_duration;
  inputs_[0].buf = image_buf;
  int ret = rknn_inputs_set(app_ctx_.rknn_ctx, app_ctx_.io_num.n_input,
                            inputs_.get());
  if (ret < 0) {
    KAYLORDUT_LOG_ERROR("rknn_input_set failed! error code = {}", ret);
    return -1;
  }
  TimeDuration time_duration;
  ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      time_duration.DurationSinceLastTime());
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_run failed, error code = {}", ret);
    return -1;
  }
  for (int i = 0; i < app_ctx_.io_num.n_output; ++i) {
    outputs_[i].index = i;
    outputs_[i].want_float = (!app_ctx_.is_quant);
  }
  outputs_lock_.lock();
  ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output,
                         outputs_.get(), nullptr);
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_outputs_get failed, error code = {}", ret);
    return -1;
  }
  const float nms_threshold = NMS_THRESH;       // 默认的NMS阈值
  const float box_conf_threshold = BOX_THRESH;  // 默认的置信度阈值
  // Post Process
  // 把输出结果列表置零
  memset(od_results, 0, sizeof(object_detect_result_list));
  od_results->model_type = model_type_;
  KAYLORDUT_TIME_COST_INFO(
      "rknn_outputs_post_process",
      if (model_type_ == ModelType::SEGMENT) {
        post_process_seg(&app_ctx_, outputs_.get(), &letter_box,
                         box_conf_threshold, nms_threshold, od_results);
      } else if (model_type_ == ModelType::DETECTION ||
                 model_type_ == ModelType::V10_DETECTION) {
        post_process(&app_ctx_, outputs_.get(), &letter_box, box_conf_threshold,
                     nms_threshold, od_results);
      } else if (model_type_ == ModelType::OBB) {
        post_process_obb(&app_ctx_, outputs_.get(), &letter_box,
                         box_conf_threshold, nms_threshold, od_results);
      } else if (model_type_ == ModelType::POSE) {
        post_process_pose(&app_ctx_, outputs_.get(), &letter_box,
                          box_conf_threshold, nms_threshold, od_results);
      }
      /*else if (model_type_ == ModelType::V10_DETECTION) {
        post_process_v10_detection(&app_ctx_, outputs_.get(), &letter_box,
      box_conf_threshold, od_results);
      }*/
  );
  od_results->model_type = model_type_;

  // Remeber to release rknn outputs_
  rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output,
                       outputs_.get());
  outputs_lock_.unlock();
  auto total_delta = std::chrono::duration_cast<std::chrono::milliseconds>(
      total_duration.DurationSinceLastTime());
  KAYLORDUT_LOG_DEBUG("Inference time is {}ms and total time is {}ms",
                      duration.count(), total_delta.count());
  return 0;
}

int Yolov8::get_model_width() { return app_ctx_.model_width; }

int Yolov8::get_model_height() { return app_ctx_.model_height; }
