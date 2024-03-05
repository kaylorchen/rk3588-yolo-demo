//
// Created by kaylor on 3/4/24.
//

#include "yolov8.h"
#include "kaylordut/log/logger.h"
#include "kaylordut/time/time_duration.h"
#include "postprocess.h"

int read_data_from_file(const char *path, char **out_data) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    KAYLORDUT_LOG_INFO("fopen {} failed!", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  char *data = (char *) malloc(file_size + 1);
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

Yolov8::Yolov8(std::string &&model_path) {
  int model_len = 0;
  char *model;
  int ret = 0;
  model_len = read_data_from_file(model_path.c_str(), &model);
  if (model == nullptr) {
    KAYLORDUT_LOG_ERROR("Load model failed");
    return;
  }
  rknn_context ctx = 0;
  ret = rknn_init(&ctx, model, model_len, 0, NULL);
  free(model);
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_init failed! error code = {}", ret);
    return;
  }
  // Get Model Input Output Number
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_query failed! error code = {}", ret);
    return;
  }
  KAYLORDUT_LOG_INFO("model input num: {}, and output num: {}", io_num.n_input, io_num.n_output);
  // Get Model Input Info
  KAYLORDUT_LOG_INFO("input tensors:");
  rknn_tensor_attr
      input_attrs[io_num.n_input]; //这里使用的是变长数组，不建议这么使用
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("input rknn_query failed! error code = {}", ret);
      return;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  // Get Model Output Info
  KAYLORDUT_LOG_INFO("output tensors:");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("output rknn_query fail! error code = {}", ret);
      return;
    }
    dump_tensor_attr(&(output_attrs[i]));
  }
  // Set to context
  app_ctx_.rknn_ctx = ctx;
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

  init_post_process();
}

Yolov8::~Yolov8() {
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
  deinit_post_process();
}

int Yolov8::Inference(void *image_buf, object_detect_result_list *od_results, letterbox_t letter_box) {
  TimeDuration total_duration;
  auto input = std::make_unique<rknn_input[]>(app_ctx_.io_num.n_input);
  auto output = std::make_unique<rknn_output[]>(app_ctx_.io_num.n_output);
  input[0].index = 0;
  input[0].type = RKNN_TENSOR_INT8;
  input[0].fmt = RKNN_TENSOR_NHWC;
  input[0].size =
      app_ctx_.model_width * app_ctx_.model_height * app_ctx_.model_channel;
  input[0].buf = nullptr;
  static int count = 0;
  count++;
  input[0].buf = image_buf;
  int ret =
      rknn_inputs_set(app_ctx_.rknn_ctx, app_ctx_.io_num.n_input, input.get());
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
    output[i].index = i;
    output[i].want_float = (!app_ctx_.is_quant);
  }
  ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output,
                         output.get(), nullptr);
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_outputs_get failed, error code = {}", ret);
    return -1;
  }
  const float nms_threshold = NMS_THRESH;       // 默认的NMS阈值
  const float box_conf_threshold = BOX_THRESH;  // 默认的置信度阈值
  // Post Process
  post_process(&app_ctx_, output.get(), &letter_box, box_conf_threshold, nms_threshold,
               od_results);

  // Remeber to release rknn output
  rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output,
                       output.get());
  auto total_delta = std::chrono::duration_cast<std::chrono::milliseconds>(
      total_duration.DurationSinceLastTime());
  KAYLORDUT_LOG_DEBUG(
      "Inference time is {}ms and total time is {}ms, count = {}",
      duration.count(), total_delta.count(), count);
  return 0;
}