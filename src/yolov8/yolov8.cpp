//
// Created by kaylor on 3/4/24.
//

#include "yolov8.h"
#include "kaylordut/log/logger.h"

int read_data_from_file(const char *path, char **out_data) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    printf("fopen %s fail!\n", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  char *data = (char *) malloc(file_size + 1);
  data[file_size] = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != fread(data, 1, file_size, fp)) {
    printf("fread %s fail!\n", path);
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
  printf(
      "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, "
      "size=%d, fmt=%s, type=%s, qnt_type=%s, "
      "zp=%d, scale=%f\n",
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
  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  // Get Model Output Info
  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(output_attrs[i]));
  }

}
