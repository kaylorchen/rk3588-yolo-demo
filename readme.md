
# Yolov8/v10 Demo for RK3588
The project is a multi-threaded inference demo of Yolov8 running on the RK3588 platform, which has been adapted for reading video files and camera feeds. The demo uses the Yolov8n model for file inference, with a maximum inference frame rate of up to 100 frames per second.

> If you want to test yolov8n with ros2 for yourself kit, click the [link](./yolov8n-ros2.md)

# Model
## Download Model File
you can find the model file in the 'src/yolov8/model', and some large files: 
Link: https://pan.baidu.com/s/1zfSVzR1G7mb-EQvs6A6ZYw?pwd=gmcs Password: gmcs   
Google Drive: https://drive.google.com/drive/folders/1FYluJpdaL-680pipgIQ1zsqqRvNbruEp?usp=sharing

## Model pt --> onnx
### For Yolov8 
go to my blog --> [blog.kaylordut.com](https://blog.kaylordut.com/2024/02/09/rk3588's-yolov8-model-conversion-from-pt-to-rknn/#more)
### For Yolov10
go to my another repository --> [yolov10](https://github.com/kaylorchen/yolov10)  
download pt model and export:
```bash
# End-to-End ONNX
yolo export model=yolov10n/s/m/b/l/x.pt format=onnx opset=13 simplify
```

## Model onnx --> rknn
go to my blog --> [blog.kaylordut.com](https://blog.kaylordut.com/2024/02/09/rk3588's-yolov8-model-conversion-from-pt-to-rknn/#more)
> TIPS: (Yolov10)
> - rknn-toolkit2(release:1.6.0) does not support some operators about attention, so it runs attention steps with CPU, leading to increased inference time. 
> - rknn-toolkit2(beta:2.0.0b12) has the attention operators for 3588, so I build a docker image, you can pull it from __**kaylor/rknn_onnx2rknn:beta**__

## Inference Time
Please refer to the spreadsheet '[8vs10.xlsx](./8vs10.xlsx)' for details.

|V8l-2.0.0|	V8l-1.6.0|	V10l-2.0.0|	V10l-1.6.0|	V8n-2.0.0	|V8n-1.6.0	|V10n-2.0.0|	V10n-1.6.0|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|133.07572815534|	133.834951456311|	122.992233009709|	204.471844660194|	17.8990291262136|	18.3300970873786|	21.3009708737864|	49.9883495145631|





# Demo Video and Guideline
https://space.bilibili.com/327258623?spm_id_from=333.999.0.0  
QQ group: 957577822

# Prepare

## Build the Cross-Compilation Environment
Set up a cross-compilation environment based on the following [link](https://github.com/kaylorchen/rk3588_dev_rootfs).

## Install Runtime Libraries in Your RK3588 Target Board
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
EOF
sudo mkdir /etc/apt/keyrings -pv
sudo wget -O /etc/apt/keyrings/kaylor-keyring.gpg http://apt.kaylordut.cn/kaylor-keyring.gpg
sudo apt update
sudo apt install kaylordut-dev libbytetrack
```
> If your OS is not Ubuntu22.04, and find [kaylordut-dev](https://github.com/kaylorchen/kaylordut) and [libbytetrack](https://github.com/kaylorchen/ByteTrack) sources in my github.


## Build the Project for Your RK3588

- Compile

```bash
git clone https://github.com/kaylorchen/rk3588-yolo-demo.git 
cd rk3588-yolo-demo/src/yolov8
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain-aarch64.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make 
```
> /path/to/toolchain-aarch64.cmake is .cmake file absolute path

- Run
  
``` bash

Usage: ./videofile_demo [--model_path|-m model_path] [--input_filename|-i input_filename] [--threads|-t thread_count] [--framerate|-f framerate] [--label_path|-l label_path]  

Usage: ./camera_demo [--model_path|-m model_path] [--camera_index|-i index] [--width|-w width] [--height|-h height][--threads|-t thread_count] [--fps|-f framerate] [--label_path|-l label_path]

Usage: ./imagefile_demo [--model_path|-m model_path] [--input_filename|-i input_filename] [--label_path|-l label_path]

```

> you can run the above command in your rk3588 



