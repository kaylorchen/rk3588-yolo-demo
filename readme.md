
# 写在前面的话
如果你看到这个仓库，证明你想试试这个多线程的推理。
1. 这个里的代码不是最优的，有些错序的问题需要其他手段解决，我没有在这里解决，你可以看下一个标题新版本仓库的链接。新仓库解决了这个问题
2. 本仓库的代码思路想法，在我的B站上有详细的讲解，需要理解程序的可以去b站搜我“kaylordut”
3. 项目合作的可以发邮件到kaylor.chen@qq.com, 邮件请说明来意，和简单的需求，以及你的预算。邮件我一般都回复，请不要一来就索要微信，一个切实可行的项目或者良好的技术交流是良好的开始。

如果你缺一个运行改代码的开发板，[访问这里](https://github.com/HuntersRobotics/ai_hunter_guideline/blob/main/96-munual/AI%20Hunter%E4%BA%A7%E5%93%81%E8%AF%A6%E7%BB%86%E4%BB%8B%E7%BB%8D.md)  
If you need a development board to run this code, [visit here](https://github.com/HuntersRobotics/ai_hunter_guideline/blob/main/96-munual/AI%20Hunter%E4%BA%A7%E5%93%81%E8%AF%A6%E7%BB%86%E4%BB%8B%E7%BB%8D.md)


# New Project
An inference framework compatible with TensorRT, OnnxRuntime, NNRT and RKNN  
If you want to find some more yolo8/yolo11 demo and depth anything demo, visit my another [repository](https://github.com/kaylorchen/ai_framework_demo)

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
QQ group1: 957577822 (full)
QQ group2: 546943464

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



