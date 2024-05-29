# Demo Video
https://space.bilibili.com/327258623?spm_id_from=333.999.0.0
QQ group: 957577822

# Yolov8 Demo for RK3588
The project is a multi-threaded inference demo of Yolov8 running on the RK3588 platform, which has been adapted for reading video files and camera feeds. The demo uses the Yolov8n model for file inference, with a maximum inference frame rate of up to 100 frames per second.

## Prepare

### Build the Cross-Compilation Environment
Set up a cross-compilation environment based on the following [link](https://github.com/kaylorchen/rk3588_dev_rootfs).

### Install Runtime Libraries in Your RK3588 Target Board
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
EOF
sudo wget -O /etc/apt/keyrings/kaylor-keyring.gpg http://apt.kaylordut.cn/kaylor-keyring.gpg
sudo apt update
sudo apt install kaylordut-dev libbytetrack
```

### Build the Project and Deploy IT to Your RK3588

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

Usage: ./videofile_demo [--model_path|-m model_path] [--input_filename|-i input_filename] [--threads|-t thread_count] [--framerate|-f framerate] [--label_path|-l label_path]  
Usage: ./camera_demo [--model_path|-m model_path] [--camera_index|-i index] [--width|-w width] [--height|-h height][--threads|-t thread_count] [--fps|-f framerate] [--label_path|-l label_path]
Usage: ./imagefile_demo [--model_path|-m model_path] [--input_filename|-i input_filename] [--label_path|-l label_path]

> you can run the above command in your rk3588 

### Download Model File
you can find the model file in the 'src/yolov8/model', and some large files: 
Link: https://pan.baidu.com/s/1zfSVzR1G7mb-EQvs6A6ZYw?pwd=gmcs Password: gmcs 


