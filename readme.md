# Yolov8 Demo for RK3588
The project is a multi-threaded inference demo of Yolov8 running on the RK3588 platform, which has been adapted for reading video files and camera feeds. The demo uses the Yolov8n model for file inference, with a maximum inference frame rate of up to 100 frames per second.

## Prepare

### Build the Cross-Compilation Environment
Set up a cross-compilation environment based on the following [link](https://github.com/kaylorchen/rk3588_dev_rootfs).

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

> you can run the above command in your rk3588 

