# The simplest way to test yolo with ROS2

# Install pacakge
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
EOF
sudo mkdir /etc/apt/keyrings -pv
sudo wget -O /etc/apt/keyrings/kaylor-keyring.gpg http://apt.kaylordut.cn/kaylor-keyring.gpg
sudo apt update
sudo apt install ai-framework ros-humble-yolo ros-humble-kaylordut-usb-cam
```

# Run

```bash
ros2 launch kaylordut_usb_cam test.launch.py
ros2 launch yolo yolo.launch.py
```