#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3-pip 

# Ubuntu 24.04(Python 3.12)부터는 시스템 보호를 위해 전역 pip 설치 시 --break-system-packages 옵션이 필수입니다.
pip install opencv-python pyserial --break-system-packages 
pip install ultralytics==8.2.69 --break-system-packages
# 주의: Python 3.12에서는 구버전 setuptools(58.2.0) 설치가 실패할 수 있습니다. 오류 시 버전 지정 없이 설치하세요.
pip install setuptools==58.2.0 --break-system-packages 
pip install pynput --break-system-packages

# ROS 2 Jazzy부터는 기존 Gazebo Classic이 완전히 단종되고 새로운 Gazebo(Harmonic)로 대체되었습니다.
# 관련 패키지들을 ros-jazzy-ros-gz 생태계에 맞게 수정했습니다.
sudo apt install ros-jazzy-xacro
sudo apt install ros-jazzy-ros-gz
sudo apt install ros-jazzy-ros-gz-sim
sudo apt install ros-jazzy-ros-gz-bridge
sudo apt install ros-jazzy-ros-gz-interfaces

pip install transformers --break-system-packages
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
pip install huggingface_hub --break-system-packages


# (참고) 새로운 Gazebo는 모델 디렉토리로 ~/.gz/models 를 사용하지만, 
# 기존 패키지 소스가 ~/.gazebo 에 의존할 수 있으므로 기존 폴더 생성 및 복사 로직은 유지했습니다.
GAZEBO_DIR="/home/$(whoami)/.gazebo"
if [ -d "$GAZEBO_DIR" ]; then
    echo "" # .gazebo 폴더가 존재하는지 확인
else
    echo ".gazebo 폴더가 존재하지 않아 생성합니다."
    mkdir "$GAZEBO_DIR"
fi
MODELS_DIR="$GAZEBO_DIR/models"
if [ -d "$MODELS_DIR" ]; then
    echo "" # .gazebo/models 폴더가 존재하는지 확인
else
    echo "models 폴더가 존재하지 않아 생성합니다."
    mkdir "$MODELS_DIR"
fi


# 패키지 폴더의 모든 내용을 .gazebo/models 폴더로 복사
SOURCE_MODELS_DIR="/home/$(whoami)/ros2_autonomous_vehicle_simulation/src/simulation_pkg/models"
if [ -d "$SOURCE_MODELS_DIR" ]; then
    echo "$SOURCE_MODELS_DIR의 내용을 $MODELS_DIR로 복사합니다."
    cp -r "$SOURCE_MODELS_DIR/"* "$MODELS_DIR/"
else
    echo "$SOURCE_MODELS_DIR 폴더가 존재하지 않습니다."
fi



# .gazebo/models 폴더 안의 etc 폴더 삭제
if [ -d "$MODELS_DIR/etc" ]; then
    #echo "$MODELS_DIR/etc 폴더를 삭제합니다."
    rm -rf "$MODELS_DIR/etc"
else
    echo "$MODELS_DIR/etc 폴더가 존재하지 않습니다."
fi



# .bashrc에 기본 설정 추가
BASHRC_FILE="$HOME/.bashrc"

add_alias() {
    local alias_cmd="$1"
    if ! grep -q "$alias_cmd" "$BASHRC_FILE"; then
        echo "$alias_cmd" >> "$BASHRC_FILE"
        echo "'$alias_cmd'가 추가되었습니다."
    else
        echo "'$alias_cmd'가 이미 존재합니다."
    fi
}

if ! grep -q "export ROS_DOMAIN_ID=" "$BASHRC_FILE"; then
    echo "export ROS_DOMAIN_ID=0 # 0~232 사이의 숫자로 변경" >> "$BASHRC_FILE"
    echo "ROS_DOMAIN_ID 설정이 추가되었습니다."
else
    echo "ROS_DOMAIN_ID 설정이 이미 존재합니다."
fi

add_alias "alias MOVE='ros2 service call /go std_srvs/srv/SetBool \"{data: true}\"'"
add_alias "alias STOP='ros2 service call /go std_srvs/srv/SetBool \"{data: false}\"'"

if ! grep -q "qqq()" "$BASHRC_FILE"; then
    cat << 'EOF' >> "$BASHRC_FILE"
qqq() {
    # 기존 Gazebo Classic(gzserver)과 새로운 Gazebo(ruby/gz sim) 프로세스 모두 종료
    PIDS=$(ps aux | grep -E '[g]zserver|[r]uby.*gz' | awk '{print $2}')

    for pid in $PIDS; do
        kill -9 $pid
    done
}
EOF
    echo "qqq 함수가 추가되었습니다."
else
    echo "qqq 함수가 이미 존재합니다."
fi

add_alias "alias bashrc='gedit ~/.bashrc'"
add_alias "alias ㅠㅁ녹ㅊ='gedit ~/.bashrc'"
add_alias "alias bashup='source ~/.bashrc'"
add_alias "alias ㅠㅁ노ㅕㅔ='source ~/.bashrc'"
add_alias "alias c='clear'"
add_alias "alias ㅊ='clear'"
add_alias "alias rma='rm -rf'"

echo "변경된 .bashrc를 적용합니다."
source "$HOME/.bashrc"
