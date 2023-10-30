#!/bin/bash
sudo systemctl stop lightdm jottad docker containerd
sudo swapoff -a
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -u 2800MHz
sudo cpupower frequency-set -d 2800MHz
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
#sudo cpupower idle-set -D 0
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
sudo chmod -R a+r /sys/class/powercap/intel-rapl
