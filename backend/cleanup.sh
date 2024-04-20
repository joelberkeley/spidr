dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 20
sudo dpkg --remove libcudnn8 libcudnn8-dev libnvinfer8 libnvinfer-dev
sudo apt-get autoremove
sudo apt-get autoclean
