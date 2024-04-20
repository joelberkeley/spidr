df -H

dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 20

sudo apt-get purge -y \
    libnvinfer8 \
    libnvinfer-dev \
    libcudnn8 \
    libcudnn8-dev \
    libcublas-12-3 \
    libcublas-dev-12-3 \
    libcusparse-12-3 \
    libcusparse-dev-12-3 \
    libcusolver-12-3 \
    libcusolver-dev-12-3
sudo apt-get autoremove -y
sudo apt-get autoclean

df -H
