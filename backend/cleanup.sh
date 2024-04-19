dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 20
sudo docker image ls
