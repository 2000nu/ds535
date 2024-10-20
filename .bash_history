cd tee4ehr/
docker build -t tee4ehr_retweets_mc .
cd ../..
cd ..
l
cd data
mkdir seminars
l
cd seminars/
l
cat /etc/group
chmod 777 /data/seminars
sudo chmod 777 /data/seminars
ls -l
sudo chmod u+rwx /data/seminars
ls -l
cd ..
l
cd seminars
cd ..
l
cd samsung-project/
ls -l
cd data-20240715/
ls -l
sudo chmod g+rwx /data/seminars
ls -l
cd ../..
l
cd seminars/
l
ls -l
sudo chmod g-wx /data/seminars
l
cd
l
cd tee4ehr/
docker build -t tee4ehr_retweets_mc .
docker ps
docker ps -a
docker images
nvidia-smi
docker run --gpus '"device=0"' -it --rm --name tee4ehr_retweets_mc_container tee4ehr_retweets_mc
nvidia-smi
docker imaegs
docker images
docker ps
docker ps -a
docker images
docker rmi tee4ehr_retweets_mc:latest 
l
docker images
docker build -t tee4ehr .
docker ps
docker ps -a
docker images
nvidia-smi
docker run --gpus '"device=0"' -it --rm --name tee4ehr_container tee4ehr
ls -l
chmod +x start.sh
ls -l
docker images
docker run --gpus '"device=0"' -it --rm --name tee4ehr_container tee4ehr
ls -l
docker images
docker rmi tee4ehr:latest 
docker ps -a
docker build -t tee4ehr .
docker ps -a
docker images
docker run --gpus '"device=0"' -it --rm --name tee4ehr_container tee4ehr
docker rmi tee4ehr:latest 
l
docker build -t tee4ehr .
docker images
nvidia-smi
docker run --gpus '"device=0"' -it --rm --name tee4ehr_container tee4ehr
nvidia-smi
vim /home/cheonwoolee/tee4ehr/data/retweets_mc/train.pkl
nvidia-
nvidia-smi
nvidia-
nvidia-smi
ㅣ
l
ls -k
ls -l
du
du -sh
cd ../../data
l
cd seminars
ls -l
cd
nvidia-smi
docker images
nvidia-smi
stat --help
stat /proc/2726087
ps -p 2726087 =o lstart
ps -p 2726087 -o lstart
nvidia-smi
htop
top
free --help
free -h
nvidia-sm
nvidia-smi
docker images
docker ps
docker stop tee4ehr_container 
docker ps -a
docker ps
docker images
docker rmi tee4ehr:latest 
docker images
cd tee4ehr/
l
ls -l
docker build -t tee4ehr .
docker images
docker run --gpus '"device=0"' -it --name tee4ehr_container --rm tee4ehr
docker images
nvidia-smi
ps
ps -a
nvidia-smi
docker images
docker ps
docker exec -it tee4ehr_container /bin/bash
docker ps -a
nvidia-smi
docker ps
nvidia-smi
ls -l
cat /etc/groups
cat /etc/group
nvidia-smi
pstree -aps 2887087 | grep id
docker ps | grep 6c2e9c
nvidia-smi --help
nvidia-smi
conda --version
l
rm -r docker-practice
ls -l
clear
cd docker-practice/
docker build -t cheonwoolee-docker-practice .
docker run -it --name "cheonwoolee-docker-practice" cheonwoolee-docker-practice python main.py
docker rm cheonwoolee-docker-practice 
docker rmi cheonwoolee-docker-practice:latest 
docker ps -a
docker images
l
docker ps
docker stop 6c2e9c43cb06
docker start 6c2e9c43cb06
docker ps -a
docker images
docker tag tee4ehr:latest cheonwoolee_tee4ehr:latest
docker images
docker ps
docker ps -a
docker images
docker rmi tee4ehr:latest 
docker images
docker ps -a
nvidia-smi
docker images
docker rmi cheonwoolee_tee4ehr:latest 
docker images
docker ps -a
docker build -t cheonwoolee-tee4ehr .
cd tee4ehr/
docker build -t cheonwoolee-tee4ehr .
docker images
docker run -gpus '"devices=0"' -it --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash start.sh
docker run --gpus '"devices=0"' -it --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash start.sh
docker run --gpus '"device=0"' -it --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash start.sh
docker images
docker rmi cheonwoolee-tee4ehr:latest 
docker ps -a
docker ps
docker rm cheonwoolee-tee4ehr-supervised 
docker rmi cheonwoolee-tee4ehr:latest 
docker build -t cheonwoolee-tee4ehr .
l
ls -l
docker images
cd ..
df -h
docker images
docker ps -a
cd ..
cd
du -h
du -sh
ls -l
df -h
docker images
docker rmi cheonwoolee-tee4ehr:latest 
df -h
sudo du -ahx / | sort -rh | head -n 20
sudo du -sh /*
sudo du -sh /var/*
sudo du 0sh /var/lib/*
nvidia-smi
du -sh
df -h
df .
du -a /dev/nvme1n1p2 | sort -n -r | head -n 20
df -h
du -h --max-depth=1 /var/lib
docker ps -a
docker images
du -h /var/lib
du -h /var
du -h
df -h
du -sh /var
du -sh /*
sudo du -sh /*
sudo du -sh /var/*
sudo du -sh /var/lib
sudo du -sh /var/lib/*
sudo du -sh /var/lib/docker/*
sudo du -sh /var/lib/docker/
docker system prune -a
cd /var/lib/docker
sudo cd /var/lib/docker
sudo -i
docker images -a
docker builder prune
du -sh
df -h
nvidia-smi
nvidia images
docker images
docker ps -a
ls -l
cd tee4ehr/
ls -l
chmod +x supervised.sh
ls -l
docker build -t cheonwoolee-tee4ehr .
docker images
docker run --gpus='"device=0"' -it --name cheonwoolee-tee4ehr-supervised cheonwoolee-tee4ehr bash supervised.sh
docker run --gpus='"device=0"' -it --name cheonwoolee-tee4ehr-supervised cheonwoolee-tee4ehr bash p19.sh
docker ps 
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
l
docker run --gpus='"device=0"' -it --name cheonwoolee-tee4ehr-supervised cheonwoolee-tee4ehr bash p19.sh
docker run --gpus='"device=0"' -it --name cheonwoolee-tee4ehr-supervised cheonwoolee-tee4ehr bash ./p19.sh
docker rm cheonwoolee-tee4ehr-supervised 
docker run --gpus='"device=0"' -it --name cheonwoolee-tee4ehr-supervised cheonwoolee-tee4ehr bash ./p19.sh
docker rm cheonwoolee-tee4ehr-supervised 
docker images
docker ps -a
docker run -it --gpus='"device=0"'   -v $(pwd)/supervised.sh:/home/cheonwoolee/tee4ehr/supervised.sh   tee4ehr   bash /home/cheonwoolee/tee4ehr/supervised.sh \
docker run -it --gpus='"device=0"'   -v $(pwd)/supervised.sh:/home/cheonwoolee/tee4ehr/supervised.sh --name "cheonwoolee-tee4ehr-supervised"  cheonwoolee-tee4ehr   bash /home/cheonwoolee/tee4ehr/supervised.sh
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker run -it --gpus='"device=0,1"'   -v $(pwd)/supervised.sh:/home/cheonwoolee/tee4ehr/supervised.sh --name "cheonwoolee-tee4ehr-supervised"  cheonwoolee-tee4ehr   bash /home/cheonwoolee/tee4ehr/supervised.sh
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker run -it --gpus='"device=0,1"' -v $(pwd)/Main.py:/home/cheonwoolee/tee4ehr/Main.py --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash /home/cheonwoolee/tee4ehr/supervised.sh
docker rm cheonwoolee-tee4ehr-supervised 
docker run -it --gpus='"device=0,1"'   -v $(pwd)/Main.py:/home/cheonwoolee/tee4ehr/Main.py   -v $(pwd)/supervised.sh:/home/cheonwoolee/tee4ehr/supervised.sh   --name "cheonwoolee-tee4ehr-supervised"   cheonwoolee-tee4ehr   bash /home/cheonwoolee/tee4ehr/supervised.sh
scp cheonwoolee@como1.kaist.ac.kr:/home/cheonwoolee/tee4ehr/Main.py /Users/2000nu/Downloads
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker run -it --gpus='"device=0,1"'   -v $(pwd)/Main.py:/home/cheonwoolee/tee4ehr/Main.py   -v $(pwd)/supervised.sh:/home/cheonwoolee/tee4ehr/supervised.sh   --name "cheonwoolee-tee4ehr-supervised"   cheonwoolee-tee4ehr   bash /home/cheonwoolee/tee4ehr/supervised.sh
docker ps -a
df --help
df -h
docker run -it --gpus='"device=0"'   -v $(pwd)/Main.py:/home/cheonwoolee/tee4ehr/Main.py   -v $(pwd)/supervised.sh:/home/cheonwoolee/tee4ehr/supervised.sh   --name "cheonwoolee-tee4ehr-supervised"   cheonwoolee-tee4ehr   bash /home/cheonwoolee/tee4ehr/supervised.sh
nvidia-smi
docker stop cheonwoolee-tee4ehr-supervised
nvidia-smi
docker stop cheonwoolee-tee4ehr-supervised 
docker rm cheonwoolee-tee4ehr-supervised 
nvidia-smi
ㅣㄴ-ㅣ
ls -l
du -h
python main.py
python check_timestamp.py 
ls -l
python plot_per_length.py 
exit
conda activate cheonwoolee-wandb
pip install wandb
python wandb_run.py
python wandb_check/wandb_run.py
python wandb_check/wandb_check.py
pip install numpy
python wandb_check/wandb_check.py
conda deactivate
conda env list
clear
nvidia-smi
python main.py
python
python plot_per_length.py 
python plot_multivar.py
python check_nan.py
python plot_multivar.py
python check_nan.py
python plot_multivar.py
python check_nan.py
python plot_multivar.py
python check_nan.py
python check_interval.py
python check_nan.py
python plot_multivar.py 
python check_unique.py 
python check_timestamp.py
python check_unique.py
clear
python check_nan.py
docker images
docker images -f "dangling=true"
docker images --filter since=7284c2248969
docker images | grep "<none>"
docker inspect 7284c2248969
parent_image_id="7284c2248969"
docker images -q | while read image_id; do     docker inspect --format='{{.Id}} {{.Parent}}' "$image_id" | grep "$parent_image_id"; done
docker history 7284c2248969
docker images -a
docker images -f "dangling=true"
docker images -af "dangling=true"
docker image prune
docker images -a
docker inspect f9eb6222b712
docker images
docker images -a
clear
python check_nan.py
python check_timestamp.py
python check_timestamp_length.py 
python check_unique.py 
python plot_multivar.py 
python check_missing_rate.py
python plot_multivar.py 
python check_missing_rate.py
python plot_multivar.py 
python check_timestamp_length.py 
python plot_multivar.py 
scp ./multivar/EA0_CA_LB106_S12.png /Users/2000nu/Downloads
scp cheonwoolee@como1.kaist.ac.kr:multivar/EA0_CA_LB106_S12.png /Users/2000nu/Downloads
scp cheonwoolee@como1.kaist.ac.kr:/home/cheonwoolee/samsung-project/multivar/EA0_CA_LB106_S12.png /Users/2000nu/Downloads
python plot_multivar.py 
l
cd ../../data/seminars
l
ls -l
cd ../..
l
cd etc
l
cd docker
l
df -h
docker system df
cd ../../mnt
l
cd ssd1
l
cd lost+found/
ls -l
cd ..
ls -l
cat /etc/groups
cat /etc/group
docker ps
docker logs -f cheonwoolee-tee4ehr-supervised 
docker ps
docker ps -a
docker images
nvidia-smi
docker ps
docker top
docker top --help
docker container top
docker container cheonwoolee-tee4ehr-supervised
docker top --help
docker top cheonwoolee-tee4ehr-supervised 
docker images
docker run --gpus='"device=0"' -it --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash supervised.sh
python wandb_run.py 
conda --help
conda env list
conda craete --name cheonwoolee-wandb python=3.8
conda create --name cheonwoolee-wandb python=3.8
conda activate cheonwoolee-wandb
conda init
cd ..
docker images
docker rmi cheonwoolee-tee4ehr:latest 
docker ps
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker rmi cheonwoolee-tee4ehr:latest 
docker build -t cheonwoolee-tee4ehr .
docker images
nvidia-smi
conda deactivate
conda config --set auto_activate_base false
docker images
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker run -it --gpus='"device=0"' -v /home/cheonwoolee/tee4ehr/Main.py:/home/cheonwoolee/tee4ehr/Main.py --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash supervised.sh
conda activate cheonwoolee-wandb
python wandb_check.py
conda deactivate
docker run -it --gpus='"device=0"' -v /home/cheonwoolee/tee4ehr/Main.py:/home/cheonwoolee/tee4ehr/Main.py --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash supervised.sh
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker run -it --gpus='"device=0"' -v /home/cheonwoolee/tee4ehr/Main.py:/home/cheonwoolee/tee4ehr/Main.py --name "cheonwoolee-tee4ehr-supervised" cheonwoolee-tee4ehr bash supervised.sh
conda activate cheonwoolee-wandb
python wandb_check.py
conda deactivate
docker ps -a
docker rm cheonwoolee-tee4ehr-supervised 
docker system df
docker images
docker system df -v
docker system df
docker images
docker images -a
docker images
docker system df
df
df -h
docker system df
df -h
docker ps -a
docker ps
docker images
git --help
git remote -v
l
git init
git remote -v
git remote add origin https://github.com/2000nu/ISTS.git
git remote -v
git add .
git commit -m "initial setting"
git config 
git config --list
git config --global user.name "cheonwoolee"
git config --global user.email "dlcjsdn07@kaist.ac.kr"
git config --global user.name "cheonwoolee"
git config --list
git commit -m "initial setting"
git remote -v
git push -u origin master
python -v
python --version
pip list
conda lsit
python check_missing_rate.py 
python plot_multivar.py 
python main.py
python tmp.py
nvidia-si
nvidia-smi
docker system df
docker images
docker images -a
l
ls -l
python
git list
git remote list
cd samsung-project/
python main.py
cd
cd ../../data/seminars/
l
ls -l
cd
docker system df
df -h
docker images
docker ps -a
python
docker ps -a
docker images -a
docker images
docker images "dangling=true"
docker images -a
docker images
docker rmi 9c6663c20306
docker images
docker rmi 31d38c162b25
docker rmi 554a0001eb90
docker images
docker system df
df- h
df -h
cd /mnt/ssd1
ls -l
cd
docker build --help
docker builder --help
docker builder ls
docker builder l
nvidia-smi
df -h
nvidia-smi
cd ../../../..
ls
cd data
l
cd taewookham/
l
cd MUTAG/
l
cd processed/
l
cd ../..
cd ..
l
mkdir cheonwoolee
l
cd cheonwoolee/
l
mv tee4ehr-data/ tee4ehr/
l
cd tee4ehr/
l
cd
cd tee4ehr/
clear
git status
git checkout --help
git status
git checkout -b intensity-encoding
git add transformer/Models.py 
git push origin intensity-encoding 
docker build -t "cheonwoolee-tee4ehr" .
docker run --gpus='"device=0"' -it -v /data/cheonwoolee/tee4ehr/:/app/data --name "cheonwoolee-tee4ehr-test1" cheonwoolee-tee4ehr bash supervised.sh
docker run --gpus='"device=0"' -it -v /data/cheonwoolee/tee4ehr/:/app/data -v /home/cheonwoolee/tee4ehr/transformer/Models.py:/app/transformer/Models.py --name "cheonwoolee-tee4ehr-test1" cheonwoolee-tee4ehr bash supervised.sh
conda activate cheonwoolee-wandb
python wandb-check.py
python wandb_check.py
conda deactivate
clear
python
clear
nvidia-smi
conda activate cheonwoolee-tee4ehr
clear
python tmp.py
conda deactivate
docker ps -a
df -h
docker images 
nvidia-smi
clear
l
cd ..
ls
mv tee4ehr-data/ ../../data/cheonwoolee/
l
cd
cd tee4ehr/
clear
nvidia-smi
docker images
docker ps -a
docker rmi cheonwoolee-tee4ehr:latest 
docker images
docker system df
df -h
docker system df
git add .
git commit -m "change data directory"
git push origin intensity-encoding 
docker system df
df -h
cat /etc/group
nvidia-smi
docker stop cheonwoolee-tee4ehr-test1 
docker ps -a
docker rm cheonwoolee-tee4ehr-test1 
nvidia-smi
df -h
docker system df
docker ps -a
docker rm cheonwoolee-tee4ehr-test1 
docker system df
nvidia-smi
clear
docker system df
docker images
docker system df
df -h
docker system df
nvidia-smi
docker system df
docker ps -a
docker images
docker rmi cheonwoolee-tee4ehr:latest 
docker iamges
docker images
docker ps -a
docker system df
docker images
docker system df
docker ps -a
python
python tmp.py
conda --help
conda env list
conda activate cheonwoolee-wandb
conda deactivate
conda create --name cheonwoolee-tee4ehr --clone cheonwoolee-wandb
conda remove --name cheonwoolee-wandb --all
conda env list
conda activate cheonwoolee-tee4ehr
conda install pandas
python tmp.py
conda update pandas
python tmp.py
conda install pandas<2.0.0
conda install "pandas<2.0.0"
nvidia-smi
conda deactivate
df -h
conda activate cheonwoolee-tee4ehr
python tmp.py
python Main.py
conda install matplotlib
python Main.py
conda install torch
python Main.py
python tmp.py
conda deactivate
cd
clear
cd docker-practice/
docker build -t cheonwoolee-toy-example .
docker run -v '$(pwd)/data.csv:/app/data.csv' -it --rm cheonwoolee-toy-example
docker run -v "$(pwd)/data.csv:/app/data.csv" -it --rm cheonwoolee-toy-example
docker run -v "$(pwd)/data.csv:/app/data.csv" -it --rm cheonwoolee-toy-example python main.py
python
python main.py
docker run -v "$(pwd)/data.csv:/app/data.csv" -it --rm cheonwoolee-toy-example python main.py
docker run -v "$(pwd)/data.csv:/app/data.csv" -v "$(pwd)/main.py:/app/main.py"  -it --rm cheonwoolee-toy-example python main.py
docker imags
docker images
docker rmi cheonwoolee-toy-example:latest 
docker images
docker build -t cheonwoolee-toy-example .
docker run -v "$(pwd)/data.csv:/app/data.csv" -v "$(pwd)/main.py:/app/main.py"  -it --rm cheonwoolee-toy-example python main.py
cat /data
cat /data/
cat /etc/group
docker system df -v
docker system df --help
docker system df --format 'table'
docker system df --format 'table TEMPLATE'
docker system df --format 'json'
docker system df
clear
docker system df
docker builder --help
docker builder ls
docker images
docker rmi cheonwoolee-toy-example:latest 
docker images -a
docker system df
docker ps -a
df -h
conda env list
du -sh 
du -h
du -sh $(conda info --base)/envs/*
cd ..
cd .conda/
ls -l
du -h
du -sh
docker system df
df -h
cd
htop
df -h
cd /data
ls -l
du -h
du -s
df -h
cd 
cd /run
ls -l
cd
lsblk
df -hT
top
free -h
lscpu
free -h
htop
df -h
systemctl list-units --type=service
cat /etc/group
cd /data
l
cd cheonwoolee/
l
cd tee4ehr/
l
rm -r retweets_mc
rm -r retweets_ml
rm -r so
rm -r synthea_100
l
cd ..
du -h
cd
df -h
docker system df
docker commit --help
df -h
du -h
du -sh
cd ..
du -sh
cd
clear
docker system df
docker ps -a
df -sh
df -h
docker images -a
docker images
docker system df
python
docker system df
docker images
python
cd
celar
clear
df -h
docker system df
docker builder prune --help
docker system df
df -h
docker builder prune --filter "until=168h"
df -h
docker system df
docker ps -a
nvidia-smi
docker ps -a
docker images
docker system df
df- h
df -h
docker system df
docker images
docker ps -a
nvidia-smi
du -h
ds -h
df -h
l
nvidia-smi
df -h
scp -r /data/seminars /Users/2000nu/Downloads
scp -r cheonwoolee@como1.kaist.ac.kr:/data/seminars /Users/2000nu/Downloads
nvidia-smi
docker system df
docker ps -a
docker images
docker system df
docker image prune -f
docker ps -a
docker system df
docker images
df -h
df- h
docker system df
df -h
nvidia-smi
df -h
cd samsung-project/
python main.py
nvidia-smi
docker ps -a
df -h
cd ../..
l
cd
l
cd
cd /data/seminars
ls -l
clear
cd
df -h
git clone https://github.com/ojus1/Time2Vec-PyTorch.git
ls
cd Time2Vec-PyTorch/
l
python3 experiment.py
python --version
conda create -n time2vec python=3.6
conda list
conda env list
conda activate time2vec
conda install pytorch=1.1.0
conda install pytorch=1.1.0 torchvision=0.3.0 -c pytorch
python3 experiments.py
ls
python3 experiment.py
conda install pandas
python3 experiment.py
git remote -v
git remote set-url origin https://github.com/2000nu/time2vec.git
git remote -v
python3 experiment.py
conda install matplotlib
python3 experiment.py
conda deactivate
df -h
docker system df
df -h
docker images
docker ps -a
du -sh $(conda info --base)/envs/your_env_name
du -sh $(conda info --base)/envs
du -h $(conda info --base)/envs
du -h $(conda info --base)
nvidia-smi
df -h
docker ps -a
clear
conda activate cheonwoolee-tee4ehr
cd tee4ehr-practice/
l
nvidia-smi
CUDA_VISIBLE_DEVICES=7 bash supervised.sh
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=7 bash supervised.sh
cd /data/
ls
cd test
l
vim test.txt
df -h
df -h /mnt/ssd1
cat /sys/block/sda/queue/rotational
fdisk -l
fdisk --help
sudo fdisk -l
lsblk
cd /mnt/ssd1
l
cd lost+found/
sudo ls lost+found/
sudo cd lost+found/
sudo /
cd /
ls
cd lost+found/
cd
df -h
nvidia-smi
df -h
nvidia-smi
cd tee4ehr-practice/
conda env list
conda activate cheonwoolee-tee4ehr
bash supervised.sh
cd /data/cheonwoolee
l
cd tee4ehr/
l
conda activate cheonwoolee-tee4ehr
CUDA_VISIBLE_DEVICES=7 bash supervised.sh
echo $CUDA_VISIBLE_DEVICES
bash supervised.sh
conda deactivate
l
conda activate cheonwoolee-tee4ehr
cd tee4ehr-practice/
bash supervised.sh
quit
conda activate cheonwoolee-tee4ehr
cd tee4ehr-practice/
bash supervised.sh
conda activate cheonwoolee-tee4ehr
cd tee4ehr-practice/
bash supervised.sh
cd ../tee4ehr
bash supervised.sh
ps ps aux | grep wandb
cd tee4ehr
conda activate cheonwoolee-tee4ehr
bash supervised.sh
nvidia-smi
top
nvidia-smi
nsight --help
htop
watch -n -1 nvidia-smi
grep -i processor /proc/cpuinfo
nvidia-smi
watch -n -1 nvidia-smi
top -bn1 | grep "Cpu(s)" | awk '{printf("CPU 사용률 : %.1f%%\n", 100 - $8)}'
vmstat 1
top
ps -eo user,%cpu --sort=-%cpu | awk '{a[$1]+=$2} END {for (i in a) print i, a[i]}' | sort -k2 -nr
ps -u --help
ps -u cheonwoolee
lsof -p 2944423
cat /proc/2944423/cmdline
cd /data/cheonwoolee
cd tee4ehr/
l
du -s
du -sh
du -h
find --help
find . --help
find . -type d -name '*nextmark-concat*' -exec rm -rf {} +
ls -l
cd ../..
ls
ls -l
cd cheonwoolee/
ls -l
cd tee4ehr/
ls -l
sudo chmod -R u+w .
ls -l
find . -type d -name '*nextmark-concat*' -exec rm -rf {} +
ㅣ -ㅣ
ls -l
sudo chmod -R u+w p12-raindrop/ p19-raindrop/
ls -l
find . -type d -name '*nextmark-concat*' -exec rm -rf {} +
sudo find . -type d -name '*nextmark-concat*' -exec rm -rf {} +
du -h
conda activate cheonwoolee-tee4ehr
bash supervised.sh
ps -p 2946049 --help
ps -p 2946049 -o pid,ppid,cmd,%mem,%cpu,etime
ps -p 2944423 -o pid,ppid,cmd,%mem,%cpu,etime
ps
ps -ef
ps -aux
pstree
ps -o
ps --help o
watch -n -10 nvidia-smi
df -h
nvidia-smi
htop
ps -eo pid,comm,%cpu --sort=-%cpu | head
top
nvidia-smi
top
lsof -p 2944423
kill 2944423
kill 2946049
kill 2946110
kill 2919597
top
df -h
clear
python main.py
top
ls
ls -l
pwd
l
top
ps -ef | grep vscode
ls -l
ls -la
ps aux | grep vscode-server | grep cheonwoolee
kill -9 1964325
kill -9 1964335
kill -9 1964522
ps aux | grep vscode-server | grep cheonwoolee
ls -la
rm -rf ~/.vscode-server
ls -la
top
clear
cd /data/cheonwoolee/tee4ehr
du -h
cd
cd tee4ehr
conda activate cheonwoolee-tee4ehr
python metrics_summary.py 
conda deactivate
top
clear
cd tee4ehr/
python metrics_summary.py 
conda activate cheonwoolee-tee4ehr
python metrics_summary.py 
nvidia-smi
df -h
cd /boot/efi
cd /boot
ls -l
cd
cd tee4ehr/
python
nvidia-smi
watch -n 1 nvidia-smi
watch -n 1 nvidia-smi -i 0
nvidia-smi
ps -o user= -p 3345382
kill -9 3345382
nvidia-smi
nvidia-smi
py-spy --help
watch -n 1 nvidia-smi 
nvidia-smi
watch -n 1 nvidia-smi 
ps -p 3268750 -o user,cmd
nvidia-smi
kill -9 3268750
nvidia-smi
ps -p 3285090 -o user,cmd
clear
nvidia-smi
conda activate cheonwoolee-tee4ehr
cd tee4ehr/
bash supervised.sh
df -h
docker system df
df -h
docker images
nvidia-smi
ps aux
top
nvidia-smi
nvidia-smi -i 7
nvidia-smi -i 7 --query-compute-apps=pid,process_name,used_memory --format=csv
ps -o user= -p 2317028
ps -o user= -p 2328452
ps -o cmd= -p 2328452
nvidia-smi
kill 2328452
nvidia-smi
kill -9 2328452
nvidia-smi
ps -o user= -p 2322245
ps -o user= -p 2332245
nvidia-smi
watch -n 1 nvidia-smi
watch -n 1 nvidia-smi -i 7
watch -n 1 nvidia-smi -i 0
nvidia-smi
nvidia-smi
df -h
conda activate cheonwoolee-tee4ehr
cd tee4ehr
bash supervised.sh
pip install tensorflow==1.14.0 tensorflow-probability==0.7.0
pip install tensorflow
bash supervised.sh
pip install tensorflow=1.14.0
pip uninstall tensorflow
bash supervised.sh
ls /data/cheonwoolee/tee4ehr/
ls /data/cheonwoolee/tee4ehr/p12-raindrop/
ls /data/cheonwoolee/tee4ehr/p12-raindrop/split0
ls
python summary.py
python summary.py -name "abc"
python summary.py -name "abc-def"
bash supervised.sh
nvidia-smi
df -h
conda activate cheonwoolee-tee4ehr
cd tee4ehr/
bash supervised.sh
conda activate cheonwoolee-tee4ehr
cd tee4ehr/
bash supervised.sh
cd
l
df -h
rm -r Idea_develop_SSLRec
rm -r Idea_develop_SSLRec_Pytorch/
ls /data/seminars
df -h
watch -n -1 nvidia-smi
nvidia-smi
df -h
cd samsung-project/
python
l
sudo adduser larry
cat /etc/group
sudo usermod -aG users,docker,datagroup larry
cat /etc/group
groups larry
groups cheonwoolee
groups doyunchoi
nvidia-smi
docker ps -a
df -h
top
clear
python
df -h
docker df
docker system df
cd ../..
ls
cd data
ls
cd
df -h
cat /etc/groups
cat /etc/group
cd samsung-project/
python preprocess.py 
python dataloader.py 
conda create samsung
conda create --name samsung
conda activate samsung
conda install torch
python dataloader.py
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python dataloader.py
conda install pandas
python dataloader.py
python preprocess.py
python dataloader.py
python main.py
conda deactivate
conda env list
conda remove samsung
conda env list
conda remove -name samsung --all
conda create --name samsung python=3.8
conda activate samsung
python main.py
pip install pandas
python main.py
pip install matplotlib
python main.py
pip install torch
python main.py
df -h
nvidia-smi
df -h
nvidia-smi
docker images
docker ps -a
cd tmp
cd tee4ehr/
conda activate cheonwoolee-tee4ehr
python tmp.py
cd ../..
ls
cd ..
ls
cd
cd samsung-project/
git init
git remote -v
git remote add origin https://github.com/2000nu/samsung-project.git
git branch -M main
git push -u origin main
git remote -v
git commit -m "first commit"
git add .
git push -u origin master
git init
git add .
git commit -m "Initial commit"
git remote -v
git push -u origin master
git push -u origin main
cd samsung-project/
git remote -v origin
git remote -v 
git init
git remote origin https://github.com/2000nu/samsung-project.git
git remote set-url origin https://github.com/2000nu/samsung-project.git
git remote -v
git add .
git remote add origin https://github.com/2000nu/samsung-project.git
git remote -v
git add .
git commit -m "Initial commit"
git push -u origin main
git push -u origin master
git commit --amend
git reset HEAD dataset/test.pkl
git reset HEAD dataset/train.pkl
git reset HEAD dataset/test.pkl
git reset HEAD dataset/data.pkl
git push -u origin master
git rm --cached dataset/data.pkl
git rm --cached dataset/train.pkl
git rm --cached dataset/validation.pkl
git rm --cached dataset/test.pkl
git commit -m "Initial commit without large files"
git push origin main
git push -u origin master
git rm --cached dataset/data.pkl
git rm --cached dataset/train.pkl
git rm --cached dataset/validation.pkl
git rm --cached dataset/test.pkl
git commit --amend -C HEAD
echo "dataset/data.pkl" >> .gitignore
echo "dataset/train.pkl" >> .gitignore
echo "dataset/validation.pkl" >> .gitignore
echo "dataset/test.pkl" >> .gitignore
git push -u origin master --force
git commit --amend -C HEAD
git push -u origin master --force
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch dataset/data.pkl dataset/train.pkl dataset/validation.pkl dataset/test.pkl' --prune-empty --tag-name-filter cat -- --all
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
python tmp.py
python preprocess.py
python tmp.py
git push origin master
git checkout -b data-preprocessing
git push origin data-preprocessing 
python
du -h
du -sh
cd samsung-project/
du -h
cd
cd tee4ehr/
du -h
cd
cd Time2Vec-PyTorch/
du -h
cd
du -h
cd .conda
du -h
cd
df -h
python
cd samsung-project/
python preprocess
python preprocess.py
python dataloader.py
conda activate samsung
python dataloader.py
python models.py
df -h
nvidia-smi
python dataloader.py
python tmp.py
zip -r samsung-project.zip .
scp cheonwoolee@como1.kaist.ac.kr:/home/cheonwoolee/samsung-project.zip ~/Downloads
scp cheonwoolee@como1.kaist.ac.kr:/home/cheonwoolee/samsung-project/samsung-project.zip ~/Downloads
cd ../..
l
cd ..
l
find ~ -name "samsung-project.zip"
ls
cd
cd samsung-project/
python tmp.py
df -h
python tmp.py
python preprocess.py
python dataloader.py
python preprocess.py 
git add .
git commit -m "modification of data pre-processing"
git push origin master
git remote set-url origin https://github.com/2000nu/project.git
git remote -v
git push origin master
df -h
cd /data/samsung-project
ls
cd /data/samsung-project/
ls
l
ls -l
datashader 
cd data-20240715/
ls -l
mv /home/cheonwoolee/samsung-project/dataset .
sudo chmod -R u+w .
ls -l
mv /home/cheonwoolee/samsung-project/dataset .
cat /etc/group
ls -l
sudo chmod -R 775 .
ls -l
mv /home/cheonwoolee/samsung-project/dataset .
ls -l
cp /home/cheonwoolee/samsung-project/dataset .
du -h
cd ..
l
cd data-20240715/
sudo chmod 664 .
ls -l
ls
sudo chmod 775 .
ls
ls -l
chmod u-x .
sudo chmod u-x .
ls -l
sudo chmod g-x .
ls -l
sudo chmod -R u-x g-x .
sudo chmod -R g-x .
ls -l
sudo chmod -R o-x .
ls -l
sudo chmod -R o+x .
ls -l
chmod o-x fdc_result.csv 
sudo chmod o-x fdc_result.csv 
sudo chmod o-x oes_result.csv 
ls -l
cd ..
cp /home/cheonwoolee/samsung-project/dataset .
cp -r /home/cheonwoolee/samsung-project/dataset .
ls
ls -l
cd data
cd dataset
l
cd ..
mv ./dataset ./preprocess
ls

l
conda activate samsung
python preprocess.
python preprocess.py
cd
cd samsung-project/
python preprocess.py
du -h
python dataloader.py
python models.py
python dataloader.py
python preprocess.py
python dataloader.py
python models.py
cp -f dataset/data.pkl /home/data/samsung-project
ls /home/data
ls /home/dataset

cp -f dataset/data.pkl /data/samsung-project
python models.py
git remote -v
git status
git add .
git commit -m "9/6 data preprocessing"
git remote -v
git branch -a
git push origin data-preprocessing 
nvidia-smi
df -h
mv -r dataset /data/samsung-project/data-20240715/
mv dataset /data/samsung-project/data-20240715/
nvidia-smi
du -sh
df -h
docker system df
git clone https://github.com/YuliaRubanova/latent_ode.git
df -h
du -h
du -sh
cd .conda
du -sh
cd ..
conda env list
conda activate cheonwoolee-tee4ehr
nvidia-smi
python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
cd latent_ode/
python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
conda deactivate
conda env list
conda create --name tee4ehr clone cheonwoolee-tee4ehr
conda remove --name cheonwoolee-tee4ehr --all
conda env list
conda remove --name time2vec --all
conda env list
conda create --name latent_ode
conda remove --name latent_ode
conda remove --name latent_ode --all
conda create --name latent_ode python=3.9
conda acitvate latent_ode
conda activate latent_ode
conda install matplotlib numpy pandas scikit-learn pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
pip install torchdiffeq
python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
pip install umap-learn
python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
python3 run_models.py --niters 100 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
python3 run_models.py --niters 100 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 --viz
conda activate samsung
python tmp.py
cd /data/samsung-project/
ls
ls -l
cd preprocess/
ls
ls -l
cd ..
rm data.pkl
ls
cd data-20240715/
ls -l
cd
df -h
docker images
docker ps -a
git remote -v
git pull origin data-preprocess
git pull origin data-preprocessing
conda activate samsung
python preprocess.py
pip install -r requirements.txt
python preprocess.py
ls /data/samsung-project/
ls /data/samsung-project/preprocess/
python preprocess.py
ls /data/samsung-project/
ls /data/samsung-project/preprocess/
du -h
clear
python preprocess.py
conda activate samsung
python tmp.py
git remote -v
git add .
git commit -m "aggregate each sensor values as its mean"
git push origin data-preprocessing
git add .gitignore
git commit -m "without preprocess file"
git push origin data-preprocessing
git rm -r --cached preprocess
git rm -r --cached tmp.py
git commit -m "Remove preprocess folder from repository"
git push origin data-preprocessing
df -h
docker images
docker ps -a
df -h
docker images df
docker system df
nvidia-smi
docker system df
df -h
docker images 
docker ps -a
docker system df
df -
df -h
docker system f
docker system df
cd /mnt/ssd1
l
ls /
ls
df --help
df -h
docker system --help
docker builder --help
docker builder prune
docker system df
df -h
cd
df -h
ls /
ls -l /
cd /mnt/ssd1
ls -l
cd ..
ls -l
ls -ld /mnt/ssd1
ls -ld /home
ls -ld /home/
ls -l /home/
nvidia-smi
df -h
nvidia-smi
df -h
git add .
git commit -m "adding more models"
git branch
git push origin data-preprocessing 
git clone --granch data-preprocessing
git clone --branch data-preprocessing git@github.com:2000nu/project.git
git clone -b data-preprocessing https://github.com/2000nu/project.git
conda env list
conda activate samsung
python preprocess.py
python models.py
python preprocess.py
python models.py
python main.py
git remote -v
git add .
git commit -m "modify the data to work with scikit-learn framework"
git push origin data-preprocessing
df -h
python main.py
pip install requirements.txt
pip install -r requirements.txt
python main.py
pip install -r requirements.txt
python main.py
git pull origin data-preprocessing
git add .
git commit -m "solving catboostregressor problem"
git push origin data-preprocessing 
git reset --mixed HEAD~1
git push --help
git pull origin data-preprocessing 
git checkout -b cheonwoolee
git add .
git commit -m "solve XGBoostRegeressor problem"
git remote -v
git push origin cheonwoolee 
conda activate samsung
python main.py
python preprocess
python preprocess.py
python main.py
python preprocess.py
python main.py
git fetch --help
git fetch 
git fetch origin
git fetch origin/data-preprocessing
git pull origin data-preprocessing 
git checkout main
git remote -v
git checkout master
git checkout data-preprocessing 
ls /data/samsung-project/
ls /data/samsung-project/preprocess
ls /data/samsung-project/preprocessing
cd /data/samsung-project/
ls
ls -l
rm -r preprocess
ls -l
conda activate samsung
python main.py --file catboost --config catboost
python main.py --help
python main.py --log linear2 --config linear
ls /data/samsung-project/
python main.py --log linear2 --config linear
python main.py --log extratree --config extratree
conda activate samsung
python main.py --log xgboost --config xgboost
conda activate samsung
python main.py
python main.py --config linear
python main.py --config lightgbm --name lightgbm
python main.py --config lightgbm --log lightgbm
conda activate samsung
top
clear
df
df -h
python main.py --log catboost2 --config catboost
python main.py --config naivebayes --log naivebayes
python tmp.py
python main.py --config naivebayes --log naivebayes
python tmp.py
python main.py --config naivebayes --log naivebayes
python main.py --config randomforest --log randomforest
python main.py --config gradientboosting --log gradientboosting
python main.py --config kneighbors --log kneighbors
python main.py --log decisiontree --config decisiontree
git add .
git commit -m "set config, modify gitignore, code refactorization"
git checkout master
git checkout --help
git branch
git pull origin master
git remote -v
git checkout cheonwoolee
git push origin master
git checkout master
git checkout data-preprocessing 
git branch
git push origin master
git remote set-url origin https://github.com/2000nu/project.git
git remote -v
git push origin master
git remote remove origin
git remote add origin https://dlcjsdn07:SHA256:/ufdPALLwNUWPY4fOFMjM/z0D7c5fTx4ke0joux3AhU@github.com/2000nu/project
git remote -v
git push origin master
git remote remove origin
git remote set-url origin https://github.com/2000nu/project.git
git remote add origin https://github.com/2000nu/project.git
git push origin master
git remote -v
git remote set-url origin https://2000nu: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@github.com/2000nu/project.git
git remote -v
git push origin master
git push origin cheonwoolee
git branch
git checkout master
git checkout data-preprocessing 
git pull origin master
git merge data-preprocessing
git branch
git checkout master
git checkout data-preprocessing 
git status
git checkout master
git pull origin master
git merge data-preprocessing
git add .
git commit -m "solve conflicts"
git push origin master
conda activate samsung
python preprocess.py
python main.py
pip install -r requirements.txt 
python main.py
ls /data/samsung-project/
python main.py
python main.py --help
python main.py --file linear --config linear
python main.py --file mlp --config mlp
python main.py --log mlp2 --config mlp
conda activate samsung
python main.py --config mlp --log mlp3
python main.py --config mlp --log mlp_tmp
python main.py --config mlp --log mlp2
python main.py --config lightgbm
nvidia-smi
CUDA_VISIBLE_DEVICES=3 python main.py --config xgboost --log xgboost_gpu
conda activate samsung
python main.py --config lightgbm --log lightgbm_tmp
python main.py --config xgboost --log xgboost2
python preprocess.py
nvidia-smi
tasklist
nvidia-smi
ps aux | grep 1362402
kill -9 1362402
nvidia-smi
ps aux | grep 1377438
ps aux | grep 1370753
nvidia-smi
conda activate samsung
python tmp.py
cd /data/seminars/m
cd /data/seminars/
ls
conda activate samsung
python tmp.py
conda activate samsung
python tmp.py
python tmp2.py
df -h
nvidia-smi
ps -aux | grep 409223
ps -aux | grep 522252
ps -aux | grep 3785515
ps -aux | grep 563987
ps -aux | grep 580701
docker df -h
docker system df
nvidia-smi
watch -n 1 nvidia-smi
rocminfo --help
cat /opt
cd /opt
ls
cd
dpkg --help
dpkg -l
dpkg -l | grep rocm
df -h
ls /data/cheonwoolee/coformer/P12
ls /data/cheonwoolee/coformer/P12/data
cd preprocess/
python P12.py
cat /data/cheonwoolee/coformer/
cat /data/cheonwoolee/coformer
ls /data/cheonwoolee/coformer
ls /data/cheonwoolee/coformer/P12/
ls /data/cheonwoolee/coformer/P12/data
conda create --name coformer
conda create --name coformer python=3.9
conda env list
conda activate coformer
pip install -r requirements.txt
pip install scikit-learn
pip install -r requirements.txt
python tmp.py
ls
python tmp.py
cd preprocess/
python tmp.py
python P12.py
cd ..
python tmp.py
ls /data
ls /data/cheonwoolee
ls /data/cheonwoolee/tee4ehr
cd /data/cheonwoolee/tee4ehr/
df 0h
df -h
du -h
df -h
cd ..
mkdir coformer
ls
cd coformer
ls
git clone https://github.com/MediaBrain-SJTU/CoFormer.git
conda env list
conda activate coformer
cd rawdata/
python parsedata.py 
pip install pandas
python parsedata.py 
nvidia-smi
cd /data
ls -l
rm -r douban
ls
cd test
ls
cd cgi
cd CGI-main/
ls
conda env list
df -h
nvidia-smi
conda env create -f environment.yml
conda activate CGI
python test.py
pip install torch
python test.py
CUDA_VISIBLE_DEVICES=7 python test.py
python tmp.py
CUDA_VISIBLE_DEVICES=7 python test.py
conda activate CGI
python test.py
cd CGI-main/
python test.py
cd fig
cd data/douban/fig/
cd ..
cd ../..
l
conda deactivate
conda remove --name CGI --all
conda env create -f environment.yml
conda env list
conda remove --name CGI --all
conda env create -n CGI python=3.9
conda env create -n CGI python==3.9
conda env create --name CGI python=3.8
conda create --name CGI python=3.9
conda activate CGI
pip install -r requirements.txt
python test.py
CUDA_VISIBLE_DEVICES=5 python test.py
pip install tqdm
CUDA_VISIBLE_DEVICES=5 python test.py
nvidia-smi
conda env list
conda activate CGI
CUDA_VISIBLE_DEVICES=6 python test.py
pip install seaborn
CUDA_VISIBLE_DEVICES=6 python test.py
df -h
docker images
conda env list
python sensor.py -s all
cd data
python sensor.py -s all
conda activate coformer
cd data
cd preprocess/
python p12.py
cd ..
cd data/
python p12.py
df -h
python p12.py
git remote -v
python p12.py
python tmp.py
python plot.py
python measurement.py
python sensor.py -s 6,31
python sensor.py -s all
python measurement.py -s 1,2,5
python measurement.py -s 6,31
python measurement.py -s all
python plot.py
python print_causlity.py

python print_causlity.py
python print_causality.py
python plot_measurement.py --ehlp
python plot_measurement.py --d 0
python print_causality.py --help
python print_causality.py 
python plot_measurement.py --d 0
python print_causality.py 
python print_causality.py --help
python print_causality.py -w 60
python print_causality.py -w 120
python print_causality.py -w 60
python print_causality.py -w 180
python print_causality.py -w 240
python plot_measurement.py --help
python plot_measurement.py -s false
python plot_measurement.py -s true
python plot_value.py --help
python plot_value.py -s -d 0
python plot_value.py -d 0 -s 33,4,7,11
python plot_measurement.py --help
python plot_measurement.py -s true -t 5
python plot_measurement.py -s false -t 5
python plot_value.py -d 0 -s 33,4,7,11
python plot_value.py -d 0 -s 33,20,4,7,11
python print_causality.py -d 1
python print_causality.py -d 0
python print_causality.py -d 1
python plot_value.py -d 1 -s 4,7,9,25
python print_causality.py -d 1
python plot_value.py -d 1 -s 9,13,19,33
python print_causality.py -d 1
python plot_measurement.py --help
python plot_measurement.py -t 5 -s true
python plot_value.py -d 1 -s 9,18,19,33
python print_causlity.py --help
python print_causality.py --help
python print_causality.py -d 7 -t 15
python plot_value.py -d 7 -s 9,10,18,25
python plot_value.py -d 7 -s 9,10,18,25 -t 15
python print_similarity.py 
python print_similarity.py --help
python print_similarity.py 
python print_causality.py -d 20 -t 10
python plot_value.py -d 20 -t 10 -s 10,20,27,30
python print_causality.py -d 20 -t 10
python print_causality.py -d 0 -t 10
python plot_value2.py --help
python plot_value2.py -d 7 -t 10
python plot_value2.py -d 20 -t 10
conda activate coforemr
conda activate coformer
python p12.py
cd data
python p12.py
cd CoFormer
conda activate coformer
python plot_measurement.py --help
cd data
python plot_measurement.py --help
python plot_measurement.py -t 5 -s true
python plot_measurement.py -t 5 -s false
python print_regularity.py
python plot_measurement.py -t 5 -s false
python print_regularity.py
python print_outlier.py
python plot_measurement.py -d 428
python plot_value.py -d 428
python plot_value.py -d 428 -s 5
python plot_measurement.py -d 1373
python plot_measurement.py --help
python plot_measurement.py -t 5 -s true
python plot_measurement.py -t 5 -s false
python plot_measurement.py -d 428
python plot_value.py -d 428
python plot_measurement.py -d 1373
python plot_value.py -d 1373
python plot_value.py -d 2350
python plot_measurement.py -d 2350
python print_data.py -d 428
python print_data.py -d 428, 1373, 2350, 2571, 4761, 8372, 9586, 10147
python print_data.py -d 428,1373,2350,2571,4761,8372,9586,10147
python plot_value.py -d 2571
python plot_value.py -d 4761
python plot_value.py -d 10147
python print_similarity.py
python print_correlation.py 
cd CoFormer/
cd data
conda activate coforemr
conda activate coformer
python plot_value2.py --help
python plot_value2.py -d 1 -t 10
python plot_value2.py -d 0 -t 10
cd CoFormer
conda activate coformer
cd data
python print_correlation.py 
python tmp.py
python print_similarity.py 
python plot_value2 --help
python plot_value2.py --help
python plot_value2.py -d 30 -t 10
python print_correlation.py
python tmp.py
python print_correlation.py
python tmp.py
python print_correlation.py
python tmp.py
ls -l /data/seminars
cd /data/seminars
mv 241018_bini.pptx 241008_bini.pptx
ls -l
nvidia-smi
conda activate CGI
CUDA_VISIBLE_DEVICES=7 python case_study.py
python case_study.py
nvidia-sm
nvidia-smi
conda env list
conda activate CGI
CUDA_VISIBLE_DEVICES=7 python case_study.py
nvidia-smi
CUDA_VISIBLE_DEVICES=7 python case_study.py
CUDA_VISIBLE_DEVICES=6,7 python case_study.py
conda activate CGI
python case_study.py 
nvidia-smi
CUDA_VISIBLE_DEVICES=7 python case_study.py 
CUDA_VISIBLE_DEVICES=6,7 python case_study.py 
conda activate CGI
CUDA_VISIBLE_DEVICES=6,7 python case_study.py 
conda activate CGI
CUDA_VISIBLE_DEVICES=6,7 python case_study.py 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6,7 python case_study.py 
nvidia-smi
python case_study.py -u 1623
CUDA_VISIBLE_DEVICES=6,7 python case_study.py 
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -u 1 -k 10
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -u 2,3,4,5 -k 10
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -u 1 -k 10
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -u 1623,5076,5489 -k 10
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -u 1,2,3 -k 10
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -u 1,2,3 -k 5
CUDA_VISIBLE_DEVICES=6,7 python case_study.py
CUDA_VISIBLE_DEVICES=6,7 python case_study.py -k 5
conda activate CGI
python find_nonoverlap_users.py 
conda activate CGI
python case_study.py 
conda activate CGI
python case_study.py 
cd /data/seminars
ls -l
df -h
nvidia-smi
cd SocialLGN-master/
df -h
conda env list
conda remove --name CGI
conda remove CGI
conda env list
conda env remove --name CGI
conda create --name sociallgn python=3.8
conda activate sociallgn
conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 python main.py --model=SocialLGN --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048
pip install scikit-learn
CUDA_VISIBLE_DEVICES=0,1 python main.py --model=SocialLGN --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048
pip install pandas
CUDA_VISIBLE_DEVICES=0,1 python main.py --model=SocialLGN --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048
nvidia-smi
kill -9 3900936
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 python main.py --model=SocialLGN --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048
