# Amax server

- [【保姆级教程】个人深度学习工作站配置指南](https://zhuanlan.zhihu.com/p/336429888)

## 机器重启服务

- 开启docker

  - sudo docker exec -it eta bash

- connect to the Internet
  Ref:
  - [Ubuntu 18.04网络不通，netplan命令不存在](https://www.cnblogs.com/zh-dream/p/13405799.html)
  - [Ubuntu配置和修改IP地址](https://www.cnblogs.com/linjiqin/p/3148346.html)

  Command:  

  ``` bash
  # 开启网口
  ifconfig  enx000ec6ccad1a up
  # 动态主机配置协议动态的配置网络接口的网络参数
  sudo dhclient enx000ec6ccad1a
  # 发现已经有IP地址，局域网和外网都能ping通。但是，IP地址并不是之前配的静态IP地址
  sudo ifconfig enx000ec6ccad1a

  # update IP config in the server
  sudo nano /etc/network/interfaces 
  > auto enx000ec6ccad1a
  > iface enx000ec6ccad1a inet static
  > address 192.168.135.15
  # 使用netplan更改IP地址
  sudo apt-get install netplan.io
  # 使静态ip的配置文件生效
  sudo netplan apply
  ```

## 软件安装

- [GDAL](https://www.cnblogs.com/Assist/p/14034447.html)

  ``` bash
  sudo add-apt-repository ppa:ubuntugis/ppa 
  sudo apt-get update
  sudo apt-get install gdal-bin
  sudo apt-get install libgdal-dev
  export CPLUS_INCLUDE_PATH=/usr/include/gdal
  export C_INCLUDE_PATH=/usr/include/**gdal**
  gdal-config --version  #(get version)
  pip install GDAL==version
  ```

- Redis
  
  命令

  ``` bash
  # Install
  sudo apt-get install redis-server
  # Check the status
  ps -aux|grep redis
  netstat -nlt|grep 6379
  ```

- Samba
  [samba使用指定端口windows访问linux](https://www.huaweicloud.com/articles/13710590.html)

  ```bash
  netsh interface portproxy add v4tov4 listenport=445 listenaddress=127.0.0.1 connectport=139 connectaddress=192.168.135.15
  ```

  [Windows系统开启telnet命令](https://help.aliyun.com/document_detail/40796.html)

## 服务

### ssh自启动

  ``` bash
  vi /etc/rc.local
  # add
  /etc/init.d/ssh start
  ```

- [修改linux远程端口22连不上,Linux系统修改远程连接22端口](https://blog.csdn.net/weixin_42322512/article/details/116633836?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)
- [服务器22端口连接超时 ssh: connect to host *** port 22: Operation timed out](https://www.cnblogs.com/oldboyooxx/p/10387150.html)

### Postgresql

- 启动服务
  
  ``` bash
  /etc/init.d/postgresql start
  or sudo systemctl start postgresql
  # passwd: the amax passwd
  ```

- enter the postgresql

  ``` bash
  sudo su postgres
  psql
  ```

- Error:
  > could not access file "$libdir/postgis-2.4": No such file or directory

  Solution:
  这个问题是由于postgresql和postgis版本不同出现的问题
  Ref: [ERROR: could not access file "$libdir/postgis-2.3": No such file or director解决方法](https://blog.csdn.net/weixin_43966390/article/details/102955201)

  - To find out what $libdir is referring to, run the following command:

    ``` bash
    pg_config --pkglibdir # for Amax: /usr/lib/postgresql/10/lib
    ```

  - 到这个目录查看有没有相关的postgis的文件，如果postgis文件版本名为2.4就是对的，如果是其他版本名，只需要把文件名改为自己需要的文件名

## Jupyter

- AI learing: <http://192.168.135.34:8888/tree>?

  folder: `/home/pcl/Data/0_Learning/jupyer_folder`

- Pytorch: <http://192.168.135.34:8888/tree>?

## 操作命令

- tar

  ``` bash
  # zip
  tar cvf panos_test_dataset.tar  ./dataset
  # unzip
  tar -xvf ****.tar.gz -C ./
  ```

- zip & unzip

  ``` bash
  # 把/home目录下面的mydata目录压缩为mydata.zip
  zip -r mydata.zip mydata #压缩mydata目录
  
  # 把/home目录下面的mydata.zip解压到mydatabak目录里面
  unzip mydata.zip -d mydatabak
  
  # 把/home目录下面的abc文件夹和123.txt压缩成为abc123.zip
  zip -r abc123.zip abc 123.txt
  
  # 把/home目录下面的wwwroot.zip直接解压到/home目录里面
  unzip wwwroot.zip
  
  # 把/home目录下面的abc12.zip、abc23.zip、abc34.zip同时解压到/home目录里面
  unzip abc*.zip
  
  # 查看把/home目录下面的wwwroot.zip里面的内容
  unzip -v wwwroot.zip
  
  # 验证/home目录下面的wwwroot.zip是否完整
  unzip -t wwwroot.zip
  
  # 把/home目录下面wwwroot.zip里面的所有文件解压到第一级目录
  unzip -j wwwroot.zip

  
  主要参数
  -c：将解压缩的结果
  -l：显示压缩文件内所包含的文件
  -p：与-c参数类似，会将解压缩的结果显示到屏幕上，但不会执行任何的转换
  -t：检查压缩文件是否正确
  -u：与-f参数类似，但是除了更新现有的文件外，也会将压缩文件中的其它文件解压缩到目录中
  -v：执行是时显示详细的信息
  -z：仅显示压缩文件的备注文字
  -a：对文本文件进行必要的字符转换
  -b：不要对文本文件进行字符转换
  -C：压缩文件中的文件名称区分大小写
  -j：不处理压缩文件中原有的目录路径
  -L：将压缩文件中的全部文件名改为小写
  -M：将输出结果送到more程序处理
  -n：解压缩时不要覆盖原有的文件
  -o：不必先询问用户，unzip执行后覆盖原有文件
  -P：使用zip的密码选项
  -q：执行时不显示任何信息
  -s：将文件名中的空白字符转换为底线字符
  -V：保留VMS的文件版本信息
  -X：解压缩时同时回存文件原来的UID/GID
  ```

- 查看磁盘

  ``` bash
  df -hl # 查看磁盘剩余空间
  df -h # 查看每个根路径的分区大小
  du -sh [目录名] # 返回该目录的大小
  du -sm [文件夹] # 返回该文件夹总M数
  du -h [目录名] # 查看指定文件夹下的所有文件大小（包含子文件夹）
  ```

- 查找代码的函数

  ``` bash
  find RoadNetworkCreator_by_View/ -name "*.py"|xargs wc -l
  ```

- scp

  ``` bash
  scp local_file remote_username@remote_ip:remote_file 
  -P : port：注意是大写的P, port是指定数据传输用到的端口号
  -r: 递归复制整个目录
  -C 允许压缩
  ```

## Github

- check the lines

  ``` bash
  git log --format='%aN' | sort -u | while read name; do echo -en "$name\t"; git log --author="$name" --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\n", add, subs, loc }' -; done
  ```

- 切换分支

  ``` bash
  # 列出所有的分支
  git branch -a
  # 切换回dev分支，并开始开发
  git checkout dev
  ```

- .gitignore
    The GitHub’s collection of .gitignore file templates: <https://github.com/github/gitignore>
- ...

## Docker

  学习资源：
  <https://zhuanlan.zhihu.com/p/78295209>
  常用命令：
  <https://www.hangge.com/blog/cache/detail_2402.html>

- 安装并启动步骤

  ``` bash
  安装docker: 
  https://docs.docker.com/engine/install/ubuntu/
 
  安装nvidia docker: 
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker 
  https://github.com/NVIDIA/nvidia-docker
 
  安装deepo: 
  https://github.com/ufoym/deepo 
 
  创建和进入docker
  sudo NV_GPUV_GPU='1' nvidia-docker run -ti -p 7022:22 -p 5000:5000 -p 6000:6000 --name eta -v /home/pcl/Data:/Data  -d --ipc=host ufoym/deepo:tensorflow-py27 bash
  docker exec -it [name/id] /bin/bash
 
  启动已经创建的docker
  sudo docker start eta
  sudo docker exec -it eta /bin/bash
  service ssh start
 
  Jupyter设置 
  pip install jupyter
  jupyter notebook --generate-config
  jupyter notebook password
  sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g"  /root/.jupyter/jupyter_notebook_config.py
  sed -i "s/#c.NotebookApp.port = 8888/c.NotebookApp.port = 7000/g"  /root/.jupyter/jupyter_notebook_config.py
  jupyter notebook --allow-root
 
  设置新的账户
  sudo passwd root
  adduser pcl
  来自 <https://blog.csdn.net/geol200709/article/details/82116267> 
  
  ssh设置
  apt-get install openssh-server
  /etc/init.d/ssh start 
  ```

- 进入容器
  
  ``` bash
  docker exec -it XXX bash
  exit
  ```

- 重启容器

  ``` bash
  sudo docker ps -a
  sudo docker start XXX
  sudo docker exec -it XXX bash
  exit
  ```

- save images

    ``` bash
    sudo docker save imgs_name -o prp.tar
    ```

- load images

    ``` bash
    sudo docker load --input imgs_name
    ```

- 服务自启动

    ``` bash
    systemctl enable docker.service
    # 2.1在启动容器时，添加--restart=always参数，如
    docker run --restart=always
    # 2.2如果容器已经启动，可以使用命令更新参数
    docker update --restart=always 容器id    
    systemctl restart docker
    ```

- [端口映射](https://www.jb51.net/article/142462.htm)

    ``` bash
    iptables -t nat --list-rules PREROUTING
    
    iptables -t nat --list-rules DOCKER
    iptables -t nat -A DOCKER ! -i docker0 -p tcp -m tcp --dport 5000 -j DNAT --to-destination 172.17.0.3:5000

    iptables -t nat --list-rules POSTROUTING
    iptables -t nat -A POSTROUTING -s 172.17.0.3/32 -d 172.17.0.3/32 -p tcp -m tcp --dport 5000 -j MASQUERADE


    iptables --list-rules DOCKER
    iptables -t filter -A DOCKER -d 172.17.0.3/32 ! -i docker0 -o docker0 -p tcp -m tcp --dport 5000 -j ACCEPT
    ```
