# [Celery](https://www.celerycn.io/)

- Tutorial

## 基本操作

```bash
# enter redis
redis-cli

# 获取密码
config get requirepass

# 设置密码
config set requirepass 123456

# 当有密码的时候登录时需要密码登录
auth 密码

# 取消密码
config set requirepass ''

# 切换数据库
select n

# celery全杀(服务器所有的celery)
ps auxww | grep 'celery' | awk '{print $2}' | xargs kill -9

```

## 教程

- [任务队列神器：Celery 入门到进阶指南](https://blog.csdn.net/chinesehuazhou2/article/details/115153198)
- [使用celery构建分布式爬虫抓取空气质量指数](https://www.jianshu.com/p/f225fcc3c97d)
- [Celery手动配置路由](https://www.jianshu.com/p/11b420aea529)
- [Celery多队列配置][https://blog.csdn.net/sinat_38682860/article/details/104030062]
- [使用celery构建分布式爬虫抓取空气质量指数](https://www.jianshu.com/p/f225fcc3c97d)
- [启动关闭 命令](https://blog.csdn.net/qq_42327755/article/details/100670153)

### aqicn.py

``` python
from celery import Celery
from bs4 import BeautifulSoup
import re
import time

import ohRequests as requests

# 这里定义了broker和backend, 注意IP和后面的数字都是可以调整的
app = Celery('aqicn', broker='redis://:''@34.229.250.31/2', backend='redis://:''@34.229.250.31:6379/3')

@app.task
def crawl(location, url):
    req = requests.ohRequests()
    content = req.get(url)
    
    if not content:
        return None
        
    pattern = re.compile('<table class=\'api\'(.*?)</table>', re.S)
    data = pattern.findall(content)

    if data:
        data = "<table class='api' {} </table>".format(data[0])
    soup = BeautifulSoup(data, 'lxml')

    aqi = soup.find(id='aqiwgtvalue').text

    if aqi == '-':
        return None

    t = soup.find(id='aqiwgtutime').get('val')
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(t)))

    return [location, aqi, t]

```

### execute_tasks.py

``` python
from aqicn import crawl
#from celery import app

URLS = []

def readurls():
    """将URLs从文件里读取出来"""
    with open("urls.txt", 'r') as file:
        while True:
            item = file.readline()
        
            if not item:
                break
            
            data = item.split(',')
            location, url = data[0], data[1]
            URLS.append((location, url.replace('\n','')))
            
def task_manager():
    for url in URLS:
        crawl.delay(url[0],url[1])
        #app.send_task('aqicn.crawl', args=(url[0],url[1],))
    
if __name__ == '__main__':
    readurls()
    #print (len(URLS))
    task_manager()
```

### 获取数据结果

``` python
import redis
import json

r = redis.Redis(host='34.229.250.31',port=6379,db=3)

keys = r.keys()

for key in keys:
    res = r.get(key)
    res = json.loads(res.decode('utf-8'))
    print(res.get('result'))
```
