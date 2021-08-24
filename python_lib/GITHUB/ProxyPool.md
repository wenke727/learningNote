
# [ProxyPool IP代理池](https://github.com/Python3WebSpider/ProxyPool)

参考教程：
<https://zhuanlan.zhihu.com/p/59951949?utm_source=wechat_session>；
<https://cuiqingcai.com/7048.html>

``` bash
screen -r proxypool
cd /home/pcl/traffic/ProxyPool
docker-compose up
http://localhost:5555/random
```

调用资源池参考代码

``` python
import json
import requests
PROXY_POOL_URL = 'http://192.168.135.34:5555/random'

def get_proxy():
    try:
        response = requests.get(PROXY_POOL_URL)
        if response.status_code == 200:
            return response.text
    except ConnectionError:
        return None

proxies = get_proxy()
url = 'http://restapi.amap.com/v3/direction/driving?key=465d1fe32dead60bf74e470741563a99&origin=113.941655,22.572978&destination=113.902734,22.884893&strategy=10'
response = requests.get(url, proxies={'http': proxies}  )

print(proxies)
json.loads(response.content)
```
