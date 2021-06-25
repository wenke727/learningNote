# Python常用库指南

## XML解析库

****

# [multiprocessing](https://docs.python.org/zh-cn/3.8/library/multiprocessing.html)

- if __name__ == '__main__' 部分是必需的解释，

# Celery 分布式爬虫

[使用celery构建分布式爬虫抓取空气质量指数](https://www.jianshu.com/p/f225fcc3c97d)

[启动关闭 命令](https://blog.csdn.net/qq_42327755/article/details/100670153)

```
ps auxww | grep 'celery' | awk '{print $2}' | xargs kill -9
ps auxww | grep 'vscode' | awk '{print $2}' | xargs kill -9
```

- Tutorial
[任务队列神器：Celery 入门到进阶指南](https://blog.csdn.net/chinesehuazhou2/article/details/115153198)

****

## GITHUB

- [ProxyPool IP代理池](https://github.com/Python3WebSpider/ProxyPool)
    参考教程：
    <https://zhuanlan.zhihu.com/p/59951949?utm_source=wechat_session>；
    <https://cuiqingcai.com/7048.html>

    ```
    screen -r proxypool
    cd /home/pcl/traffic/ProxyPool
    docker-compose up
    http://localhost:5555/random
    ```

    调用资源池参考代码

    ```
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

****

## [shapely](https://shapely.readthedocs.io/en/stable/manual.html)

- `parallel_offset`
Returns a LineString or MultiLineString geometry at a distance from the object on its right or its left side.

# Pandas

## 格式设置

在使用dataframe时遇到datafram在列太多的情况下总是自动换行显示的情况，导致数据阅读困难
在代码中设置显示的长宽等, [REF](https://blog.csdn.net/lihuarongaini/article/details/101298171)

```
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```

## numpy

- 求差集
  `np.setdiff1d()`
