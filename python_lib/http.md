# HTTP 网络相关代码

## 常用模块

- 获取代理IP

    ``` python
    import requests
    PROXY_POOL_URL = "http://192.168.135.15:5555/random"

    def get_proxy():
        try:
            response = requests.get(PROXY_POOL_URL)
            if response.status_code == 200:
                return response.text
        except ConnectionError:
            return None
    ```

- [超时重连](https://www.cnblogs.com/gl1573/p/10129382.html)

    ``` python
    import time
    import requests
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=3))
    s.mount('https://', HTTPAdapter(max_retries=3))

    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    try:
        r = s.get('http://www.google.com.hk', timeout=5)
        print(r.text)
    except requests.exceptions.RequestException as e:
        print(e)

    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    ```
