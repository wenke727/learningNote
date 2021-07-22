# [Logging](https://pypi.org/project/logging/)

Ref:

- [python日志：logging模块使用](https://zhuanlan.zhihu.com/p/360306588)
- [python 日志 logging模块(详细解析)](https://blog.csdn.net/pansaky/article/details/90710751)

``` python
# -*- encoding:utf-8 -*-
import logging

# create logger
logger_name = "example"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# create file handler
log_path = "./log.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.WARN)

# create formatter
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

# add handler and formatter to logger
fh.setFormatter(formatter)
logger.addHandler(fh)

# print log info
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')```

```

- 文件配置
配置文件logging.conf如下:

```[loggers]
keys=root,example01

[logger_root]
level=DEBUG
handlers=hand01,hand02

[logger_example01]
handlers=hand01,hand02
qualname=example01
propagate=0

[handlers]
keys=hand01,hand02

[handler_hand01]
class=StreamHandler
level=INFO
formatter=form02
args=(sys.stderr,)

[handler_hand02]
class=FileHandler
level=DEBUG
formatter=form01
args=('log.log', 'a')

[formatters]
keys=form01,form02

[formatter_form01]
format=%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s
```

使用程序logger.py如下:

``` python
#!/usr/bin/python
# -*- encoding:utf-8 -*-
import logging
import logging.config

logging.config.fileConfig("./logging.conf")

# create logger
logger_name = "example"
logger = logging.getLogger(logger_name)

logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')
```

##### 字典配置

有兴趣的童靴可以使用```logging.config.dictConfig(config)```编写一个示例程序发给我，以提供给我进行完善本文。

##### 监听配置

有兴趣的童靴可以使用```logging.config.listen(port=DEFAULT_LOGGING_CONFIG_PORT)```编写一个示例程序发给我，以提供给我进行完善本文。

更多详细内容参考[logging.config日志配置](http://python.usyiyi.cn/python_278/library/logging.config.html#module-logging.config)

### 参考资料

- [英文Python logging HOWTO](https://docs.python.org/2/howto/logging.html#logging-basic-tutorial)
- [中文Python 日志 HOWTO](http://python.usyiyi.cn/python_278/howto/logging.html#logging-basic-tutorial)
- [Python日志系统Logging](http://www.52ij.com/jishu/666.html)
- [logging模块学习笔记：basicConfig配置文件](http://www.cnblogs.com/bjdxy/archive/2013/04/12/3016820.html)
