# [logbook](https://logbook.readthedocs.io/en/stable/quickstart.html)

- [python 日志模块--python logbook使用方法](https://xiaofan.blog.csdn.net/article/details/100927527?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-4.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-4.control)
- [logbook日志系统](https://blog.csdn.net/xuxunxiong954/article/details/84818524)

## [Log File Highlighter](https://marketplace.visualstudio.com/items?itemName=emilast.LogFileHighlighter)

Adds color highlighting to log files to make it easier to follow the flow of log events and identify problems.

``` JSON
"editor.tokenColorCustomizations": {
    "textMateRules": [
        {
            "scope": "log.error",
            "settings": {
                "foreground": "#af1f1f",
                "fontStyle": "bold"
            }
        },
        {
            "scope": "log.warning",
            "settings": {
                "foreground": "#f4ad42",
                "fontStyle": ""
            }
        }
    ]
}
```

## level

|level      | describe             |
|--         |--                    |
|critical   |严重错误，会导致程序退出|
|error      |可控范围内的错误       |
|warning    |警告信息              |
|notice     |大多情况下希望看到的记录|
|info       |大多情况不希望看到的记录|
|debug      |调试程序时详细输出的记录|

## module

logbook的日志输出方式有2种：打印到屏幕（比较适合调试时候，正式使用时可以将其注释掉）和打印输出到日志文件

```python
import os
import sys
import time
import logbook
from logbook import Logger, TimedRotatingFileHandler
from logbook.more import ColorizedStderrHandler

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(FILE_DIR, '../log/')
logbook.set_datetime_format('local')


def log_type(record, handler):
    log_info = "[{date}] [{level}] [{filename}] [{func_name}] [{lineno}]\n{msg}".format(
        date=record.time,                              # 日志时间
        level=record.level_name,                       # 日志等级
        filename=os.path.split(record.filename)[-1],   # 文件名
        func_name=record.func_name,                    # 函数名
        lineno=record.lineno,                          # 行号
        msg=record.message                             # 日志内容
    )
    
    return log_info


def log_type_for_std(record, handler):
    log_info = "[{date}] [{level}] [{filename}] [{func_name}] [{lineno}] {msg}".format(
        date=record.time,                              # 日志时间
        level=record.level_name,                       # 日志等级
        filename=os.path.split(record.filename)[-1],   # 文件名
        func_name=record.func_name,                    # 函数名
        lineno=record.lineno,                          # 行号
        msg=record.message                             # 日志内容
    )
    
    return log_info


class LogHelper(object):
    def __init__(self, log_dir=BASE_DIR, log_name='log.log', backup_count=10, log_type=log_type, stdOutFlag=False, std_log_type=log_type_for_std):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            
        self.log_dir = log_dir
        self.backup_count = backup_count
        
        handler = TimedRotatingFileHandler(filename= os.path.join(self.log_dir, log_name),
                                        date_format='%Y-%m-%d',
                                        backup_count=self.backup_count)
        self.handler = handler
        if log_type is not None:
            handler.formatter = log_type
        handler.push_application()

        if not stdOutFlag:
            return
        
        handler_std = ColorizedStderrHandler(bubble=True)
        if log_type is not None:
            handler_std.formatter = std_log_type
        handler_std.push_application()

    def get_current_handler(self):
        return self.handler

    @staticmethod
    def make_logger(level, name=str(os.getpid())):
        return Logger(name=name, level=level)


def log_helper(log_file, content):
    log_file.write( f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, {content}\n" )
    return 


if __name__ == "__main__":
    g_log_helper = LogHelper(log_name='log.log', stdOutFlag=True)
    log = g_log_helper.make_logger(level=logbook.INFO)
    log.critical("critical")    # 严重错误，会导致程序退出
    log.error("error")          # 可控范围内的错误 
    log.warning("warning")      # 警告信息
    log.notice("notice")        # 大多情况下希望看到的记录
    log.info("info")            # 大多情况不希望看到的记录
    log.debug("debug")          # 调试程序时详细输出的记录
    pass

```

----

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

## 字典配置

有兴趣的童靴可以使用```logging.config.dictConfig(config)```编写一个示例程序发给我，以提供给我进行完善本文。

## 监听配置

有兴趣的童靴可以使用```logging.config.listen(port=DEFAULT_LOGGING_CONFIG_PORT)```编写一个示例程序发给我，以提供给我进行完善本文。

更多详细内容参考[logging.config日志配置](http://python.usyiyi.cn/python_278/library/logging.config.html#module-logging.config)

## 参考资料

- [英文Python logging HOWTO](https://docs.python.org/2/howto/logging.html#logging-basic-tutorial)
- [中文Python 日志 HOWTO](http://python.usyiyi.cn/python_278/howto/logging.html#logging-basic-tutorial)
- [Python日志系统Logging](http://www.52ij.com/jishu/666.html)
- [logging模块学习笔记：basicConfig配置文件](http://www.cnblogs.com/bjdxy/archive/2013/04/12/3016820.html)
