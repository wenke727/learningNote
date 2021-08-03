
# [multiprocessing](https://docs.python.org/zh-cn/3.8/library/multiprocessing.html)

## Refs

- [python并行计算（上）：multiprocessing、multiprocess模块](https://zhuanlan.zhihu.com/p/46798399)
- [LintCode 并行计算](https://www.lintcode.com/problem/?typeId=9)

## 注意事项

``` python
#  部分是必需的解释
if __name__ == '__main__'
```

## Demo

- 子进程有返回值，且返回值需要集中处理，则建议采用map方式

``` python
def f(a): #map方法只允许1个参数
    pass

pool = multiprocessing.Pool() 
result = pool.map_async(f, (a0, a1, ...)).get()
pool.close()
pool.join() 
```

- 借助`partial`函数实现map方法多个参数

``` python
from multiprocessing import Pool, Lock
from functools import partial

step_lst.sort()

# ref: https://www.cnblogs.com/c-x-a/p/9049651.html
f = partial(shp_parser, agrs=args)

pools = Pool(16)
pools.map_async(f, step_lst)
pools.close()
pools.join()
```

## LintCode

- [2503 · 实现一个线程安全的计数器 counter](https://www.lintcode.com/problem/2503/)

``` python
from threading import Lock

class ThreadSafeCounter:
    def __init__(self):
        self.i = 0
        self.lock = Lock()

    def incr(self, increase: callable) -> None:
        with self.lock:
            self.i = increase(self.i)

    def decr(self, decrease: callable) -> None:
        with self.lock:
            self.i = decrease(self.i)

    def get_count(self):
        return self.i
```

- [2496 · 4 个线程修改同一个变量](https://www.lintcode.com/problem/2496/solution/33974)

  - 锁： 由于是修改同一个变量，为了线程安全，我们必须在每次求改操作的时候都加锁
  - 信号量： 使用一个信号量，计数器初始为 1，每次需要修改操作的时候将计数器用 acquire() 减少，使它变成 0 阻塞，完成修改后调用 release() 方法增加计数器，最终完成修改操作

``` python
from threading import Semaphore
class VariableModification:
    def __init__(self):
        self.i = 0
        self.lock = Semaphore()

    def add_1(self, increase: callable) -> None:
        self.lock.acquire()
        self.i = increase(self.i)
        self.lock.release()

    def add_2(self, increase: callable) -> None:
        self.lock.acquire()
        self.i = increase(self.i)
        self.lock.release()

    def sub_1(self, decrease: callable) -> None:
        self.lock.acquire()
        self.i = decrease(self.i)
        self.lock.release()
            

    def sub_2(self, decrease: callable) -> None:
        self.lock.acquire()
        self.i = decrease(self.i)
        self.lock.release()

    def check_i(self):
        return self.i
```

- []()

``` python
```
