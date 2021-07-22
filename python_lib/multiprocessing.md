
# [multiprocessing](https://docs.python.org/zh-cn/3.8/library/multiprocessing.html)

Refs：

- [python并行计算（上）：multiprocessing、multiprocess模块](https://zhuanlan.zhihu.com/p/46798399)

注意事项

    ``` python
    #  部分是必需的解释
    if __name__ == '__main__'
    ```

Demo

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
