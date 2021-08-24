# Postgre SQl

## 优化

- [PostgreSQL 空间st_contains，st_within空间包含搜索优化 - 降IO和降CPU(bound box)](hhttps://developer.aliyun.com/article/228263)
  - 优化手段1 - 空间聚集
    当数据存放与索引顺序的线性相关性很差时, 会引入IO放大
  
  - 优化手段2 - 空间分裂查询
    空间索引实际上是针对bound box的，所以在有效面积占比较低时，可能圈选到多数无效数据，导致IO和CPU同时放大，我们就来解决它。

[PostGIS官方教程汇总目录](https://blog.csdn.net/qq_35732147/article/details/85256640)

1. [PostGIS介绍](https://blog.csdn.net/qq_35732147/article/details/85158177)
   空间数据库将空间数据和对象关系数据库（Object Relational database）完全集成在一起。实现从以GIS为中心向以数据库为中心的转变。

2. [PostGIS的安装](https://blog.csdn.net/qq_35732147/article/details/86299060)

3. [创建空间数据库](https://blog.csdn.net/qq_35732147/article/details/85226864)

4. [加载空间数据](https://blog.csdn.net/qq_35732147/article/details/85228444)

5. [数据](https://blog.csdn.net/qq_35732147/article/details/85242296)

6. [简单的SQL语句](https://blog.csdn.net/qq_35732147/article/details/85243978)

7. [几何图形（Geometry）](https://blog.csdn.net/qq_35732147/article/details/85258273)

8. [关于几何图形的练习](https://blog.csdn.net/qq_35732147/article/details/85338695)

9. [空间关系](https://blog.csdn.net/qq_35732147/article/details/85615057)

10. [空间连接](https://blog.csdn.net/qq_35732147/article/details/85676670)

11. [空间索引](https://blog.csdn.net/qq_35732147/article/details/86212840)

12. [投影数据](https://blog.csdn.net/qq_35732147/article/details/86301242)

13. [地理](https://blog.csdn.net/qq_35732147/article/details/86489918)

14. [几何图形创建函数](https://blog.csdn.net/qq_35732147/article/details/86576507)

15. [更多的空间连接](https://blog.csdn.net/qq_35732147/article/details/86606486)

16. [几何图形的有效性](https://blog.csdn.net/qq_35732147/article/details/86620358)

17. [相等](https://blog.csdn.net/qq_35732147/article/details/87343551)

18. [线性参考](https://blog.csdn.net/qq_35732147/article/details/87450027)

19. [索引集群](https://blog.csdn.net/qq_35732147/article/details/88048758)

20. [3-D](https://blog.csdn.net/qq_35732147/article/details/88099418)

21. [最近邻域搜索](https://blog.csdn.net/qq_35732147/article/details/88219928)
