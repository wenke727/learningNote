# SPARK

## graphframes

### 安装

- spark install path: `/usr/local/spark/bin`

Ref:

- [正确打开GraphFrames](https://zhuanlan.zhihu.com/p/130970313)
- [java.lang.ClassNotFoundException: org.graphframes.GraphFramePythonAP](https://blog.csdn.net/qq_42166929/article/details/105983616)
- [Java gateway process exited before sending its port number解决方法](https://www.codeprj.com/blog/9c440a1.html)
- [Spark应用依赖jar包的添加解决方案](https://blog.csdn.net/u012369535/article/details/90485805)
- [Spark-GraphFrames入门使用示例](https://blog.csdn.net/weixin_44275063/article/details/106072696)

```bash
# 1.目前GraphFrames库还没有并入Spark项目中，使用该库时，要安装GraphFrames包：
$pyspark --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11

# 2.使用SparkConf的spark.jars.packages属性指定依赖包：
from pyspark import SparkConf
conf = SparkConf().set('spark.jars.packages' ,'graphframes:graphframes:0.5.0-spark2.1-s_2.11')

# 3.在SparkSession中配置：（Spark2.x版本）
from pyspark.sql import SparkSession
spark = SparkSession.builder.config('spark.jars.packages' ,'graphframes:graphframes:0.5.0-spark2.1-s_2.11') .
```

```bash
pip3 install graphframes

# 3.copy jar包到Python site-packages/pyspark/jars 中
cp ./graphframes-0.8.1-spark2.4-s_2.12.jar /home/pcl/.local/lib/python3.6/site-packages/pyspark/jars/

# 4.使用参数启动pyspark，以便下载所有graphframe的jars依赖项
# 终端显示如下：Ivy Default Cache set to: /root/.ivy2/cache; 将路径中的jar包复制到cp /Users/qudian/.ivy2/jars/* .
cd /home/pcl/.local/lib/python3.6/site-packages/pyspark/jars/
pyspark --packages graphframes:graphframes:0.8.1-spark2.4-s_2.12 --jars graphframes-0.8.1-spark2.4-s_2.12.jar 

# 5.第二次启动pyspark
pyspark --packages graphframes:graphframes:0.8.1-spark2.4-s_2.12 --jars graphframes-0.8.1-spark2.4-s_2.12.jar 

# 6.在jupyter中使用需要添加路径
vi .zshrc
# export PYSPARK_DRIVER_PYTHON=jupyter
# export PYSPARK_DRIVER_PYTHON_OPTS=notebook

# 7.启动jupyter，测试
pyspark --packages graphframes:graphframes:0.8.1-spark2.4-s_2.12



pyspark --packages graphframes:graphframes-0.8.1-spark2.4-s_2.12 --repositories http://maven.aliyun.com/nexus/content/groups/public
pyspark --jars graphframes-0.8.1-spark2.4-s_2.12.jar 
```

```python
localVertices = [(1,"A"), (2,"B"), (3, "C")]
localEdges = [(1,2,"love"), (2,1,"hate"), (2,3,"follow")]
v = sqlContext.createDataFrame(localVertices, ["id", "name"])
e = sqlContext.createDataFrame(localEdges, ["src", "dst", "action"])
g = GraphFrame(v, e)
```
