# Spark

Spark 是一个分布式编程模型，用户可以在其中指定转换操作（ transformation）。 多次转换操作后建立起指令的有向无环图。 指令图的执行过程作为一个作业（ job）由一个动作操作（ action）触发，在执行过程中一个作业被分解为多个阶段（ stage） 和任务（ task） 在集群上执行。转换操作和动作操作操纵的逻辑结构是 DataFrame 和Dataset ，执行一次转换 操作会都会创建一个新的 DataFrame 或Dataset， 而动作操作则会触发计算，或者将 DataFrame 和 Dataset 转换成本地语言类型。

## 2 Spark 浅析

- 转换操作(transformation)
  Spark核心数据结构在计算过程中是保持不变的；但需要更改DataFrame的时候，需要告诉Spark如何执行修改，这个过程就是转换
  - 宽依赖的操作
    每个输入分区决定了多个输出分区。如：洗牌（shuffle）操作
  - 窄依赖的操作
    每个输入分区仅决定一个输出分区的转换。
    如果是窄依赖，Spark将自动执行流水线操作

- 惰性评估

  等到绝对需要的时候才执行操作

- 动作操作
  转换操作使我们能够建立逻辑转换计划。为了触发计算，我们需要运行一个动作操作action。一个动作指示Spark在一系列转换操作后计算一个结果
  - count
  - take

## 3 Spark工具集

## 4 结构化API概述

- Schema 数据模式
定义了 DataFrame 的列名和 类型 可以手动定义或者从数据源读取模式（通常定义为模式读取） 。 Schema 数据模式需要指定数据类型，这意味着 你 需要指定在什么地方放置什么类型的数据

- 结构化API 执行概述

  1. 编写 DataFrame / Dat aset / SQL 代码。
  2. 如果代码能有效执行， Spark 将其转换为一个`逻辑执行计划`（ Logical Plan ）。
     这个逻辑计划仅代表一组抽象转换，并不涉及执行器 或 驱动器 ，它只是将用户的表达式集合转换为最优的版本。它通过将用户代码转换为 未解析的逻辑计划来实现这一点.
  3. Spark 将此逻辑执行计划转化为一个`物理执行计划`（ Physical Plan ），并检查可行的优化策略，并在此过程中检查优化。
     在成功创建优化的逻辑计划后，Spark 开始执行物理计划流程。物理计划（通常称为 Spark 计划）通过生成 不同的物理执行策略，并通过代价模型进行比较分析，从而指定如何在集群上执行逻辑计划。例如执行一个连接操作就会涉及到代价比较，它通过分析数据表的物理属性（表的大小或分区的大小），对不同的物理执行策略进行代价比较.
  4. 然后，Spark 在集群上执行该物理执行计划（ RDD 操作）。

## 5 基础结构化操作

### 5.1 模式

模式定义 DataFrame的列名以及列的数据类型 ，它可以由数据源来定义模式（称为读时模式schemaschema-on-readread），也可以由我们自己来显式地定义。

```python
from pyspark.sql.types import StructField, StructType, StringType, LongType

spark.read.format("json").load("../data/flight-data/json/2015-summary.json").schema

myManualSchema = StructType([
  StructField("DEST_COUNTRY_NAME", StringType(), True),
  StructField("ORIGIN_COUNTRY_NAME", StringType(), True),
  StructField("count", LongType(), False, metadata={"hello":"world"})
])
df = spark.read.format("json").schema(myManualSchema)\
          .load("../data/flight-data/json/2015-summary.json")
```

### 5.2 列和表达式

Spark中的列与电子表格、Rdataframe或pandasDataFrame中的列类似，可以对DataFrame中的列进行选择、转换操作和删除，并将这些操作表示为表达式。
对于Spark而言，列是逻辑结构，它只是表示根据表达式为每个记录计算出的值。这意味着要为一个列创建一个真值，我们需要有一个行而要有一个行，则需要有一个DataFrame。你不能在DataFrame的范围外操作一个列必须对DataFrame使用Spark的转换操作来修改列的内容。

``` python
from pyspark.sql.functions import col, column, expr
col("someColumnName")
column("someColumnName")

expr("(((someCol + 5) * 200) - 6) < otherCol")
```

### 5.3 记录和行

在Spark中，DataFrame的每一行都是一个记录，而记录是Row类型的对象。Spark使用列表达式操纵Row类型对象。Row对象内部其实是字节数组，但是Spark没有提供访问这些数组的接口，因此我们只能使用列表达式去操纵。

``` python
# 创建一行
from pyspark.sql import Row
myRow = Row("Hello", None, 1, False)
```

### 5.4 DF转换操作

``` python
from pyspark.sql import Row
from pyspark.sql.functions import desc, asc
from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.sql.functions import expr, col, column, lit

# 创建DF
df = spark.read.format("json").load("../data/flight-data/json/2015-summary.json")
df.createOrReplaceTempView("dfTable")

myManualSchema = StructType([
    StructField("some", StringType(), True),
    StructField("col", StringType(), True),
    StructField("names", LongType(), False)
])
myRow = Row("Hello", None, 1)
myDf = spark.createDataFrame([myRow], myManualSchema)
myDf.show()

# select函数：单列、多列
df.select("DEST_COUNTRY_NAME").show(2)
df.select("DEST_COUNTRY_NAME", "ORIGIN_COUNTRY_NAME").show(2)
df.select(expr("DEST_COUNTRY_NAME"), col("DEST_COUNTRY_NAME"), column("DEST_COUNTRY_NAME")).show(2)

df.select(expr("DEST_COUNTRY_NAME AS destination")).show(2)
df.select(expr("DEST_COUNTRY_NAME as destination").alias("DEST_COUNTRY_NAME")).show(2)

df.selectExpr("DEST_COUNTRY_NAME as newColumnName", "DEST_COUNTRY_NAME").show(2)
df.selectExpr(
  "*", # all original columns
  "(DEST_COUNTRY_NAME = ORIGIN_COUNTRY_NAME) as withinCountry")\
  .show(2)

df.selectExpr("avg(count)", "count(distinct(DEST_COUNTRY_NAME))").show(2)

# 字面量: 转换操作成Spark类型
df.select(expr("*"), lit(1).alias("One")).show(2)
df.withColumn("numberOne", lit(1)).show(2)

# 添加列， `转移字符
df.withColumn("withinCountry", expr("ORIGIN_COUNTRY_NAME == DEST_COUNTRY_NAME"))
dfWithLongColName.selectExpr("`This Long Column-Name`", "`This Long Column-Name` as `new col`").show(2)
dfWithLongColName.select(expr("`This Long Column-Name`")).columns

# 过滤
df.where(col("count") < 2).where(col("ORIGIN_COUNTRY_NAME") != "Croatia")

# 获取去重后的行
df.select("ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME").distinct().show()
df.select("ORIGIN_COUNTRY_NAME").distinct().count()


# 随机抽样
seed = 5; withReplacement = False; fraction = 0.5
df.sample(withReplacement, fraction, seed).count()

# 随机分割
dataFrames = df.randomSplit([0.25, 0.75], seed)
dataFrames[0].count() > dataFrames[1].count() # False

# 连接和追加，联合操作是基于位置的
schema = df.schema
newRows = [
  Row("New Country", "Other Country", 5),
  Row("New Country 2", "Other Country 3", 1)
]
parallelizedRows = spark.sparkContext.parallelize(newRows)
newDF = spark.createDataFrame(parallelizedRows, schema)

df.union(newDF)\
  .where("count = 1")\
  .where(col("ORIGIN_COUNTRY_NAME") != "United States")\
  .show()

# 行排序
df.sort("count").show(5)
df.orderBy("count", "DEST_COUNTRY_NAME").show(5)
df.orderBy(col("count"), col("DEST_COUNTRY_NAME")).show(5)
df.orderBy(expr("count desc")).show(2)
df.orderBy(col("count").desc(), col("DEST_COUNTRY_NAME").asc()).show(2)

# 出于性能优化的目的，最好在进行别的转换之前，先对每个分区进行内部排序
spark.read.format("json").load("../data/flight-data/json/*-summary.json")\
  .sortWithinPartitions("count")

# limit方法
df.limit(5).show()

# 重划分和合并
df.rdd.getNumPartitions() # 1
df.repartition(5)
# 优化方案：针对一些经常过滤的列对数据进行划分，控制跨群集数据的物理布局，包括分区方案和分区数
df.repartition(col("DEST_COUNTRY_NAME"))
df.repartition(5, col("DEST_COUNTRY_NAME"))
df.repartition(5, col("DEST_COUNTRY_NAME")).coalesce(2)

# 驱动器获取行
collectDF = df.limit(10)
collectDF.take(5) # take works with an Integer count
collectDF.show() # this prints it out nicely
collectDF.show(5, False)
collectDF.collect()
```

## 6 处理不同的类型

## 7 聚合操作

### 7.1 聚合函数

- count
- countDistinct
- approx_count_distinct
- first/last
- min/max
- sum
- sumDistinct
- avg
- 方差和标准差
- skewnwss, kurtosis
- 协方差和相关性

### 7.2 分组

``` python
df.groupby("col").count().show()
```

## 8 连接操作

## 9 数据源

## 10 Spark SQL

## 11 Dataset

----

## 12 RDD 弹性分布式数据集

## 13 高级RDD

## 14 分布式共享变量

----

## 15

----

## 18 监控和调试

- 任务缓慢或落后者
  `大多数情况都是某种数据倾斜导致的，所以最好先在Spark UI上检查跨任务的不均匀负载`
  此问题在优化应用程序时非常常见，这可能是由于工作负载 没有被 均匀分布在集群各 节点上（导致负载“倾斜”），或者是由于某台计算 节点 比其他计算 节点 速度慢（例如，由于硬件问题） 。
  缓慢任务通常被称为“落后者”，有缓慢任务通常被称为“落后者”，有很多原因会导致缓慢任务很多原因会导致缓慢任务，但最常见的原因是你的数据不均，但最常见的原因是`数据不均匀地分布`到DataFrame或RDD分区上。发生这种情况时，一些分区上。发生这种情况时，一些执行器执行器节点可能需要比其他节点可能需要比其他执行执行器器节点更多的工作量。一个特别常见的情况是，你使用按键节点更多的工作
  - 应对措施

    - 尝试增加分区数以减少每个分区被分配到的数据量。
    - 尝试通过另一种列组合来重新分区。例如，当你使用ID列进行分区时，如果ID是倾斜分布的，那么就容易产生落后者。或者当你使用存在许多空值的列进行分区时，许多对应空值列的行都被集中分配到一台节点上，也会造成落后者，在后一种情况下，首先筛选出空值可能会有所帮助。
    - 尽可能分配给执行器进程更多的内存。
    - 监视有缓慢任务的执行器节点，并确定该执行器节点在其他作业上也总是执行缓慢任务这说明集群中可能存在一个不健康的执行器节点，例如，磁盘空间不足的节点。
    - 如果在执行连接join）操作或聚合aggregation操作时产生缓慢任务，请参阅309页的缓慢连接操作和308页的“缓慢聚合操作。
    - 检查用户定义函数（UDF）是否在其对象分配或业务逻辑中有资源浪费的情况。如果可能，尝试将它们转换为DataFrame代码。
    - 确保你的UDF或用户定义的聚合函数（UDAF）在足够小的数据上可以运行。通常情况下，聚合操作要将大量数据存入内存以处理对某个key的聚合操作，从而导致该执行器比其他执行器要完成更多的工作。
    - 打开推测执行（speculation）功能，这将为缓慢任务在另外一台节点上重新运行一个任务副本，关于该功能将在310页的缓慢读写问题中介绍。如果缓慢问题是由于硬件节点的原因，推测执行功能将会有所帮助，因为任务会被迁移到更快的节点上运行。然而，推测执行也会付出代价，第一是因为它会消耗额外的资源，另外，对于一些使用最终一致性的存储系统，如果写操作不是幂等的，则可能会产生重复冗余的输出数据。（第17章讨论了推测执行的具体配置。）
    - 使用Dataset时可能会出现另一个常见问题。由于Dataset执行大量的对象实例化并将记录转换为用户定义函数中的Java对象，这可能会导致大量垃圾回收。如果你使用Dataset请查看SparkUI中的垃圾回收指标，以确定它们是否是导致缓慢任务的原因

- 缓慢的聚合操作

  - 在聚合操作之前增加分区数量可能有助于减少每个任务中处理的不同key的数量。
  - 增加执行器进程的内存配额也可以帮助缓解此问题。如果一个key拥有大量数据，这将允许其执行器进程更少地与磁盘交互数据并更快完成任务，尽管它可能仍然比处理其他key的执行器进程要慢得多。
  - 如果你发现聚合操作之后的任务也很慢，这意味着你的数据集在聚合操作之后可能仍然不均衡。尝试调用repartition并对数据进行随机重新分区。
  - 确保涉及的所有过滤操作和select操作在聚合操作之前完成这样可以保证只对需要执行聚合操作的数据进行处理避免处理无关数据。Spark的查询优化器将自动为结构化API执行此操作。
  - 确保空值被正确地表示（建议使用Spark的null关键字），不要用”或EMPTY”之类的空值表示。Spark优化器通常会在作业执行初期来跳过对null空值的处理，但它无法为你自己定义的空值形式进行此优化。
  - 一些聚合操作本身也比其他聚合操作慢。例如，collect_list和collect_set是非常慢的聚合函数因为它们必须将所有匹配的对象返回给驱动器进程，所以在代码中应该尽量避免使用这些聚合操作

- 缓慢的读写操作缓

  - 开启推测执行（将spark.speculation设置为true）有助于解决缓慢读写的问题。推测执行功能启动执行相同操作的任务副本，如果第一个任务只是一些暂时性问题，推测执行可以很好地解决读写操作慢的问题。推测执行是一个强大的工具，与支持数据一致性的文件系统兼容良好。但是，如果使用支持最终一致性的云存储系统，例如AmazonS3，它可能会导致重复的数据写入，因此请检查你使用的存储系统连接器是否支持。
  - 确保网络连接状态良好。你的Spark集群可能因为没有足够的网络带宽而导致读写存储系统缓慢。
  - 如果在相同节点上运行Spark和HDFS等分布式文件系统，确保Spark与文件系统的节点主机名相同。Spark将考虑数据局部性进行任务调度，用户将在SparkUI的“locality”列中看到该调度情况。我们将在下一章讨论更多关于数据局部性的问题数据局部性的问题

## 19 性能调优

需要优先考虑的主要因素包括：

- 尽可能地通过分区和高效的二进制格式来读取较少的数据
- 确保使用分区的集群具有足够的并行度并通过分区方法减少数据倾斜
- 尽可能多地使用诸如结构化API之类的高级API来使用已经优化过的成熟代码

### 19.1 间接性能优化

- 设计选择
- 集群配置
- 静息数据
- 基于文件的长期数据存储
- 表分区

### 19.2 直接性能优化

- 增加并行度 -> spark.default.partitions
- 避免使用udf
- 临时数据存储（cache）
  在重复使用数据集的应用程序上，最有用的优化之一是缓存
- 连接操作
  尽量优先选择等值连接，避免笛卡尔连接或避免完全外连接提升稳定性和性能
- 聚合操作
  聚合之前过滤数据
- 广播变量
