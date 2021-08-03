# 轨迹相关文献和算法

## 学习资料

### [数据挖掘技术在轨迹数据上的应用实践](https://zhuanlan.zhihu.com/p/259876086)

### [利用轨迹数据自动校准道路交叉口拓扑信息](https://zhuanlan.zhihu.com/p/136783117)

  [Paper: Automatic Calibration of Road Intersection Topology using Trajectories](https://ieeexplore.ieee.org/abstract/document/9101626)
  创新性地提出了一种交叉路口三阶段校准算法框架-CITT。CITT首先将道路交叉口检测问题扩展为道路交叉口影响区的拓扑校准问题。与现有的道路交叉口更新方法不同，该方法不仅确定道路交叉口核心区的中心位置和覆盖范围，同时挖掘出路口与邻接路段的转向路径，之后与现有路网进行匹配，找出整个影响区内的错误或缺失的转向模式。大量的基于滴滴实际数据和公开数据的对比实验表明，CITT方法具有很强的稳定性和鲁棒性，并且明显优于现有方法。

- 抽象问题：路口范围内的轨迹矢量模式与路网是否匹配？

- 框架
  ![Data Frame]("./../fig/Overview%20of%20Calibration%20Framework.png)

- 关键问题

  - 第一，轨迹数据包含了大量噪声，如何进行有效去噪；
  - 第二，路口位置及范围如何确定；
  - 第三，轨迹矢量模式如何表达以及如何与路网差分

- 关键模块
  - `轨迹质量提升`
    基于轨迹点的密度（时间密度、空间密度）进行数据过滤，并对局部自相交轨迹段进行分段，最后通过Douglas Peucker算法提取轨迹段关键形状点，在保留轨迹转向特征的同时，对数据实现了压缩。因此，通过轨迹分段、去噪、压缩的预处理，实现了对原始轨迹数据的质量提升。
  - 路口影响区域检测
  - `拓扑结构校准模块`
    在路口范围拓扑结构的校准阶段，我们基于检测的路口中心位置和核心区范围向外扩展，获取交叉路口影响区内的全部轨迹。我们对这些轨迹进行转向簇提取与中心线拟合，并将拟合的转向路径与基准路网进行地图匹配。Frechet距离适于评测曲线之间的相似性，但是对于复杂形状的路口以及路口邻接路段间朝向偏差较小的情况，Frechet表现不佳。鉴于此，我们将方向权重引入轨迹相似性度量中。对于任意两条轨迹序列，分别计算起点与终点间的方向差，并结合Frechet距离生成轨迹集合的距离矩阵。基于该矩阵结合DBSCAN聚类实现路口范围内的转向簇提取。
    ![Turning Clusters]("./../fig/Turning%20Clusters.jpg)

    在提取转向簇后，需要对各簇轨迹进行拟合来得到转向矢量模式。我们采用基于Force Attraction的聚类方法获取各簇对应的转向路径，相比如其他依赖点信息的拟合算法如Sweeping等，Force Attraction方法能够充分运用轨迹线信息，因此对复杂转向场景的拟合更加鲁棒。Force Attraction方法首先随机采样簇中的一条轨迹作为参考轨迹，随后使用同簇内其余轨迹对参考轨迹中点的位置进行迭代调整。在调整过程中，Force Attraction算法假定任意轨迹点上有吸引力和排斥力作用，通过搜索两个力达到平衡的位置来获得参考轨迹对应点的新位置。由于随机采样轨迹容易导致拟合得到的中心线不精准，特别是当随机采样的参考轨迹远离实际道路中心时，拟合偏差较大。因此，我们引入基于Frechet的采样策略。具体来说，我们从簇中随机采样k条轨迹作为候选参考轨迹，并分别计算每个候选者与该簇的其余轨迹之间的Frechet距离。将具有最小距离和的候选轨迹视为参考轨迹。

    在获得转向路径后，我们采用经典的HMM算法结合基准路网进行地图匹配。为加速匹配过程，我们基于每个路口的转向路径集生成凸包再与路网空间关联。根据匹配概率得到低置信度转向路径，作为需修正拓扑情报

### [Map-Matching for Low-Sampling-Rate GPS Trajectories](https://www.microsoft.com/en-us/research/publication/map-matching-for-low-sampling-rate-gps-trajectories/)

地图匹配是将观察到的用户位置序列与数字地图上的道路网络对齐的过程，大多数当前的地图匹配算法仅处理高采样率（10-30秒一个点）的GPS数据，但是实际上存在大量的低采样率的GPS轨迹。本文主要针对低采样率的GPS轨迹提出一种新颖的全局地图匹配算法，称为ST-Matching，其考虑两方面：

1. 道路网络的空间几何和拓扑结构
1. 轨迹的速度/时间约束。

基于时空分析，构建候选图，从中确定最佳匹配路径。并与增量算法和基于平均弗里切特（AFD）的全局地图匹配算法进行比较，实验在真实数据集和合成数据集上进行。提高了匹配精度，准确性，运行时间。

- 框架
  ![Frame]("./../fig/Overview_of_system_architecture.png")

## 技术帖子

- [高德：AI在出行场景的应用实践：路线规划、ETA、动态事件挖掘…](https://mp.weixin.qq.com/s?__biz=Mzg4MzIwMDM5Ng==&mid=2247486000&idx=1&sn=849f844bf445fa7545dd1ffa74b115a3&scene=21#wechat_redirect)

- [YOLOV3-SORT实现车辆跟踪与车流统计-学习教程](https://zhuanlan.zhihu.com/p/351577881)
  
## 相关代码

- 方位角计算
  
``` python
# 计算方位角函数
import math
def azimuthAngle(x1, y1, x2, y2):
    angle = 0.0
    dx, dy = x2 - x1, y2 - y1

    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)

    return (angle * 180 / math.pi)
```

- 计算向量夹角

```python
import math 

AB = [1,-3,5,-1]
CD = [4,1,4.5,4.5]
EF = [2,5,-2,6]
PQ = [-3,-4,1,-6]

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

ang1 = angle(AB, CD)
print("AB和CD的夹角")
print(ang1)
ang2 = angle(AB, EF)
print("AB和EF的夹角")
print(ang2)
ang3 = angle(AB, PQ)
print("AB和PQ的夹角")
print(ang3)
ang4 = angle(CD, EF)
print("CD和EF的夹角")
print(ang4)
ang5 = angle(CD, PQ)
print("CD和PQ的夹角")
print(ang5)
ang6 = angle(EF, PQ)
print("EF和PQ的夹角")
print(ang6)
```
