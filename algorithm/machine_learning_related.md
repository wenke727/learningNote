# Machine Learning

- [2714 · 手动实现梯度提升树（GBDT）算法完成乳腺癌患病预测](https://www.lintcode.com/problem/2714/solution/35868)

    ```python
    from numpy import ndarray
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    class GBDT():
        def __init__(self, max_tree_num = 3):
            self.tree_list = []
            self.max_tree_num = max_tree_num
            
        def fit(self, x, y):
            residual = y
            for i in range(self.max_tree_num):
                model = DecisionTreeClassifier()
                model.fit(x, residual)
                self.tree_list.append(model)
                prediction = model.predict(x)
                residual = residual - prediction
        
        def predict(self, x):
            y = np.zeros(x.shape[0])
            for model in self.tree_list:
                new_pred = np.array(model.predict(x))
                y += new_pred
                print('new_pred', new_pred)

            return y
    ```

- [2607 · 手写K-means算法实现鸢尾花数据集的聚类](https://www.lintcode.com/problem/2607/solution/35136)

  1. 初始化聚类中心
    根据给定的k值随机选取k个样本作为初始的聚类中心。
  2. 根据聚类中心划分簇
    计算每个样本与各个聚类中心之间的距离，把每个样本分配给距离他最近的聚类中心。
  3. 重新选择聚类中心
    将每个聚类中所有样本的平均值确定为新的聚类中心
  4. 停止移动
    重复第2步，直到聚类中心不再移动或达到最大迭代次数为止。

    ```python
    from numpy import ndarray
    import numpy as np

    class KMeans(object):
        def dis(self, x: ndarray, y: ndarray) -> float:
            return np.sqrt(np.sum(np.power(x - y, 2)))

        def k_means(self, x: ndarray, k: int, epochs=300, delta=1e-4) -> list:
            indices = np.random.randint(0, len(x), size=k)
            centers = x[indices]

            # 保存分类结果、 对应索引
            results，index = [], []
            for i in range(k):
                results.append([])
                index.append([])

            step = 1
            flag = True
            while flag:
                if step > epochs:
                    return index
                else:
                    # 合适的位置清空
                    for i in range(k):
                        results[i] = []
                        index[i] = []

                # 将所有样本划分到离它最近的中心簇
                for i in range(len(x)):
                    cur = x[i]
                    min_dis, tmp = np.inf, 0
                    for j in range(k):
                        distance = self.dis(cur, centers[j])
                        if distance < min_dis:
                            min_dis = distance
                            tmp = j

                    results[tmp].append(cur)
                    index[tmp].append(i)

                # 更新中心
                for i in range(k):
                    old_center = centers[i]
                    new_center = np.array(results[i]).mean(axis=0)
                    if self.dis(old_center, new_center) > delta:
                        centers[i] = new_center
                        flag = False

                if flag:
                    break
                else:
                    flag = True
                    
                step += 1

            return index
    ```

- [手动实现决策树算法——完成计算Gini指数函数](https://www.lintcode.com/problem/2629/)

    ```python
    from numpy import ndarray

    def cal_gini(data_vector: ndarray) -> float:
        # 数据集样本数
        nums_data = len(data_vector)
        # 用来保存每个label下的样本数
        counts_by_labels = {} 
        #每个类别的样本数
        p_sum=0       
        gini = 0 
    
        for vector in data_vector:
            if vector[-1] not in counts_by_labels:  # vector[-1]为label值
                counts_by_labels[vector[-1]] = 0
            
            counts_by_labels[vector[-1]] += 1  # 统计label出现的次数
        
        for key in counts_by_labels:
            p = float(counts_by_labels[key] / nums_data)  # 计算每个标签出现的概率
            p_sum += p**2
        
        gini = 1 - p_sum        

        return gini
    ```

----

- []()

```python
```
