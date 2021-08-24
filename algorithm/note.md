# Algorithm

Tips

- 学习算法的框架思维
- 穷尽算法：回溯算法，动态规划

## 1 数据结构

数据结构可以认为是一个数据集合以及定义在这个集合上的若干操作和功能
数据之间的关系，好的关系可以使得数据处理起来更加高效

考点：

- 某种数据结构的基本原理，并要求实现
- 使用某种数据结构完成事情
- 实现一种数据结构，提供一些特别的功能

### 1.1 队列

队列的基本操作就是用来做BFS
操作： O(1) Push / O(1) Pop / O(1)Top

例题

- [数据流滑动窗口平均值](https://www.lintcode.com/problem/moving-average-from-data-stream/description)

### 1.2 栈 Stack

- 递归转非递归， 非递归实现DFS的主要数据结构
- 利用栈暂且保存有效信息
- 翻转栈的运用

操作： O(1) Push / O(1) Pop / O(1)Top

- [带最小值操作的栈](https://www.lintcode.com/problem/min-stack/description)
- [用栈实现队列](https://www.lintcode.com/problem/implement-queue-by-two-stacks/description)
- [字符串解码](https://www.lintcode.com/problem/decode-string/description): 利用栈结构暂存信息

### 1.3 单调栈

- 找出每个元素左边或者右边第一个比它 大/小 的元素，用单调栈来维护；

模板

``` python
# 接雨水 ⭐⭐⭐
# https://www.lintcode.com/problem/trapping-rain-water/description
class Solution:
    def trap(self, heights):
        stack, ans = [], 0

        for hi, h in enumerate(heights):
            while stack and h >= heights[stack[-1]]:
                ground_height = heights[stack.pop()]
                if not stack: 
                    continue
            
                lo = stack[-1]
                water_line = min( heights[lo], h )
                ans += (water_line - ground_height) * (hi-lo-1)
           
            stack.append(hi)

        return ans
```

例题

- [下一个更大的数 II](https://www.lintcode.com/problem/1201/): % 运算模拟出环形数组
- [栈排序](https://www.lintcode.com/problem/229/)
- TODO ⭐⭐⭐[直方图最大矩形覆盖](https://www.lintcode.com/problem/largest-rectangle-in-histogram/description)
  idea: find the first smaller numer in the left, and caculate the area between them. And the answer is the maximun of these area.
- ⭐⭐⭐[最大数](https://www.lintcode.com/problem/126/)
  理解最大数的构建过程

### 1.4 单调队列

使用了一点巧妙的方法，使得队列中的元素是单调递增/减

- [滑动窗口的最大值](https://www.lintcode.com/problem/362/)

```python
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums, k):
        if not nums or not k:
            return []
            
        dq = deque([])
        
        for i in range(k - 1):
            self.push(dq, nums, i)
        
        result = []
        for i in range(k - 1, len(nums)):
            self.push(dq, nums, i)
            result.append(nums[dq[0]])
            self.pop(dq, i-k+1)
                
        return result
            
    def push(self, dq, nums, i):
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)

    def pop(self, dq, i):
        if dq[0] == i:
            dq.popleft()
```

### 1.5 哈希表

操作：O(1) Insert / O(1) Find / O(1) Delete

相关知识

- 什么是哈希function
- 什么是open hashing, close hashing
- 什么是rehashing

例题

- [LRU缓存策略](https://www.lintcode.com/problem/lru-cache/description), API是从尾部插入，靠尾部的数据是最近使用的
- [数据流中第一个唯一的数字](https://www.lintcode.com/problem/first-unique-number-in-data-stream/description)
- [O(1)实现数组插入/删除/随机访问](https://www.lintcode.com/problem/insert-delete-getrandom-o1/description)

### 1.6 堆

- 求集合的最大值

操作：O(log N) Add; O(Log N) remove; O(1) Min or Max; O(n) heapify

``` python
class Solution:
    def heapify(self, A):
        for i in range(len(A) // 2, -1, -1):
            self.siftdown(A, i)
            
    def siftdown(self, A, index):
        n = len(A)
        
        while index < n:
            left, right  = index * 2 + 1, index * 2 + 2
            minIndex = index
            
            if left < n and A[left] < A[minIndex]:
                minIndex = left
            if right < n and A[right] < A[minIndex]:
                minIndex = right

            if minIndex == index: break
            
            A[minIndex], A[index] = A[index], A[minIndex]
            index = minIndex
```

例题

- [堆化](https://www.lintcode.com/problem/130/)
- [K个最近的点](https://www.lintcode.com/problem/k-closest-points/description)
- [合并k个排序链表](https://www.lintcode.com/problem/merge-k-sorted-lists/description)
- [接雨水 II](https://www.lintcode.com/problem/trapping-rain-water-ii/description): 矩阵从外到内遍历；怎么想到利用堆
- [数据流中位数](https://www.lintcode.com/problem/find-median-from-data-stream/description)

#### 1.6.1 HashHeap

数据结构

```python
class HashHeap:
    def __init__(self, desc=False):
        self.hash = dict()
        self.heap = []
        self.desc = desc
        
    @property
    def size(self):
        return len(self.heap)
        
    def push(self, item):
        self.heap.append(item)
        self.hash[item] = self.size - 1
        self._sift_up(self.size - 1)
        
    def pop(self):
        item = self.heap[0]
        self.remove(item)
        return item
    
    def top(self):
        return self.heap[0]
        
    def remove(self, item):
        if item not in self.hash:
            return
            
        index = self.hash[item]
        self._swap(index, self.size - 1)
        
        del self.hash[item]
        self.heap.pop()
        
        # in case of the removed item is the last item
        if index < self.size:
            self._sift_up(index)
            self._sift_down(index)

    def _smaller(self, left, right):
        return right < left if self.desc else left < right

    def _sift_up(self, index):
        while index != 0:
            parent = index // 2
            if self._smaller(self.heap[parent], self.heap[index]):
                break
            self._swap(parent, index)
            index = parent
    
    def _sift_down(self, index):
        if index is None:
            return

        while index * 2 < self.size:
            smallest = index
            left = index * 2
            right = index * 2 + 1
            
            if self._smaller(self.heap[left], self.heap[smallest]):
                smallest = left
                
            if right < self.size and self._smaller(self.heap[right], self.heap[smallest]):
                smallest = right
                
            if smallest == index:
                break
            
            self._swap(index, smallest)
            index = smallest
        
    def _swap(self, i, j):
        elem1 = self.heap[i]
        elem2 = self.heap[j]
        self.heap[i] = elem2
        self.heap[j] = elem1
        self.hash[elem1] = j
        self.hash[elem2] = i
```

例题

- ⭐⭐⭐[滑动窗口的中位数](https://www.lintcode.com/problem/sliding-window-median/description)
- [大楼轮廓](https://www.lintcode.com/problem/the-skyline-problem/description)

----

### 1.7 Interval

例题

- [合并排序数组 II](https://www.lintcode.com/problem/merge-two-sorted-arrays/description)
- [合并排序数组](https://www.lintcode.com/problem/merge-sorted-array/description), 一个数组足够大，可以考虑从后往前
- [合并两个排序的间隔列表](https://www.lintcode.com/problem/merge-two-sorted-interval-lists/description)
- [合并K个排序间隔列表](https://www.lintcode.com/problem/merge-k-sorted-interval-lists/description), heap

### 1.8 Array

- [两数组的交集](https://www.lintcode.com/problem/intersection-of-two-arrays/description)
- [多个数组的交集](https://www.lintcode.com/problem/intersection-of-arrays/description), dict

### 1.9 Matrix

- [稀疏矩阵乘法](https://www.lintcode.com/problem/sparse-matrix-multiplication/description)

### 1.10 Union find

操作：O(1) find / O(1) union

模板

``` python
# https://www.lintcode.com/problem/connecting-graph/description
class UnionFind:
    def __init__(self, n):
        self.father = {}
        # other attribute, e.g., node_num, islands Num
        for i in range(n + 1):
            self.father[i] = i

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            self.father[roota] = rootb

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]
```

例题

- [连接图](https://www.lintcode.com/problem/connecting-graph/description); [连接图 II](https://www.lintcode.com/problem/connecting-graph-ii/description); [连接图 III](https://www.lintcode.com/problem/connecting-graph-iii/description)
- [岛屿的个数](https://www.lintcode.com/problem/433/); [岛屿的个数II](https://www.lintcode.com/problem/434/)
- ⭐[被围绕的区域](https://www.lintcode.com/problem/surrounded-regions/description): 用`X`替换所有不被包围的`O`
  idea: 从外围的`O`开始连接到`dummy`
- ⭐⭐[账户合并](https://www.lintcode.com/problem/accounts-merge/description )
- ⭐⭐⭐[最小生成树](https://www.lintcode.com/problem/minimum-spanning-tree/description)

### 1.11 Trie Tree

一个一个字母查找，快速判断前缀

- 利用Trie前缀特征解题
- 矩阵类字符串一个一个字符深度遍历的问题

模板

``` python
from collections import OrderedDict
class TrieNode:
    def __init__(self):
        self.children = OrderedDict()
        self.isWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for letter in word:
            node.children[letter] = node.children.get(letter, TrieNode())
            node = node.children[letter]
        node.isWord = True

    def search(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                return False
            node = node.children[letter]
        
        return node.isWord

    def startsWith(self, prefix):
        node = self.root
        for letter in prefix:
            if letter not in node.children:
                return False
            node = node.children[letter]
        
        return True
```

例题

- [实现 Trie（前缀树）](https://www.lintcode.com/problem/implement-trie-prefix-tree/description)
- [单词的添加与查找](https://www.lintcode.com/problem/add-and-search-word-data-structure-design/description)
- [单词搜索 II](https://www.lintcode.com/problem/word-search-ii/description)
- ⭐⭐⭐[单词矩阵](https://www.lintcode.com/problem/634/)

## 2 二分法

二分法常见痛点

- 循环结束条件
  - start + 1 < end
- 指针变化
  - start = mid
- 死循环的发生
  - eg: nums = [1, 1], target = 1
- 第一个/最后一个位置
  - 分三种情况讨论: <, >, =

二分法深入理解

- 根据判断，保留有解的那一半
- 二维二分
- 按照值域二分

例题：

- [在排序数组中找最接近的K个数](https://www.lintcode.com/problem/find-k-closest-elements/description)

``` python
# 61. 搜索区间 🌟
# https://www.lintcode.com/problem/search-for-a-range/description
class Solution:
    def searchRange(self, A, target):
        n = len(A) 
        if not n: 
            return [-1, -1]
        
        return [self.findFirstTargetNum(A, target, n), self.findLastTargetNum(A, target, n)]   

    def findFirstTargetNum(self, nums, target, n):
        start, end = 0, n -1

        while start + 1 < end:
            mid = (start+end) //2
            # three cases: <, = , >
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        
        if nums[start] == target: 
            return start
        if nums[end] == target: 
            return end
        
        return -1 

    def findLastTargetNum(self, nums, target, n):
        start, end = 0, n -1

        while start + 1 < end:
            mid = (start+end) //2
            if nums[mid] > target:
                end = mid
            else:
                start = mid
        
        if nums[end] == target: 
            return end
        if nums[start] == target: 
            return start
        
        return -1 
```

``` python
# 63. 搜索旋转排序数组 II ⭐⭐⭐
# https://www.lintcode.com/problem/search-in-rotated-sorted-array-ii/description
class Solution:
    def search(self, nums, target):
        if not nums: return False

        start, end = 0, len(nums)-1
        while start+1 < end:
            # ! DESC Similiar as previous problem, just keep going if duplicate
            while start + 1 < end and nums[start] == nums[start+1]:
                start += 1
            while start + 1 < end and nums[end] == nums[end-1]:
                end -= 1

            mid = (start+end)//2
            if nums[mid] == nums[end]:
                return True
            elif nums[mid] > nums[end]:
                if nums[start] <= target <= nums[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if nums[mid] <= target <= nums[end]:
                    start = mid
                else:
                    end = mid

        return target in [nums[start], nums[end]]

```

----

## 3 双指针

基础

- 同向双指针
- 反向双指针
  - Two sum类型
  - Partition: quick select
- 链表上的快慢指针
- 快速排序,  归并排序（有点递归的意思）

``` python
# 移动零
# https://www.lintcode.com/problem/move-zeroes/description 
class Solution:
    def moveZeroes(self, nums):
        left, right = 0, 0
        while right < len(nums):
            if nums[right] == 0:
                right += 1
                continue

            nums[left] = nums[right]
            right += 1
            left += 1
        
        while left < len(nums):
            if nums[left] != 0:
                nums[left] = 0
            left += 1
```

Partition 模板

```python
def partitionArray(self, nums, k):
    left, right =  0, len(nums) - 1
    while left <= right:
        while left <= right and con 应该在左边:
            left += 1
        
        while left <= right and con 应该在右边:
            right -= 1
        
        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

例题：

- ⭐⭐[无序数组K小元素](https://www.lintcode.com/problem/kth-smallest-numbers-in-unsorted-array/description)
- [排颜色](https://www.lintcode.com/problem/sort-colors/description)
  易错点：color为2的时候，和left指针对调时，cur是否需要+1
- [排颜色 II](https://www.lintcode.com/problem/sort-colors-ii/)
  Three cases: <, >, ==
  The idea of divide conquer
- [两个排序数组的中位数](https://www.lintcode.com/problem/median-of-two-sorted-arrays/description)

----

## 4 BFS

把一些问题抽象成图，从一个点开始，向四周扩散。一般来说，写BFS都使用队列这种数据结构，每次将一个节点周边的所有节点加入到队列中。

算法优化

- 启发式算法 A*
- 双向BFS
  - 传统的BFS是从起点向四周扩散，遇到终点停止；双向BFS则是从起点和终点同时开始扩散，当两边有交集的时候停止

应用场景

- 图的遍历
  - 层次遍历（size = queue.size）
  - 由点及面
  - 拓扑排序
- 最短路径
- 非递归的方式找所有方案

拓扑排序

- 统计每个点的入度
- 将每个入度为0的点放到queue作为起始节点
- 不断从队列里边取点，去掉这个点的所有连接边，然后其他的入度-1
- 一旦发现新的入度为0的点，放回到队列中

例题

- [克隆图](https://www.lintcodinfe.com/problem/clone-graph/description)
- [单词接龙](https://www.lintcode.com/problem/word-ladder/description)
- [不同岛屿的数量II](https://www.lintcode.com/problem/804/description?_from=collection&fromId=208)
- [骑士的最短路线](https://www.lintcode.com/problem/knight-shortest-path/description)

示例

``` python
# 120/127. 单词接龙
# https://www.lintcode.com/problem/word-ladder/description
from collections import deque
class Solution:
    def ladderLength(self, start, end, dict):
        dict  = set( list(dict) + [start, end] )
        queue = deque([start])
        dis   = {start:1}

        while queue:
            node = queue.popleft()
            if node == end:
                return dis[node]

            for nxtWord in self.get_next_words(node, dict):
                if nxtWord in dis:
                    continue
                dis[nxtWord] = dis[node] + 1
                queue.append(nxtWord)

        return 0


    def get_next_words(self, word, dict):
        words = []
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + char + word[i+1:]
                if char == word[i] or new_word not in dict:
                    continue
                words.append(new_word)

        return words
```

拓扑排序

``` python
# 892. 外星人词典 ⭐⭐
# https://www.lintcode.com/problem/alien-dictionary/description
from heapq import heapify, heappop, heappush
class Solution:
    def alienOrder(self, words):
        graph = self.build_graph(words)
        if graph is None:
            return ""

        return self.topological_sort(graph)

    
    def build_graph(self, words):
        graph = {}
        for w in words:
            for c in w:
                graph[c] = set()

        for i in range(len(words)-1):
            j_min = min(len(words[i]), len(words[i+1]))
            for j in range(j_min):
                if words[i][j] != words[i+1][j]:
                    graph[words[i][j]].add( words[i+1][j] )
                    break

                if j == j_min - 1 and len(words[i]) > len(words[i+1]):
                    return None

        return graph


    def topological_sort(self, graph):
        indegree = {node: 0 for node in graph}
        for node in graph:
            for nxt in graph[node]:
                indegree[nxt] += 1
        
        queue = [ node for node in indegree if indegree[node]==0 ]
        heapify(queue)

        topo_order = ''
        while queue:
            cur = heappop(queue)
            topo_order += cur

            for nxt in graph[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heappush(queue, nxt)
        
        return topo_order if len(topo_order) == len(graph) else ""
```

----

## 5 Tree

碰到二叉树的问题，就想想整棵树在该问题上的结果
和左右儿子在该问题上的结果之间的联系是什么

### 框架思维

先序：考察到一个节点后，即刻输出该节点的值，并继续遍历其左右子树。(根左右)
中序：考察到一个节点后，将其暂存，遍历完左子树后，再输出该节点的值，然后遍历右子树。(左根右)
后序：考察到一个节点后，将其暂存，遍历完左右子树后，再输出该节点的值。(左右根)

``` python
def traverse(root:TreeNode) { 
    #  前序遍历
    traverse(root.left)
    # 中序遍历 
    traverse(root.right) 
    # 后序遍历
}
```

示例

``` python
# 【前序】 翻转二叉树
# https://www.lintcode.com/problem/invert-binary-tree/description
class Solution:
    def invertBinaryTree(self, root):
        self.dfs(root)
        return root

    def dfs(self, root):
        if root is None:
            return
        
        root.left, root.right =  root.right,  root.left

        self.dfs(root.left)
        self.dfs(root.right)


# 【后序】 具有最大平均数的子树
# https://www.lintcode.com/problem/subtree-with-maximum-average/description
class Solution:
    def findSubtree2(self, root):
        _, max_root, _, _ = self.dfs(root)
        return max_root

    def dfs(self, root):
        if root is None:
            return -sys.maxsize, None, 0, 0
        
        maxAvg_l, maxTree_l, size_l, sum_l = self.dfs(root.left)
        maxAvg_r, maxTree_r, size_r, sum_r = self.dfs(root.right)

        size_cur = size_l + size_r + 1
        sum_cur  = sum_r + sum_l + root.val
        avg_cur  = sum_cur / size_cur

        if maxAvg_l == max( maxAvg_l, maxAvg_r, avg_cur ):
            return maxAvg_l, maxTree_l, size_cur, sum_cur
        if maxAvg_r == max( maxAvg_l, maxAvg_r, avg_cur ):
            return maxAvg_r, maxTree_r, size_cur, sum_cur

        return avg_cur, root, size_cur, sum_cur

# 【后序】最近公共祖先 ⭐
# https://www.lintcode.com/problem/lowest-common-ancestor-of-a-binary-tree/description
class Solution:
    def lowestCommonAncestor(self, root, A, B):
        if root is None:
            return None

        if root is A or root is B:
            return root

        left  = self.lowestCommonAncestor(root.left,  A, B)
        right = self.lowestCommonAncestor(root.right, A, B)

        if left is not None and right is not None:
            return root
        if left is not None:
            return left
        if right is not None:
            return right

        return None

```

### 增删改查

``` python
# 1524. 在二叉搜索树中查找
# https://www.lintcode.com/problem/search-in-a-binary-search-tree/description
class Solution:
    def searchBST(self, root, val):
        if root == None or root.val == val:
            return root

        if val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

# 85. 在二叉查找树中插入节点
# https://www.lintcode.com/problem/insert-node-in-a-binary-search-tree/description
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: node: insert this node into the binary search tree
    @return: The root of the new binary search tree.
    """
    def insertNode(self, root, node):
        return self.__helper(root, node)
    
    def __helper(self, root, node):     
        # helper函数定义成私有属性   
        if root is None:
            return node

        if node.val < root.val:
            root.left = self.__helper(root.left, node)
        else:
            root.right = self.__helper(root.right, node)
        
        return root

# 87. 删除二叉查找树的节点
# https://www.lintcode.com/problem/remove-node-in-binary-search-tree/description
class Solution:
    def removeNode(self, root, value):
        if root is None:
            return None

        # check if node to delete is in left/right subtree
        if value < root.val:
            # not `self.removeNode(root.left, value)`
            root.left = self.removeNode(root.left, value)
        elif value > root.val:
            root.right = self.removeNode(root.right, value)
        else:
            # if root is has 2 childs/only one child/leaf node
            if root.left and root.right:
                max = self.find_left_Max(root)
                root.val = max.val
                root.left = self.removeNode(root.left, max.val)
            elif root.left:
                root = root.left
            elif root.right:
                root = root.right
            else:
                root = None

        return root

    # find max node in left subtree of root
    def find_left_Max(self, root):
        node = root.left
        while node.right:
            node = node.right
        return node

```

### inorder
  
``` python
# 67. 二叉树的中序遍历
# https://www.lintcode.com/problem/binary-tree-inorder-traversal/description
class Solution:
    def inorderTraversal(self, root):
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]
        inorder = []

        while stack:
            node = stack.pop()

            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            
            if stack:
                inorder.append(stack[-1].val)
        
        return inorder

# 86. 二叉查找树迭代器
# https://www.lintcode.com/problem/binary-search-tree-iterator/description
class BSTIterator:
    def __init__(self, root):
        dummy = TreeNode(0)
        dummy.right = root
        self.stack = [dummy]
        self.next()

    def hasNext(self, ):
        return len(self.stack) > 0

    def next(self, ):
        node = self.stack.pop()
        nxt_node = node

        if node.right:
            node = node.right
            while node:
                self.stack.append( node )
                node = node.left
        
        return nxt_node
```

### 经典题目

- [最近公共祖先 III](https://www.lintcode.com/problem/lowest-common-ancestor-iii/description)
- [Binary Tree Path Sum](https://www.lintcode.com/problem/binary-tree-path-sum/description)
- [二叉树的路径和 II](https://www.lintcode.com/problem/binary-tree-path-sum-ii/description): 路径和处理比较巧妙
- [二叉搜索树中最接近的值 II](https://www.lintcode.com/problem/closest-binary-search-tree-value-ii/description)

----

## 6 DFS

### 6.1 Combination-based DFS

碰到找所有方案的题目，基本可以确定是DFS
除了二叉树以外的90%dfs的题目，要么是排列，要么是排列

递归三要素

- 递归的定义
- 递归的拆解
- 递归的出口

模版

```python
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

![test](./.fig/backtrack.jpeg)

案例

```python
# 135/39. 数字组合
# https://www.lintcode.com/problem/combination-sum/description
class Solution:
    def combinationSum(self, candidates, target):
        if candidates is None:
            return []
        
        candidates.sort()
        result = []
        self.dfs(candidates, target, 0, [], result)

        return result

    def dfs(self, nums, target, start, combination, res):
        if target == 0:
            res.append(combination[:])
            return
        
        for i in range(start, len(nums)):
            if nums[i] > target:
                continue
            
            if i > 0 and nums[i] == nums[i-1]:
                continue

            combination.append(nums[i])
            self.dfs(nums, target-nums[i], i, combination, res)
            combination.pop()
```

例题

- [数字组合 II](https://www.lintcode.com/problem/combination-sum-ii/description)
- [k数和 II](https://www.lintcode.com/problem/k-sum-ii/description)
- [分割回文串](https://www.lintcode.com/problem/palindrome-partitioning/description)

### 6.2 Permutation DFS

知识点

- 如何使用DFS求解全排列问题
- 有重全排列问题如何去重
- 下一个排列怎么算

```python
# 10. 字符串的不同排列, 去重套路
# https://www.lintcode.com/problem/string-permutation-ii/description
class Solution:
    def stringPermutation2(self, s):
        chars = sorted(list(s))
        visited = [False] * len(chars)
        res = []
        self.dfs(chars, visited, [], res)

        return res
    
    def dfs(self, chars, visited, permutation, result):
        if len(chars) == len(permutation):
            result.append( "".join(permutation))
            return 
        
        for i in range(len(chars)):
            if visited[i]:
                continue

            # 去重：不同位置的同样的字符，必须按照顺序用。
            # 不能跳过一个a选下一个a. a' a" b; => a' a" b => √; => a" a' b => x
            if i > 0 and chars[i] == chars[i-1] and not visited[i-1]:
                continue

            visited[i] = True
            permutation.append(chars[i])

            self.dfs(chars, visited, permutation, result)
           
            permutation.pop()
            visited[i] = False
```

例题

- [电话号码的字母组合](https://www.lintcode.com/problem/letter-combinations-of-a-phone-number/description)
- ⭐ [字模式 II](https://www.lintcode.com/problem/word-pattern-ii/description)
- [单词接龙 II](https://www.lintcode.com/problem/word-ladder-ii/description): bfs + dfs
- ⭐ [单词搜索 II](https://www.lintcode.com/problem/word-search-ii/description): Trie Tree

----

## 8 Sweep Line

思路

- 事件往往是以区间的形式存在
- 区间两端代表事件的开始和结束
- 需要排序

```python
```

例题

- [数飞机](https://www.lintcode.com/problem/number-of-airplanes-in-the-sky/description)
- [大楼轮廓](https://www.lintcode.com/problem/the-skyline-problem/description)

----

## 9 Memoization searching

本质上: 动态规划， `从大到小`；
动态规划就是解决了重复计算的搜索, 将函数的结果保存下来，下次通过同样的参数访问时，可以直接返回保存下来的结果；

什么时候用记忆化搜索：

- 状态转移特别麻烦，不是顺序性
- 初始化状态不是很容易找到
- 从大到小

思路：

- 先思考最小状态
- 然后思考大的状态 -> 往小的递推，归纳总结

例题

- ⭐[通配符匹配](https://www.lintcode.com/problem/wildcard-matching/description)
- ⭐[正则表达式匹配](https://www.lintcode.com/problem/regular-expression-matching/description)
- ⭐⭐[Word Break III](https://www.lintcode.com/problem/word-break-iii/description)
- ⭐⭐[最长上升连续子序列 II](https://www.lintcode.com/problem/longest-continuous-increasing-subsequence-ii/description)
- [硬币排成线 II](https://www.lintcode.com/problem/coins-in-a-line-ii/description)

----

## 10 DP

动态滚动数组的四点要素：

- 状态
存储小规模的结果（最优解、Yes/No、Count）
- 方程
状态之间是怎么转换的，小的状态 -> 大的状态
- 初始化
最极限的小状态是什么来求最大值，起点
- 答案
最大的那个状态是什么，终点

```python
# Longest Increasing Subsequence
class Solution:
    def longestIncreasingSubsequence(self, nums):
        if nums is None or not nums: 
            return 0
    
        # state: dp[i] 表示以第 i 个数结尾的 LIS 的长度
        dp = [1] * len(nums)
        
        # dp[i] = max(dp[j] + 1), j < i && nums[j] < nums[i]
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
```

### 例题

- [编辑距离](https://www.lintcode.com/problem/edit-distance/description)
- [Longest Increasing Subsequence](https://www.lintcode.com/problem/longest-increasing-subsequence/description)

### 区间类DP

特点

1. 求一段区间的解max/min/count
2. 转移方程通过区间更新
3. 从大到小的更新

共性就是求[0, n-1]这样一个区间
逆向思维分析，从大到小
记忆化搜索的思路，从大到小，先考虑最后的0-n-1 合并的总花费

- [石子归并](https://www.lintcode.com/problem/stone-game/description)
- [吹气球](https://www.lintcode.com/problem/168/)
- []()

### 背包类DP

特点

1. 用值作为DP维度
2. Dp过程就是填写矩阵

   ```python
    if j < coins[i-1]:
        dp[i][j] = dp[i-1][j]
    else:
        dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]]
   ```

3. 可以滚动数组优化

例题

- [背包问题](https://www.lintcode.com/problem/92)
- [0-1 背包问题](https://www.lintcode.com/problem/backpack-ii/description)
- [划分和相等的子集](https://www.lintcode.com/problem/588/)
- [零钱兑换2](https://www.lintcode.com/problem/coin-change-2/description)
- [组合总和 IV](https://www.lintcode.com/problem/564/)
- [会议室4](https://www.lintcode.com/problem/300/description)
- [凑 N 分钱的方案数](https://www.lintcode.com/problem/279/description)

### 算法小抄

- [ ] 动态规划解题套路框架
- [ ] 动态规划设计：[最⻓递增⼦序列](https://www.lintcode.com/problem/longest-increasing-subsequence/description)
    两种方法：1. `DP` ; 2. `二分搜索法`
- [ ] 二维递增子序列：[俄罗斯套娃信封](https://www.lintcode.com/problem/russian-doll-envelopes/description)
- [ ] 动态规划设计：[最⼤⼦数组](https://www.lintcode.com/problem/maximum-subarray/description)
    难点：dp[i]的定义-> 以nums[i]结尾的最大子数组
- [ ] 最优子结构及其dp遍历反向
- [ ] 经典动态规划：[最⻓公共⼦序列](https://www.lintcode.com/problem/longest-common-subsequence/description)
- [x] 经典动态规划：[编辑距离](https://www.lintcode.com/problem/edit-distance/description)
- [ ] 子序列问题解题模板：[最长的回文序列](https://www.lintcode.com/problem/longest-palindromic-subsequence/description) 难点：`状态方程` 和 `遍历反向`
- [ ] 状态压缩：对动态规划进行降维打击
- [ ] [以最小插入次数构建回文串](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)
- [ ] 动态规划之[正则表达](https://www.lintcode.com/problem/regular-expression-matching/description)
- [ ] 动态规划之[四键键盘](https://www.lintcode.com/problem/4-keys-keyboard/description)
- [ ] 经典动态规划：⾼楼扔鸡蛋
- [ ] 经典动态规划：⾼楼扔鸡蛋（进阶）
- [x] 经典动态规划：[戳⽓球](https://www.lintcode.com/problem/burst-balloons/description)
- [x] 经典动态规划：[0-1 背包问题](https://www.lintcode.com/problem/backpack-ii/description)
- [x] 经典动态规划：[⼦集背包问题](https://www.lintcode.com/problem/partition-equal-subset-sum/description)
- [x] 经典动态规划：完全背包问题: [零钱兑换2](https://www.lintcode.com/problem/coin-change-2/description)
- [x] 团灭 LeetCode 打家劫舍问题： [392. 打劫房屋](https://www.lintcode.com/problem/house-robber/description); [534. 打劫房屋 II](https://www.lintcode.com/problem/house-robber-ii/description); [535. 打劫房屋 III](https://www.lintcode.com/problem/house-robber-iii/description)
- [ ] 动态规划和回溯算法到底谁是谁爹？[目标和](https://www.lintcode.com/problem/target-sum/description)

- [ ] 动态规划之`博弈问题`
- [ ] 动态规划之`KMP字符匹配算法`
- [ ] 贪⼼算法之`区间调度问题`
- [ ] 团灭 LeetCode `股票买卖问题`

----

## DEBUG

为什么要靠自己

- 如果是别人给你指出你的程序哪儿错了，你自己不会有任何收获，你下一次依旧会犯同样的错误。
- 经过长时间努力Debug 获得的错误，印象更深刻。
- Debug 能力是面试的考察范围。
- 锻炼Debug 能力能够提高自己的Bug Free的能力。

DEBUG步骤

- 重新读一遍程序
  按照自己当初想的思路，走一遍程序，看看程序是不是按照自己的思路在走。（因为很多时候，你写着写着就忘了很多事儿）这种方式是最有效最快速的 Debug 方式。
- 找到一个非常小非常小的可以让你的程序出错的数据。比如空数组，空串，1-5个数的数组，一个字符的字符串。
- 在程序的若干位置输出一些中间结果
  比如排序之后输出一下，看看是不是真的按照你所想的顺序排序的。这样可以定位到程序出错的部分。
- 定位了出错的部分之后，查看自己的程序该部分的逻辑是否有错。
- 在第4步中，如果无法通过肉眼看出错误的部分，就一步步“模拟执行”程序，找出错误。

---

## new

```python
```

- []()
