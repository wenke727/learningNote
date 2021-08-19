"""

* HashHeap
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

* 单调栈 Monotonous stack
def largestRectangleArea(self, heights):
    heights = [0, *heights, 0]
    stack, ans = [], 0

    for hi, val in enumerate(heights):
        while stack and val < heights[stack[-1]]:
            h = heights[stack.pop()]

            if not stack: continue
            lo = stack[-1] + 1
            area = (hi-lo) * h
            ans = max(ans, area)
        stack.append(hi)

    return ans


"""


import os, sys

"""Heap"""
# 363. 接雨水 ⭐⭐⭐
# [2021年3月9日 2021年8月12日]
# https://www.lintcode.com/problem/trapping-rain-water/description
# https://www.jiuzhang.com/solution/trapping-rain-water/#tag-lang-python
# version： _index_stack, 单调栈
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
# version: _value_stack
class Solution:
    def trapRainWater(self, heights):
        stack, trapped_water = [], 0
        
        for height in heights:
            if stack and height >= stack[0]:
                water_line = stack[0]
                while stack:
                    ground_height = stack.pop()
                    trapped_water += water_line - ground_height
            stack.append(height)
        
        water_line = 0
        while stack:
            ground_height = stack.pop()
            if ground_height < water_line:
                trapped_water += water_line - ground_height
            if ground_height > water_line:
                water_line = ground_height
                    
        return trapped_water
# version 相向型双指针算法
class Solution:
    def trapRainWater(self, heights):
        if not heights: return 0
            
        left, right = 0, len(heights) - 1
        left_max, right_max = heights[left], heights[right]
        water = 0
        while left <= right:
            if left_max < right_max:
                left_max = max(left_max, heights[left])
                water += left_max - heights[left]
                left += 1
            else:
                right_max = max(right_max, heights[right])
                water += right_max - heights[right]
                right -= 1
                    
        return water
# version 
class Solution:
    # DESC 从左到右扫描一边数组，获得每个位置往左这一段的最大值，再从右到左扫描一次获得每个位置向右的最大值。 然后最后再扫描一次数组
    def trapRainWater(self, heights):
        if not heights: return 0
            
        left_max = []
        curt_max = -sys.maxsize
        for height in heights:
            curt_max = max(curt_max, height)
            left_max.append(curt_max)
            
        right_max = []
        curt_max = -sys.maxsize
        for height in reversed(heights):
            curt_max = max(curt_max, height)
            right_max.append(curt_max)
            
        right_max = right_max[::-1]
            
        water = 0
        n = len(heights)
        for i in range(n):
            water += (min(left_max[i], right_max[i]) - heights[i])
        return water


# 364/407. 接雨水 II
# [2020年11月10日 2021年2月24日 2021年8月11日]
# https://www.lintcode.com/problem/trapping-rain-water-ii/description
# https://www.jiuzhang.com/solution/trapping-rain-water-ii/#tag-lang-python
from heapq import heappop, heappush
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    def trapRainWater(self, heights):
        if not heights or not heights[0]:
            return 0
        
        ans, max_height = 0, 0
        queue, visited = self.initialize(heights)

        while queue:
            cur_height, (x, y) = heappop(queue)
            
            if cur_height > max_height:
                max_height = cur_height
            else:
                ans += max_height - cur_height
            
            self.bfs(heights, x, y, visited, queue)

        return ans
    

    def bfs(self, heights, x, y, visited, queue):
        for dx, dy in DIRECTIONS:
            nxt_x, nxt_y = x+dx, y+dy

            if not ( 0<=nxt_x<self.n and 0<=nxt_y<self.m ):
                continue
            if (nxt_x, nxt_y) in visited:
                continue

            heappush(queue, (heights[nxt_x][nxt_y], (nxt_x, nxt_y)))
            visited.add((nxt_x, nxt_y))
        
        return
    
    
    def initialize(self, heights):
        queue, visited = [], set()
        self.n, self.m = len(heights), len(heights[0])

        for i in [0, self.n-1]:
            for j in range(self.m):
                heappush(queue, (heights[i][j], (i, j)))
                visited.add((i,j))
        
        for j in [0, self.m-1]:
            for i in range(1, self.n-1):
                heappush(queue, (heights[i][j], (i, j)))
                visited.add((i,j))
        
        return queue, visited


# 81. 数据流中位数
# [2020年11月11日 2021年2月24日 2021年8月11日]
# https://www.lintcode.com/problem/find-median-from-data-stream/description, https://leetcode-cn.com/problems/find-median-from-data-stream/
from heapq import heappop, heappush
class Solution:
    """
    @param nums: A list of integers
    @return: the median of numbers
    """
    def __init__(self):
        # 小顶堆放大于中位数的数
        # 大顶堆放小于中位数的数
        # 最开始中位数是数据流中的第一个数
        self.max_heap = []
        self.min_heap = []
        self.is_first_add = True

    def add(self, num):
        if self.is_first_add:
            # 第一个进入数据流的数字是第一个中位数
            self.median = num
            self.is_first_add = False
            return
    
        if num < self.median:
            heappush(self.min_heap, -num)
        else:
            heappush(self.max_heap, num)

        if len(self.min_heap) + 1 < len(self.max_heap):
            heappush( self.min_heap, -self.median )
            self.median = heappop(self.max_heap)
        elif len(self.min_heap) > len(self.max_heap):
            heappush(self.max_heap, self.median)
            self.median = -heappop(self.min_heap)

    def getMedian(self):
        return self.median


# 360. 滑动窗口的中位数 ⭐⭐⭐
# [2021年3月19日 2021年8月12日]
# https://www.lintcode.com/problem/sliding-window-median/description 
# https://www.jiuzhang.com/solutions/sliding-window-median/#tag-lang-python
# Version HashHeap
class HashHeap:
    def __init__(self, desc=False):
        self.hash = {}
        self.heap = []
        self.desc = desc


    @property
    def size(self):
        return len(self.heap)


    def push(self, item):
        self.heap.append(item)
        self.hash[item] = self.size - 1
        self._shift_up(self.size - 1)


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
        self._swap(index, self.size-1)

        del self.hash[item]
        self.heap.pop()

        if index < self.size:
            self._shift_up(index)
            self._shift_down(index)


    def _shift_up(self, index):
        while index != 0:
            parent = index // 2
            if self._smaller(self.heap[parent], self.heap[index]):
                break
            self._swap(parent, index)
            index = parent


    def _shift_down(self, index):
        if index is None:
            return

        while index * 2 < self.size:
            smallest = index
            left, right = 2*index, 2*index + 1

            if self._smaller(self.heap[left], self.heap[index]):
                smallest = left
            if right < self.size and self._smaller(self.heap[right], self.heap[smallest]):
                smallest = right
            if smallest == index:
                break
            
            self._swap(index, smallest)
            index = smallest


    def _swap(self, i, j):
        elem1, elem2 = self.heap[i], self.heap[j]
        self.heap[i], self.heap[j] = elem2, elem1
        self.hash[elem1], self.hash[elem2] = j, i 


    def _smaller(self, left, right):
        return  right < left if self.desc else left < right
    
class Solution:
    def medianSlidingWindow(self, nums, k):
        if not nums or len(nums) < k:
            return []

        self.left, self.right = HashHeap(desc=True), HashHeap()

        for i in range(k-1):
            self.add((nums[i], i))

        res = []
        for i in range(k-1, len(nums)):
            self.add((nums[i], i))
            res.append(self.median)
            self.remove((nums[i-k+1], i-k+1))

        return res


    def add(self, item):
        if self.left.size > self.right.size:
            self.right.push(item)
        else:
            self.left.push(item)
        
        if self.left.size == 0 or self.right.size == 0:
            return
        
        if self.left.top() > self.right.top():
            self.left.push(self.right.pop())
            self.right.push(self.left.pop())


    def remove(self, item):
        self.left.remove(item)
        self.right.remove(item)
        if self.left.size < self.right.size:
            self.left.push(self.right.pop())


    @property
    def median(self):
        return self.left.top()[0]

#version Heap
from heapq import *
class Heap:
    # q1存储了当前所有元素（包括未删除元素）
    # q2存储了q1中已删除的元素
    def __init__(self):
        self.__q1 = []
        self.__q2 = []

    # push 操作向 q1 中 push 一个新的元素
    def push(self, x):
        heappush(self.__q1, x)

    # q2 中 push 一个元素表示在 q1 中它已经被删除了
    def remove(self, x):
        heappush(self.__q2, x)

    # 这里就要用到 q2 里面的元素了，如果堆顶的元素已经被 remove 过，那么它此时应该在 q2 中的堆顶
    # 此时我们把两个堆一起 pop 就好了，直到堆顶元素不同或者 q2 没元素了
    def pop(self):
        while len(self.__q2) != 0 and self.__q1[0] == self.__q2[0]:
            heappop(self.__q1)
            heappop(self.__q2)
        if len(self.__q1) != 0:
            heappop(self.__q1)

    # 这里就是先进行和 pop 中类似的操作，删除已经 remove 的元素，然后取出堆顶
    def top(self):
        while len(self.__q2) != 0 and self.__q1[0] == self.__q2[0]:
            heappop(self.__q1)
            heappop(self.__q2)
        if len(self.__q1) != 0:
            return self.__q1[0]

    # 这个就是返回堆大小的，可以知道堆当前真实大小就是 q1 大小减去 q2 大小
    def size(self):
        return len(self.__q1) - len(self.__q2)

    def sol(self):
        print(self.__q1)
        # print(self.q2)

class Solution:
    def medianSlidingWindow(self, nums, k):
        # write your code here
        qmx = Heap()
        qmn = Heap()
        ans = []
        for i in range(len(nums)):
            x = nums[i]
            # 堆都为空，直接压入大根堆
            if i == 0:
                qmx.push(-x)
            else:
                # 根据当前值和大根堆堆顶的值判断，该压入哪个堆里
                if x <= -qmx.top():
                    qmx.push(-x)
                else:
                    qmn.push(x)
            # 控制滑动窗口，删除离开滑动窗口的元素
            if i >= k:
                # 根据当前要删除的值和大根堆堆顶的值判断，该从哪个堆里删除
                val = nums[i - k]
                if val <= -qmx.top():
                    qmx.remove(-val)
                else:
                    qmn.remove(val)
            if i >= k - 1:
                # 大根堆的堆顶是中位数所以要保证，大根堆的元素数量是k / 2，向上取整
                mxnum = (k + 1) // 2
                while qmx.size() != mxnum:
                    if qmx.size() > mxnum:
                        x = -qmx.top()
                        qmx.pop()
                        qmn.push(x)
                    else:
                        x = -qmn.top()
                        qmn.pop()
                        qmx.push(x)
                ans.append(-qmx.top())
        return ans


""""Stack"""
# 12/155. 带最小值操作的栈
# [2020年11月11日 2021年3月3日 2021年8月12日]
# https://www.lintcode.com/problem/min-stack/description
# DESC 使用一个 minStack 作为辅助的栈，用来更新目前的最小值序列。 如果发现了一个更小的值就往 minStack 里也 push 
# DESC 注意如果和最小值相等的情况，也需要往 minStack 里 push, 如果 push 的数比当前的最小值要大，就不需要往 MinStack 里push 了。
class MinStack:
    """
    push(val) 将 val 压入栈
    pop() 将栈顶元素弹出, 并返回这个弹出的元素
    min() 返回栈中元素的最小值
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []
 
    def push(self, number):
        self.stack.append(number)
        if not self.min_stack or self.min_stack[-1] >= number:
            self.min_stack.append(number)
 
    def pop(self):
        number = self.stack.pop()
        if number == self.min_stack[-1]: 
            self.min_stack.pop()
        return number
 
    def min(self):
        return self.min_stack[-1]
# version Leetcode
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
 
    def push(self, number):
        self.stack.append(number)
        if not self.min_stack or self.min_stack[-1] >= number:
            self.min_stack.append(number)
 
    def pop(self):
        number = self.stack.pop()
        if number == self.min_stack[-1]: 
            self.min_stack.pop()
        return number
 
    def getMin(self):
        return self.min_stack[-1]

    def top(self):
        return self.stack[-1]


# 40/232. 用栈实现队列 
# [2020年11月11日 2021年3月5日 2021年8月12日]
# https://www.lintcode.com/problem/implement-queue-by-two-stacks/description
# https://www.jiuzhang.com/solution/implement-queue-by-two-stacks/#tag-lang-python
class MyQueue:
    def __init__(self):
        self.in_, self.out_ = [], []

    def push(self, x: int):
        self.in_.append(x)

    def pop(self):
        self.top()
        return self.out_.pop()

    def top(self):
        if not self.out_:
            while self.in_:
                self.out_.append(self.in_.pop())
                
        return self.out_[-1]

    def empty(self):
        return not (self.in_ or self.out_)


# 494. 双队列实现栈
# [2021年3月5日 2021年8月12日]
# https://www.lintcode.com/problem/494/
from collections import deque
class Stack:
    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()
        
    def push(self, x):
        self.queue1.append(x)

    def pop(self):
        for _ in range(len(self.queue1) - 1):
            val = self.queue1.popleft()
            self.queue2.append(val)
            
        val = self.queue1.popleft()
        self.queue1, self.queue2 = self.queue2, self.queue1
        return val

    def top(self):
        val = self.pop()
        self.push(val)
        return val

    def isEmpty(self):
        return not self.queue1


# 575/394. 字符串解码 
# [2020年11月11日 2021年3月5日 2021年8月12日]
# https://www.lintcode.com/problem/decode-string/description
# https://www.jiuzhang.com/solutions/expression-expand/#tag-lang-python
# version: stack
class Solution:
    # DESC 把所有字符一个个放到 stack 里，当碰到 "]" 的时候，就从 stack 把对应的字符串和重复次数找到，展开，然后再丢回 stack 里
    def expressionExpand(self, s):
        stack = []
        for c in s:
            if c != ']':
                stack.append(c)
                continue
            
            strs = []
            while stack and stack[-1] != '[':
                strs.append(stack.pop())
            stack.pop() # skip '['
            
            repeats, base = 0, 1
            while stack and stack[-1].isdigit():
                repeats += int(stack.pop()) * base
                base *= 10
            stack.append(''.join(reversed(strs)) * repeats)
        
        return ''.join(stack)
# version: dfs
class Solution:
    def expressionExpand(self, s):
        if not s:
            return ''
        self.end = 0

        return self.dfs(s, 0, '')
        
    # Expand from idx to rest. Result expanded before idx is saved in result
    def dfs(self, s, idx, result):
        while idx < len(s):
            if s[idx] == ']':
                self.end = idx
                return result
            
            elif s[idx].isdigit():
                num = ''
                while s[idx] != '[':
                    num += s[idx]
                    idx += 1
                result += int(num) * self.dfs(s, idx+1, '')
            else:
                result += s[idx]
            idx = 1 + max(idx, self.end)
        return result 


"""
单调栈 Monotonous stack
    * ref: https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/shou-ba-shou-she-ji-shu-ju-jie-gou/dan-tiao-zhan#dan-tiao-zhan-mo-ban
    找每个元素左边或者右边
    第一个比它自身小/大的元素
    用单调栈来维护
"""
# 122/84. 直方图最大矩形覆盖 
# [2021年3月5日 2021年3月9日 2021年8月13日]
# https://www.lintcode.com/problem/largest-rectangle-in-histogram/description
# https://www.jiuzhang.com/solutions/largest-rectangle-in-histogram#tag-lang-python
# DESC 用九章算法强化班中讲过的单调栈(stack)。维护一个单调递增栈，逐个将元素 push 到栈里。
# DESC push 进去之前先把 >= 自己的元素 pop 出来。 每次从栈中 pop 出一个数的时候，就找到了
# DESC 往左数比它小的第一个数（当前栈顶）和往右数比它小的第一个数（即将入栈的数）， 从而可
# DESC 以计算出这两个数中间的部分宽度 * 被pop出的数，就是以这个被pop出来的数为最低的那个直
# DESC 方向两边展开的最大矩阵面积。 因为要计算两个数中间的宽度，因此放在 stack 里的是每个数的下标。
# version classical
class Solution:
    def largestRectangleArea(self, heights):
        heights = [0, *heights, 0]
        stack, ans = [], 0

        for hi, val in enumerate(heights):
            while stack and val < heights[stack[-1]]:
                h = heights[stack.pop()]

                if not stack:
                    continue

                lo = stack[-1] + 1
                ans = max(ans, (hi-lo) * h)

            stack.append(hi)

        return ans
# version 2
class Solution:
    def largestRectangleArea(self, height):
        n = len(height)
        # stack[]表示单调栈，cnt表示栈内元素个数
        stack = [0] * n
        cnt = -1
        # left[]和right[]分别表示最左/右边界
        left = [0] * n 
        right = [0] * n
        # 获取元素i作为最小元素的最左边界left[i]
        for i in range(n):
            # 栈内有元素且栈顶元素大于等于当前元素
            while cnt > -1 and height[stack[cnt]] >= height[i]:
                cnt -= 1
            if cnt == -1:
                left[i] = 0
            else:
                left[i] = stack[cnt] + 1
            # 压入当前元素
            cnt += 1
            stack[cnt] = i

        # 获取元素i作为最小元素的最右边界right[i],更新答案
        cnt, ans = -1, 0
        for i in range(n - 1, -1, -1):
            # 栈内有元素且栈顶元素大于等于当前元素
            while cnt > -1 and height[stack[cnt]] >= height[i]:
                cnt -= 1
            if cnt == -1:
                right[i] = n - 1
            else:
                right[i] = stack[cnt] - 1
            cnt += 1
            stack[cnt] = i
            # 更新最大答案
            ans = max(ans, height[i] * (right[i] - left[i] + 1))
        return ans


# 510/85. 最大矩形
# [2020年11月12日 2021年3月9日 2021年8月13日]
# https://www.lintcode.com/problem/maximal-rectangle/description
# https://www.jiuzhang.com/solution/maximal-rectangle/#tag-lang-python
class Solution:
    def maximalRectangle(self, matrix):
        if not matrix: 
            return 0
            
        max_rectangle = 0
        heights = [0] * len(matrix[0])
        for row in matrix:
            for index, num in enumerate(row):
                heights[index] = heights[index] + 1 if num else 0
            
            max_rectangle = max(
                max_rectangle,
                self.find_max_rectangle(heights),
            )

        return max_rectangle

    def find_max_rectangle(self, heights):
        heights = [0, *heights, 0]
        stack, ans = [], 0

        for hi, val in enumerate(heights):
            while stack and val < heights[stack[-1]]:
                h = heights[stack.pop()]

                if not stack:
                    continue

                lo = stack[-1] + 1
                ans = max(ans, (hi-lo) * h)

            stack.append(hi)

        return ans


# 126/654. 最大树 # TODO ⭐⭐⭐ # TODO
# [2021年8月14日]
# https://www.lintcode.com/problem/max-tree/description
# https://www.jiuzhang.com/solution/max-tree/#tag-lang-python
# DESC 保存一个单调递减栈。每个数从栈中被 pop 出的时候，就知道它往左和往右的第一个比他大的数的位置了。
class Solution:
    def maxTree(self, A):
        if not A: 
            return None

        stack, nodes =[], [TreeNode(num) for num in (A + [sys.maxsize])]

        for hi, val in enumerate(A + [sys.maxsize]):
            while stack and val > A[stack[-1]]:
                top_node = nodes[stack.pop()]
                lo = stack[-1]
                lo_val = A[stack[-1]] if stack else sys.maxsize
                
                if val > lo_val:
                    nodes[lo].right = top_node
                else:
                    nodes[hi].left = top_node
            stack.append(hi)

        # sys.maxsize 's left child is the maximum number
        return nodes[-1].left


class Solution:
    """
    @param A: Given an integer array with no duplicates.
    @return: The root of max tree.
    """
    def maxTree(self, A):
        stack = []
        for num in A:
            node = TreeNode(num)		#新建节点
            while stack and num > stack[-1].val:		#如果stk中的最后一个节点比新节点小
                node.left = stack.pop()					#当前新节点的左子树为stk的最后一个节点
                
            if stack:									#如果stk不为空
                stack[-1].right = node					#将新节点设为stk最后一个节点的右子树
                
            stack.append(node)

        return stack[0]

# version 分治
class Solution:
    """
    @param A: Given an integer array with no duplicates.
    @return: The root of max tree.
    """
    def maxTree(self, A):
        n = len(A)

        pos = 0 # 当前区间最大值的下标
        for i in range(n):
            if A[i] > A[pos]:
                pos = i

        root = TreeNode(A[pos]);
        if pos > 0:
            root.left = self.dfs(0, pos - 1, A)
        if pos < n - 1:
            root.right = self.dfs(pos + 1, n - 1, A)
        return root

    def dfs(self, left, right, A):
        pos = left
        for i in range(left, right + 1):
            if A[i] > A[pos]:
                pos = i

        son = TreeNode(A[pos])

        if left < pos:
            son.left = self.dfs(left, pos - 1, A)
        if right > pos:
            son.right = self.dfs(pos + 1, right, A)
        return son
    

# 下一个更大的数 II
# [2021年8月14日]
# https://www.lintcode.com/problem/1201/
class Solution:
    def nextGreaterElements(self, nums):
        if not nums: 
           return nums
        
        n = len(nums)
        res = [-1] * n
        stack = []
        
        for right in range(2*n):
            right %= n
            while stack and nums[right] > nums[stack[-1]]:
                cur = stack.pop()
                res[cur] = nums[right]

            stack.append(right)
        
        return res


# 229 · 栈排序
# [2021年8月14日]
# https://www.lintcode.com/problem/229/
class Solution:
    """
    @param: stk: an integer stack
    @return: void
    """
    def stackSorting(self, stk):
        tmp = []

        while stk:
            top = stk.pop()
            while tmp and top > tmp[-1]:
                stk.append(tmp.pop())

            tmp.append(top)
        
        while tmp:
            stk.append(tmp.pop())
        
        return stk
