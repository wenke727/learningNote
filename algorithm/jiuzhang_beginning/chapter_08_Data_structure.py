"""
* Queue
"""
# 642. 数据流滑动窗口平均值
# [2020年11月7日 2021年8月8日]
# https://www.lintcode.com/problem/moving-average-from-data-stream/description
# https://www.jiuzhang.com/solutions/moving-average-from-data-stream/#tag-lang-python
from collections import deque
class MovingAverage:
    def __init__(self, size):
        self.queue = deque([])
        self.size = size
        self.sum = 0
    
    def next(self, val):
        if len(self.queue) == self.size:
            self.sum -= self.queue.popleft()
        
        self.sum += val
        self.queue.append(val)

        return self.sum / len( self.queue )


"""
hash
"""
# 134. LRU缓存策略 ⭐
# [2020年11月7日 2021年8月8日]
# https://www.lintcode.com/problem/lru-cache/description
# https://www.jiuzhang.com/solutions/lru-cache/#tag-lang-python
class LinkedNode():
    def __init__(self, key=None, value=None, pre=None, next=None):
        self.key = key
        self.value = value
        self.pre = pre
        self.next = next

class LRUCache:
    def __init__(self, capacity):
        # 我们实现的双链表API是从尾部插入，也就是说靠尾部的数据是最近使用的
        self.key_to_node = {}
        self.dummy = LinkedNode()
        self.tail = self.dummy
        self.capacity = capacity

    def get(self, key):
        if key not in self.key_to_node:
            return -1
        
        cur = self.key_to_node[key]
        self._kick(cur)

        return cur.value

    def set(self, key, value):
        if key in self.key_to_node:
            self._kick(self.key_to_node[key])
            self.key_to_node[key].value = value

            return
        
        self._push_back(LinkedNode(key, value))
        if len(self.key_to_node) > self.capacity:
            self._pop_front()

    def _kick(self, cur):
        if cur == self.tail:
            return
        
        cur.pre.next = cur.next
        cur.next.pre = cur.pre
        self._push_back(cur)

    def _push_back(self, node):
        node.pre = self.tail
        self.tail.next = node
        self.tail = node
        node.next = None

        if node.key not in self.key_to_node:
            self.key_to_node[node.key] = node

    def _pop_front(self, ):
        head = self.dummy.next
        del self.key_to_node[head.key]

        self.dummy.next = head.next
        head.next.pre = self.dummy


# 657. O(1)实现数组插入/删除/随机访问
# [2020年11月7日 2021年8月8日]
# https://www.lintcode.com/problem/insert-delete-getrandom-o1/description
# https://www.jiuzhang.com/solutions/insert-delete-getrandom-o1/#tag-lang-python
import random
class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.val2index = {}

    def insert(self, val):
        if val in self.val2index: 
            return False
        
        self.nums.append( val )
        self.val2index[val] = len(self.nums) - 1
        
        return True

    def remove(self, val):
        if val not in self.val2index: 
            return False
        
        index = self.val2index[val]
        last = self.nums[-1]

        # move the last elemnet to index
        self.nums[index] = last
        self.val2index[last] = index

        # remove the last element 
        self.nums.pop()
        del self.val2index[val]

        return True

    def getRandom(self):
        return self.nums[ random.randint( 0, len(self.nums) - 1 ) ]


# 954. Insert Delete GetRandom O(1) - 允许重复
# [2020年11月7日 2021年8月8日]
# https://www.lintcode.com/problem/insert-delete-getrandom-o1-duplicates-allowed/description
# https://www.jiuzhang.com/solutions/insert-delete-getrandom-o1-duplicates-allowed/#tag-lang-python
class RandomizedCollection(object):
    def __init__(self):
        self.map = {}
        self.nums = []

    def insert(self, val):
        self.nums.append(val)

        if val in self.map:
            self.map[val].append(len(self.nums) - 1)
            return False
        else:
            self.map[val] = [len(self.nums) - 1]
            return True
        
    def remove(self, val):
        if val not in self.map:
            return False

        pos = self.map[val].pop()

        if not self.map[val]:
            del self.map[val]

        if pos != len(self.nums) - 1:
            self.map[self.nums[-1]][-1] = pos
            self.nums[pos] = self.nums[-1]
        self.nums.pop()
        
        return True

    def getRandom(self):
        import random
        return random.choice(self.nums)


# 685. 数据流中第一个唯一的数字
# [2020年11月7日]
# https://www.lintcode.com/problem/first-unique-number-in-data-stream/description
# https://www.jiuzhang.com/solutions/first-unique-number-in-data-stream/#tag-lang-python
class Solution:
    def firstUniqueNumber(self, nums, number):
        counter = {}

        for num in nums:
            counter[num] = counter.get(num, 0) + 1
            if num == number:
                break
        # 在循环正常完成时执行，这意味着循环没有遇到任何 break 语句
        else:
            return -1

        for num in counter:
            if counter[num] == 1:
                return num
            if num == number:
                break
            
        return -1


# 960. 数据流中第一个独特的数 II # TODO
# https://www.lintcode.com/problem/first-unique-number-in-data-stream-ii/description
# https://www.jiuzhang.com/solutions/first-unique-number-in-data-stream-ii/#tag-lang-python
class DataStream:
    def __init__(self):
        self.dummy = ListNode(0)
        self.tail = self.dummy
        self.num_to_prev = {}
        self.duplicates = set()
          
    def add(self, num):
        if num in self.duplicates:
            return
        
        if num not in self.num_to_prev:
            self.push_back(num)            
            return
        
        # find duplicate, remove it from hash & linked list
        self.duplicates.add(num)
        self.remove(num)
    
    def remove(self, num):
        prev = self.num_to_prev.get(num)
        del self.num_to_prev[num]
        prev.next = prev.next.next
        
        if prev.next:
            self.num_to_prev[prev.next.val] = prev
        else:
            # if we removed the tail node, prev will be the new tail
            self.tail = prev

    def push_back(self, num):
        # new num add to the tail
        self.tail.next = ListNode(num)
        self.num_to_prev[num] = self.tail
        self.tail = self.tail.next

    def firstUnique(self):
        if not self.dummy.next:
            return None
        return self.dummy.next.val


# 138. 子数组之和
# [2020年11月7日]
# https://www.lintcode.com/problem/subarray-sum/description
# https://www.jiuzhang.com/solutions/subarray-sum/#tag-lang-python
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        prefix_sum = 0
        prefix_hash = {0: -1}

        for index, num in enumerate(nums):
            prefix_sum += num

            if prefix_sum in prefix_hash:
                return prefix_hash[prefix_sum] + 1, index
            prefix_hash[prefix_sum] = index

        return -1, -1


# 105. 复制带随机指针的链表
# https://www.lintcode.com/problem/copy-list-with-random-pointer/description
# https://www.jiuzhang.com/solutions/copy-list-with-random-pointer/#tag-lang-python


# 171. 乱序字符串
# [2020年11月7日]
# https://www.lintcode.com/problem/anagrams/description
# https://www.jiuzhang.com/solutions/anagrams/#tag-lang-python
class Solution:
    """
    @param strs: A list of strings
    @return: A list of strings
    """
    def anagrams(self, strs):
        strs_hash ={}
        for word in strs:
            word_sorted = ''.join( sorted(word) )
            if word_sorted not in strs_hash:
                strs_hash[word_sorted] = [word]
            else:
                strs_hash[word_sorted] += [word]
            
        res = []
        for item in strs_hash:
            if len(strs_hash[item]) > 1:
                res += strs_hash[item]
        
        return res


# 124. 最长连续序列
# [2020年11月7日]
# https://www.lintcode.com/problem/longest-consecutive-sequence/description
# https://www.jiuzhang.com/solutions/longest-consecutive-sequence/#tag-lang-python
class Solution:
    def longestConsecutive(self, nums):
        max_len = 1
        table = {num: True for num in nums}

        for low in nums:
            if low - 1 not in table:
                high = low + 1
                while high in table:
                    high += 1

                # ! not need to + 1
                max_len = max(max_len, high - low)

        return max_len


"""heap"""
# 130. 堆化
# [2021年2月25日]
# https://www.lintcode.com/problem/130/
# https://www.jiuzhang.com/solution/heapify/
# Ref https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/shou-ba-shou-she-ji-shu-ju-jie-gou/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
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
            

# 4. 丑数 II
# [2020年11月7日 2021年2月25日]
# https://www.lintcode.com/problem/ugly-number-ii/description
# https://www.jiuzhang.com/solutions/ugly-number-ii/#tag-lang-python
import heapq
class Solution:
    """
    @param {int} n an integer.
    @return {int} the nth prime number as description.
    """
    def nthUglyNumber(self, n):
        heap = [1]
        visited = set([1])
        
        val = None
        for i in range(n):
            val = heapq.heappop(heap)
            for factor in [2, 3, 5]:
                if val * factor not in visited:
                    visited.add(val * factor)
                    heapq.heappush(heap, val * factor)
            
        return val


# 612. K个最近的点
# [2020年11月7日  2021年2月25日]
# https://www.lintcode.com/problem/k-closest-points/description
# https://www.jiuzhang.com/solutions/k-closest-points/#tag-lang-python
import heapq
class Solution:
    def kClosest(self, points, origin, k):
        heap, res = [], []
        for point in points:
            dist = self.getDistance( point, origin )
            heapq.heappush( heap, (-dist, -point.x, -point.y) )

            if len( heap ) > k:
                heapq.heappop(heap)
        
        while heap:
            _, x, y = heapq.heappop(heap)
            res.append( Point(-x, -y) )
        
        return res[::-1]

    def getDistance(self, a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


# 545. 前K大数 II
# [2020年11月7日 2021年2月25日]
# https://www.lintcode.com/problem/top-k-largest-numbers-ii/description
# https://www.jiuzhang.com/solutions/top-k-largest-number-ii/#tag-lang-python
import heapq
class Solution:
    def __init__(self, k):
        self.k = k
        self.heap = []

    def add(self, num):
        heapq.heappush( self.heap, num )
        while len(self.heap) > self.k:
            heapq.heappop( self.heap )

    def topk(self):
        return sorted( self.heap, reverse = True )


# 104. 合并k个排序链表
# [2020年11月7日]
# https://www.lintcode.com/problem/merge-k-sorted-lists/description
# https://www.jiuzhang.com/solution/merge-k-sorted-lists/#tag-lang-python
# version: 使用 PriorityQueue
import heapq
ListNode.__lt__ = lambda x, y : (x.val < y.val)
class Solution:
    def mergeKLists(self, lists):
        if not lists: 
            return None
        
        dummy = ListNode( 0 )
        tail = dummy
        heap = []

        for head in lists:
            if head:
                heapq.heappush( heap, head )
        
        while heap:
            head = heapq.heappop( heap )
            tail.next = head
            tail = head
            if head.next:
                heapq.heappush(heap, head.next)
            
        return dummy.next
# version: 类似归并排序的分治算法
class Solution:
    def mergeKLists(self, lists):
        if not lists: return None
        return self.merge_range_lists(lists, 0, len(lists) - 1)
    
    def merge_range_lists(self, lists, start, end):
        if start == end: return lists[start]
        
        mid = (start + end) // 2
        left = self.merge_range_lists( lists, start, mid )
        right = self.merge_range_lists( lists, mid + 1, end ) # ! `+1`

        return self.merge_two_lists( left, right )
    
    def merge_two_lists(self, p0, p1):
        tail = dummy = ListNode(0)
        while p0 and p1:
            if p0.val < p1.val:
                tail.next = p0
                p0 = p0.next
            else:
                tail.next = p1
                p1 = p1.next
            tail = tail.next
        
        if p0: tail.next = p0
        if p1: tail.next = p1
        
        return dummy.next
# version: 自底向上的两两归并算法
class Solution:
    def mergeKLists(self, lists):
        if not lists: return None

        while len(lists) > 1:
            next_lists = []
            for i in range(0, len(lists), 2):
                if i + 1 < len(lists):
                    new_list = self.merge_two_lists(lists[i], lists[i + 1])
                else:
                    new_list = lists[i]
                next_lists.append(new_list)
                
            lists = next_lists
            
        return lists[0]

    def merge_two_lists(self, p0, p1):
        tail = dummy = ListNode(0)
        while p0 and p1:
            if p0.val < p1.val:
                tail.next = p0
                p0 = p0.next
            else:
                tail.next = p1
                p1 = p1.next
            tail = tail.next
        
        if p0: tail.next = p0
        if p1: tail.next = p1
        
        return dummy.next


# 613. 优秀成绩
# [2020年11月7日]
# https://www.lintcode.com/problem/high-five/description
# https://www.jiuzhang.com/solutions/high-five/#tag-lang-python
import heapq
class Solution:
    def highFive(self, records):
        if not records: return {}

        data = {}
        for student in records:
            if student.id not in data:
                data[student.id] = [student.score]
            else:
                heapq.heappush(data[student.id], student.score)
        
                if len(data[student.id]) > 5:
                    heapq.heappop(data[student.id])
        
        for case in data:
            data[case]=sum(data[case])/5.0
        
        return data


# 486. 合并k个排序数组
# [2021年2月26日]
# https://www.lintcode.com/problem/merge-k-sorted-arrays/description
# https://www.jiuzhang.com/solutions/merge-k-sorted-arrays/#tag-lang-python
import heapq
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        res, heap = [], []
        for index, array in enumerate(arrays):
            if len(array) == 0:
                continue
            heapq.heappush(heap, (array[0], index, 0))
        
        while heap:
            val, arr_index, i = heapq.heappop(heap)
            res.append(val)
            
            if i + 1 < len(arrays[arr_index]):
                heapq.heappush(heap, (arrays[arr_index][i+1], arr_index, i+1))
        
        return res


# 544. 前K大数
# [2020年11月7日 2021年2月26日]
# https://www.lintcode.com/problem/top-k-largest-numbers/description
# https://www.jiuzhang.com/solutions/top-k-largest-numbers/#tag-lang-python
# version： heapq
class Solution:
    """
    @param nums: an integer array
    @param k: An integer
    @return: the top k largest numbers in array
    """
    def topk(self, nums, k):
        import heapq
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        heap.sort(reverse = True)
        return heap
# version: quick select
class Solution:
    def topk(self, nums, k):
        self.quick_select(nums, 0, len(nums) - 1, k)
        res = nums[:k]
        res.sort(reverse = True)
        return res
    
    def quick_select(self, nums, start, end, k):
        if start == end: return 
        
        left, right = start, end
        pivot = nums[(start + end) // 2]
        
        while left <= right:
            while left <= right and nums[left] > pivot:
                left += 1
            while left <= right and nums[right] < pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        if start + k - 1 <= right:
            self.quick_select(nums, start, right, k)
            return
        if start + k - 1 >= left:
            self.quick_select(nums, left, end, k - (left - start))
            return
        return


# 401. 排序矩阵中的从小到大第k个数
# [2021年2月26日]
# https://www.lintcode.com/problem/kth-smallest-number-in-sorted-matrix/description
# https://www.jiuzhang.com/solutions/kth-smallest-number-in-sorted-matrix/#tag-lang-python

import heapq
class Solution:
    """
    # 在一个排序矩阵中找从小到大的第 k 个整数。
    # 排序矩阵的定义为：每一行递增，每一列也递增。
    @param matrix: a matrix of integers
    @param k: An integer
    @return: the kth smallest number in the matrix
    """
    def kthSmallest(self, matrix, k):
        if not matrix or not matrix[0]: return None
        n, m = len(matrix), len(matrix[0])
        
        lst = [ (matrix[0][0], 0, 0) ]
        visited = set([0])
        res = None

        for _ in range(k):
            val, x, y = heapq.heappop(lst)
            if x+1 < n and (x+1)*m + y not in visited:
                heapq.heappush(lst, (matrix[x+1][y], x+1, y))
                visited.add((x+1)*m + y)

            if y+1 < m and x*m + y+1 not in visited:
                heapq.heappush(lst, (matrix[x][y+1], x, y+1))
                visited.add(x*m +y+1)
            res = val
        return res                 