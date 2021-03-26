# 6. 合并排序数组 II
# []
# https://www.lintcode.com/problem/merge-two-sorted-arrays/description
# https://www.jiuzhang.com/solutions/merge-two-sorted-arrays#tag-lang-python
class Solution:
    """
    @param A: sorted integer array A
    @param B: sorted integer array B
    @return: A new sorted integer array
    """
    def mergeSortedArray(self, A, B):
        i, j = 0, 0
        C = []
        while i < len(A) and j < len(B):
            if A[i] < B[j]:
                C.append(A[i])
                i += 1
            else:
                C.append(B[j])
                j += 1
        
        while i < len(A):
            C.append(A[i])
            i += 1
        while j < len(B):
            C.append(B[j])
            j += 1
            
        return C


# 64. 合并排序数组
# []
# https://www.lintcode.com/problem/merge-sorted-array/description
# https://www.jiuzhang.com/solutions/merge-sorted-array/#tag-lang-python
class Solution:
    """
    @param: A: sorted integer array A which has m elements, but size of A is m+n
    @param: m: An integer
    @param: B: sorted integer array B which has n elements
    @param: n: An integer
    @return: nothing
    """
    def mergeSortedArray(self, A, m, B, n):
        # write your code here
        pos = m + n - 1 
        i = m - 1  
        j = n - 1
        while  i >= 0 and j >= 0 :
            if A[i]>B[j] :
                A[pos]=A[i]
                pos-=1
                i-=1
            else :
                A[pos]=B[j]
                pos-=1
                j-=1
                
        while i >= 0 :
            A[pos] = A[i]
            
            pos-=1
            i-=1
        while j >= 0:
            A[pos] = B[j]
            pos-=1
            j-=1


# 839. 合并两个排序的间隔列表
# []
# https://www.lintcode.com/problem/merge-two-sorted-interval-lists/description
# https://www.jiuzhang.com/solutions/merge-two-sorted-interval-lists/#tag-lang-python
class Solution:
    """
    @param list1: one of the given list
    @param list2: another list
    @return: the new sorted list of interval
    """
    def mergeTwoInterval(self, list1, list2):
        i, j = 0, 0
        intervals = []
        while i < len(list1) and j < len(list2):
            if list1[i].start < list2[j].start:
                self.push_back(intervals, list1[i])
                i += 1
            else:
                self.push_back(intervals, list2[j])
                j += 1
        while i < len(list1):
            self.push_back(intervals, list1[i])
            i += 1
        while j < len(list2):
            self.push_back(intervals, list2[j])
            j += 1
        
        return intervals
        
    def push_back(self, intervals, interval):
        if not intervals:
            intervals.append(interval)
            return
        
        last_interval = intervals[-1]
        if last_interval.end < interval.start:
            intervals.append(interval)
            return
        
        intervals[-1].end = max(intervals[-1].end, interval.end)


# 486. 合并k个排序数组
# []
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


# 577. 合并K个排序间隔列表
# []
# https://www.lintcode.com/problem/merge-k-sorted-interval-lists/description
# https://www.jiuzhang.com/solutions/merge-k-sorted-interval-lists/#tag-lang-python
import heapq
class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    def mergeKSortedIntervalLists(self, intervals):
        result = []
        lst = []
        for index, item in enumerate(intervals):
            if len(item) == 0:
                continue
            heapq.heappush(lst, (item[0].start, item[0].end, index, 0))

        while lst:
            start, end, i, j = lst[0]
            self.append_and_merge(Interval(start, end), result)
            heapq.heappop(lst)
            if j + 1 < len(intervals[i]):
                heapq.heappush( lst, (intervals[i][j+1].start, intervals[i][j+1].end, i, j+1) )
        return result

    def append_and_merge(self, interval, res):
        if not res :
            res.append(interval)
            return
        
        if res[-1].end < interval.start:
            res.append(interval)
            return

        res[-1].end = max( res[-1].end, interval.end )


# 47. 两数组的交集
# []
# https://www.lintcode.com/problem/intersection-of-two-arrays/description
# https://www.jiuzhang.com/solutions/intersection-of-two-arrays/#tag-lang-python
class Solution:
    """
    @param nums1: an integer array
    @param nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        # write your code here   
        s1, s2 = set(nums1), set(nums2)
        return [x for x in s1 if x in s2]


# 548. 两数组的交集 II
# []
# https://www.lintcode.com/problem/intersection-of-two-arrays-ii/description
# https://www.jiuzhang.com/solutions/intersection-of-two-arrays-ii/#tag-lang-python
class Solution:
    # @param {int[]} nums1 an integer array
    # @param {int[]} nums2 an integer array
    # @return {int[]} an integer array
    def intersection(self, nums1, nums2):
        # Write your code here
        counts = collections.Counter(nums1)
        result = []

        for num in nums2:
            if counts[num] > 0:
                result.append(num)
                counts[num] -= 1

        return result

# 793. 多个数组的交集
# []
# https://www.lintcode.com/problem/intersection-of-arrays/description
# https://www.jiuzhang.com/solutions/intersection-of-arrays/#tag-lang-python
class Solution:
    """
    @param arrs: the arrays
    @return: the number of the intersection of the arrays
    """
    def intersectionOfArrays(self, arrs):
        count = {}

        for arr in arrs:
            for item in arr:
                if item not in count:
                    count[item] = 0
                count[item] += 1

        result = 0
        for item in count.keys():
            if count[item] == len(arrs):
                result += 1
        return result


# 654. 稀疏矩阵乘法
# []
# https://www.lintcode.com/problem/sparse-matrix-multiplication/description
# https://www.jiuzhang.com/solutions/sparse-matrix-multiplication/#tag-lang-python
class Solution:
    def multiply(self, A, B):
        row_vectors = self.convert_to_row_vectors(A)
        col_vectors = self.convert_to_col_vectors(B)
        
        matrix = []
        for row_vector in row_vectors:
            row = []
            for col_vector in col_vectors:
                row.append(self.multi_vector(row_vector, col_vector))
            matrix.append(row)
        return matrix
        
    def convert_to_row_vectors(self, matrix):
        vectors = []
        for row in matrix:
            vector = []
            for index, val in enumerate( row ):
                if val != 0:
                    vector.append((index, val))
            vectors.append(vector)
        return vectors
        
    def convert_to_col_vectors(self, matrix):
        n, m = len(matrix), len(matrix[0])
        vectors = []
        for j in range(m):
            vector = []
            for i in range(n):
                if matrix[i][j] != 0:
                    vector.append( (i, matrix[i][j]) )
            vectors.append(vector)
        return vectors

    def multi_vector(self, v1, v2):
        i, j = 0, 0
        result = 0
        
        while i < len(v1) and j < len(v2):
            if v1[i][0] < v2[j][0]:
                i += 1
            elif v1[i][0] > v2[j][0]:
                j += 1
            else:
                result += v1[i][1] * v2[j][1]
                i += 1
                j += 1
        return result
        

""" Medias """
# 65. 两个排序数组的中位数（4）
# []
# https://www.lintcode.com/problem/median-of-two-sorted-arrays/description
# https://www.jiuzhang.com/solution/median-of-two-sorted-arrays/#tag-lang-python
class Solution:
    def findMedianSortedArrays(self, A, B):
        n = len(A) + len(B)
        if n % 2 == 1:
            return self.findKth(A, 0, B, 0, n//2 + 1 )
        else:
            a = self.findKth(A, 0, B, 0, n//2 )
            b = self.findKth(A, 0, B, 0, n//2 + 1)
            return (a + b) / 2

    def findKth(self, A, index_a, B, index_b, k):
        if len(A) == index_a:
            return B[index_b + k-1]
        if len(B) == index_b:
            return A[index_a + k-1]

        if k == 1:
            return min(A[index_a], B[index_b])

        pivot = k//2 - 1
        a = A[index_a + pivot] if index_a + pivot < len(A) else None
        b = B[index_b + pivot] if index_b + pivot < len(B) else None

        # 4 cases: A None; B None; a < b; a >= b
        if b is None or (a is not None and a < b):
            return self.findKth(A, index_a + k//2, B, index_b, k- k//2)
        else:
            return self.findKth(A, index_a, B, index_b + k//2, k- k//2)


# 931. K 个有序数组的中位数 
# []
# https://www.lintcode.com/problem/median-of-k-sorted-arrays/description
# https://www.jiuzhang.com/solutions/median-of-k-sorted-arrays/#tag-lang-python
class Solution:
    def findMedian(self, nums):
        if not nums: return 0.0
            
        n = sum(len(arr) for arr in nums)
        if n == 0: return 0.0
            
        if n % 2 == 1:
            return self.find_kth(nums, n // 2 + 1) * 1.0
        return (self.find_kth(nums, n // 2) + self.find_kth(nums, n // 2 + 1)) / 2.0

    def find_kth(self, arrs, k):
        start, end = self.get_range(arrs)
        
        while start+1 < end:
            mid = (start + end) // 2
            if self.get_small_or_equal(arrs, mid) < k:
                start = mid
            else:
                end = mid
        
        if self.get_small_or_equal(arrs, start) >= k:
            return start
        return end

    def get_range(self, arrs):
        start = min( [arr[0] for arr in arrs if len(arr)] )
        end =   max( [arr[-1] for arr in arrs if len(arr)] )
        return start, end
    
    def get_small_or_equal(self, arrs, val):
        # count = 0
        # for arr in arrs:
        #     count += self.get_small_or_equal_in_arr(arr, val)
        return sum( [self.get_small_or_equal_in_arr(arr, val) for arr in arrs] )
    
    def get_small_or_equal_in_arr(self, arr, val):
        if not arr: return 0
        
        start, end = 0, len(arr) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if arr[mid] > val:
                end = mid
            else:
                start = mid
        
        if arr[start] > val: return start
        if arr[end] > val:   return end
        return end + 1


# 149. 买卖股票的最佳时机
# []
# https://www.lintcode.com/problem/best-time-to-buy-and-sell-stock/description
# https://www.jiuzhang.com/solutions/best-time-to-buy-and-sell-stock/#tag-lang-python


# 405. 和为零的子矩阵
# []
# https://www.lintcode.com/problem/submatrix-sum/description
# https://www.jiuzhang.com/solutions/submatrix-sum/#tag-lang-python
# Version 1: n**4
class Solution:
    def submatrixSum(self, matrix):
        if not matrix or not matrix[0]: return None
        
        n, m, result = len(matrix), len(matrix[0]), []

        pre = [[0 for i in range(m+1)] for j in range(n+1)]
        pre[1][1] = matrix[0][0]

        for i in range(2, n + 1):
            pre[i][1] = pre[i-1][1] + matrix[i-1][0]
        
        for j in range(2, m + 1):
            pre[1][j] = pre[1][j-1] + matrix[0][j-1]

        for i in range(2, n + 1):
            for j in range(2, m + 1):
                pre[i][j] = pre[i][j-1] + pre[i-1][j] - pre[i-1][j-1] + matrix[i-1][j-1]
        
        for x1 in range(1, n+1):
            for y1 in range(1, m+1):
                for x2 in range(x1, n+1):
                    for y2 in range(y1, m+1):
                        if pre[x2][y2] - pre[x1-1][y2] - pre[x2][y1-1] + pre[x1-1][y1-1] == 0:
                            result.append( [x1-1, y1-1] )
                            result.append( [x2-1, y2-1] )
                            return result

# version 2: n**3
class Solution:
    def submatrixSum(self, matrix):
        if not matrix or not matrix[0]: return None

        n, m = len(matrix), len(matrix[0])
        for top in range( n ):
            arr = [0] * m
            for bottom in range(top, n):
                prefix_hash, prefix_sum = {0:-1}, 0

                for col in range(m):
                    arr[col]   += matrix[bottom][col] 
                    prefix_sum += arr[col]

                    if prefix_sum in prefix_hash:
                        return [(top, prefix_hash[prefix_sum] + 1), (bottom, col)]
                    prefix_hash[prefix_sum] = col
        return None


# 944. 最大子矩阵
# []
# https://www.lintcode.com/problem/maximum-submatrix/description
# https://www.jiuzhang.com/solutions/maximum-submatrix/#tag-lang-python
class Solution:
    def maxSubmatrix(self, matrix):
        if not matrix or not matrix[0]: 
            return 0 
            
        maxSum = -float('inf')
        rows, columns = len(matrix), len(matrix[0])
        for topRow in range(rows):
            compressedRow = [0] * columns 

            for row in range(topRow, rows):
                minSum, nextPrefixSum = 0, 0 
                for col in range(columns):
                    compressedRow[col] += matrix[row][col]
                    nextPrefixSum      += compressedRow[col]
                    maxSum = max(maxSum, nextPrefixSum - minSum)
                    minSum = min(minSum, nextPrefixSum)
        return max


# 943. 区间和查询 - 不可变的
# []
# https://www.lintcode.com/problem/range-sum-query-immutable/description
# https://www.jiuzhang.com/solutions/range-sum-query-immutable/#tag-lang-python
class NumArray(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.sum = [0]
        for i in nums:
            self.sum += self.sum[-1] + i, # `,` 

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sum[j+1] - self.sum[i]

# 817. 范围矩阵元素和-可变的
# []
# https://www.lintcode.com/problem/range-sum-query-2d-mutable/description
# https://www.jiuzhang.com/solutions/range-sum-query-2d-mutable/#tag-lang-python
class NumMatrix(object):
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.matrix  = [[0 for _ in range(n+1) ] for _ in range(m+1)]
        self.fenwick = [[0 for _ in range(n+1) ] for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                self.update(i, j, matrix[i][j])

    def update(self, row, col, val):
        row += 1; col += 1
        delta = val - self.matrix[row][col]
        self.matrix[row][col] = val

        i, j, numRows, numCols = row, col, len(self.fenwick), len(self.fenwick[0])
        while i < numRows:
            j = col
            while j < numCols:
                self.fenwick[i][j] += delta
                j += self.lowbit(j)
            i += self.lowbit(i)
        
    def prefixSum(self, row, col):
        i, j, result = row, col, 0
        while i != 0:
            j = col
            while j != 0:
                result += self.fenwick[i][j]
                j -= self.lowbit(j)
            i -= self.lowbit(i)
        return result
    
    def sumRegion(self, row1, col1, row2, col2):
        row2 += 1
        col2 += 1
        return self.prefixSum(row2, col2) + self.prefixSum(row1, col1) -\
               self.prefixSum(row1, col2) - self.prefixSum(row2, col1) 
    
    def lowbit(self, k):
        return k & -k


# []
# 840. 可变范围求和
# https://www.lintcode.com/problem/range-sum-query-mutable/description
# https://www.jiuzhang.com/solutions/range-sum-query-mutable/#tag-lang-python
class NumArray:
    def __init__(self, nums):
        self.arr, self.n = nums, len(nums)
        self.bit = [0] * (self.n+1)
        for i in range(self.n):
            self.add(i, self.arr[i])
    
    def add(self, idx, val):
        idx += 1
        while idx <= self.n:
            self.bit[idx] += val
            idx += self.lowbit(idx)
    
    def lowbit(self, x):
        return x & (-x)

    def update(self, i, val):
        self.add(i, val - self.arr[i])
        self.arr[i] = val
    
    def sum(self, idx):
        idx += 1
        res = 0
        while idx > 0:
            res += self.bit[idx]
            idx -= self.lowbit(idx)
        return res
    
    def sumRange(self, i, j):
        return self.sum(j) - self.sum(i-1)


# 249. 统计前面比自己小的数的个数
# []
# https://www.lintcode.com/problem/count-of-smaller-number-before-itself/description
# https://www.jiuzhang.com/solutions/count-of-smaller-number-before-itself/#tag-lang-python
import sys
class BITree:
    def __init__(self, num_range):
        self.bit = [0] * (num_range + 1)

    def lowbit(self, x):
        return x & (-x)
        
    def update(self, index, val):
        i = index + 1 
        while i < len(self.bit):
            self.bit[i] += val
            i += self.lowbit(i)
    
    def getPresum(self, index):
        presum = 0
        i = index + 1 
        while i > 0:
            presum += self.bit[i]
            i -= self.lowbit(i)
        
        return presum
        
class Solution:
    """
    @param A: an integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def countOfSmallerNumberII(self, A):
        if not A: return []
            
        smallest, largest = sys.maxsize, -sys.maxsize
        for a in A:
            smallest = min(smallest, a)
            largest  = max(largest, a)

        bit = BITree(largest - smallest + 1)
        
        result = []
        for a in A:
            #result.append(bit.getPresum(a - 1))
            #bit.update(a, 1)
            result.append(bit.getPresum(a - smallest - 1)) # ? meaning
            bit.update(a - smallest, 1)
            
        return result



