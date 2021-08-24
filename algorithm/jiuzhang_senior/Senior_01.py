# 406. 和大于S的最小子数组
# https://www.lintcode.com/problem/minimum-size-subarray-sum/description
import sys
class Solution:
    def minimumSize(self, nums, target):
        ans = sys.maxsize
        right = 0
        addup = 0

        for left in range(len(nums)):
            while right < len(nums) and addup < target:
                addup += nums[right]
                right += 1

            if addup >= target:
                ans = min(ans, right - left)
            
            addup -= nums[left]
        
        return -1 if ans == sys.maxsize else ans


# 384. 最长无重复字符的子串
# [2020.12.29]
# https://www.lintcode.com/problem/longest-substring-without-repeating-characters/description
class Solution:
    def lengthOfLongestSubstring(self, s: str):
        if not s: return 0
        max_len, cur_len, left = 0, 0, 0
        lookup = set()

        for right in range(len(s)):
            cur_len += 1

            while s[right] in lookup:
                lookup.remove( s[left] )
                left += 1
                cur_len -= 1

            if cur_len > max_len:
                max_len = cur_len
                
            lookup.add( s[right] )
        return max_len


# 32/76. 最小子串覆盖
# [2020.12.27, 2020.12.29]
# https://www.lintcode.com/problem/minimum-window-substring/description
class Solution:
    def minWindow(self, s: str, t: str):
        from collections import defaultdict
        lookup = defaultdict(int)
        for i in t:
            lookup[i] += 1

        minLen, res = float('inf'),  ''
        length, count = len(s), len(t)

        left, right = 0, 0
        while right < length:
            if lookup[s[right]] > 0:
                count -= 1
            lookup[s[right]] -= 1
            right += 1

            while count == 0:
                if  minLen > right - left:
                    minLen = right - left
                    res    = s[left: right]
                
                if lookup[s[left]] == 0:
                    count += 1
                
                lookup[s[left]] += 1
                left += 1
        return res


# 386. 最多有k个不同字符的最长子字符串: 给定字符串S，找到最多有k个不同字符的最长子串T。
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if not s: return 0
        
        max_len, left, counter = 0, 0, {}
        for right in range(len(s)):
            counter[s[right]] = counter.get( s[right], 0 ) + 1
            
            while left <= right and len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len


'''求第K小元素FollowUp'''

# 401. 排序矩阵中的从小到大第k个数
# https://www.lintcode.com/problem/kth-smallest-number-in-sorted-matrix/description
import heapq
class Solution:
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


# 543. N数组第K大元素
# https://www.lintcode.com/problem/kth-largest-in-n-arrays/descriptionß
from heapq import heappop, heappush
class Solution:
    def KthInArrays(self, arrays, k):
        ''' idea: please see the question 486 merge k sorted arrays '''
        if not arrays or len(arrays) == 0 or k <= 0:
            return

        # sort each array in arrays
        for array in arrays:
            array.sort(reverse = True)

        # use the max heap and bfs
        # initialize it by puting all first one element in each array into max heap
        max_heap = []
        for x in range(len(arrays)):
            if len(arrays[x]) != 0:
                heappush(max_heap, (-arrays[x][0], x, 0))

        count = 0
        while max_heap:
            value, x, y = heappop(max_heap)
            count += 1

            if count == k:
                return -value

            if self._is_bound(x, y + 1, arrays):
                heappush(max_heap, (-arrays[x][y + 1], x, y + 1))

    def _is_bound(self, x, y, arrays):
        return 0 <= y < len(arrays[x])


# 465. 两个排序数组和的第K小
# https://www.lintcode.com/problem/kth-smallest-sum-in-two-sorted-arrays/description
import heapq
class Solution:
    """
    @param A: an integer arrays sorted in ascending order
    @param B: an integer arrays sorted in ascending order
    @param k: An integer
    @return: An integer
    """
    def kthSmallestSum(self, A, B, k):
        if not A or not B:
            return None
            
        n, m = len(A), len(B)
        minheap = [(A[0] + B[0], 0, 0)]
        visited = set([0])
        num = None

        for _ in range(k):
            num, x, y = heapq.heappop(minheap)
            if x + 1 < n and (x + 1) * m + y not in visited:
                heapq.heappush(minheap, (A[x + 1] + B[y], x + 1, y))
                visited.add((x + 1) * m + y)
        
            if y + 1 < m and x * m + y + 1 not in visited:
                heapq.heappush(minheap, (A[x] + B[y + 1], x, y + 1))
                visited.add(x * m + y + 1)
                
        return num