"""
Note：
经典二分法 -> First/last Position


* start < end 无论如何都容易发生死循环：
    eg:[-1,0,3,5,9,12]
        0 5 2 (start, end, mid)
        2 5 3
        3 5 4
        3 4 3
        3 4 3
        ...

* 常见痛点：`循环结束条件`，`指针变化`
"""

# 14/704. 二分查找 
# [2020年10月21日 2020年2月22日]
# https://www.lintcode.com/problem/classical-binary-search/description
# https://leetcode-cn.com/problems/binary-search/
class Solution:
    def search(self, nums, target):
        if not nums: return -1
        start, end = 0, len(nums)-1

        while start + 1 < end:
            mid = (start + end)//2
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        
        if nums[start] == target: return start
        if nums[end] == target: return end
        
        return -1


# 61. 搜索区间
# [2020年10月22日 2020年2月22日]
# https://www.lintcode.com/problem/search-for-a-range/description
class Solution:
    """
    @param A: an integer sorted array
    @param target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        n = len(A) 
        if not n: return [-1, -1] # error: the case []
        return [self.findFirstTargetNum(A, target, n), self.findLastTargetNum(A, target, n)]   

    def findFirstTargetNum(self, nums, target, n):
        start, end = 0, n -1

        while start + 1 < end:
            mid = (start+end) //2
            # ! the target side first
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[start] == target: return start
        if nums[end] == target: return end
        return -1 

    def findLastTargetNum(self, nums, target, n):
        start, end = 0, n -1

        while start + 1 < end:
            mid = (start+end) //2
            if nums[mid] > target:
                end = mid
            else:
                start = mid
        if nums[end] == target: return end
        if nums[start] == target: return start
        return -1 


# 74. 第一个错误的代码版本
# [2020年10月21日]
# https://www.lintcode.com/problem/first-bad-version/description
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        left, right = 0, n
        while left+1 < right:
            mid = (left + right) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid
        
        return left if isBadVersion(left) else right


# 460/658. 在排序数组中找最接近的K个数
# [2020年10月22日]
# https://www.lintcode.com/problem/find-k-closest-elements/description https://leetcode-cn.com/problems/find-k-closest-elements/
# version Lintcode
class Solution:
    def kClosestNumbers(self, arr, target, k):
        right   = self.findUpperClosest( arr, target )
        left    = right - 1
        results = []
        
        for _ in range(k):
            if self.isLeftCloser(arr, target, left, right):
                results.append(arr[left])
                left -= 1
            else:
                results.append(arr[right])
                right += 1
        return results

    def findUpperClosest(self, arr, target):
        start, end = 0, len(arr)-1
        while start + 1 < end:
            mid =(start + end) // 2
            if arr[mid] < target:
                start = mid
            else:
                end = mid
        
        # error: `>`-> `>=`
        if arr[start] >= target: return start
        if arr[end] >= target: return end
        return len(arr)

    def isLeftCloser(self, arr, target, left, right):
        if left < 0:
            return False
        if right >= len(arr):
            return True
        return target - arr[left] <= arr[right] - target

# version Leetcode
class Solution:
    def findClosestElements(self, arr, k, target):
        right = self.findUpperClosest( arr, target )
        print(right, arr[right])
        left = right - 1
        results = []
        
        for _ in range(k):
            if self.isLeftCloser(arr, target, left, right):
                results.append(arr[left])
                left -= 1
            else:
                results.append(arr[right])
                right += 1
        results.sort()

        return results

    def findUpperClosest(self, arr, target):
        start, end = 0, len(arr)-1
        while start + 1 < end:
            mid =(start + end) // 2
            if arr[mid] < target:
                start = mid
            else:
                end = mid
        # error: `>`-> `>=`
        if arr[start] >= target: return start
        if arr[end] >= target: return end
        return len(arr)

    def isLeftCloser(self, arr, target, left, right):
        if left < 0:
            return False
        if right >= len(arr):
            return True
        return target - arr[left] <= arr[right] - target


# 447. 在大数组中查找
# [2020年10月21日]
# https://www.lintcode.com/problem/search-in-a-big-sorted-array/note
class Solution:
    """
    @param reader: An instance of ArrayReader.
    @param target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        n = 1
        while reader.get(n) < target:
            n *= 2
        
        start, end = n//2, n
        while start + 1 < end:
            mid = (start + end) // 2 
            if reader.get(mid) < target:
                start = mid
            else:
                end = mid
        
        if reader.get(start) == target: return start
        if reader.get(end) == target: return end
        return -1 


# 159/152. 寻找旋转排序数组中的最小值
# [2020年10月22日]
# https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description https://leetcode-cn.com/problems/maximum-product-subarray/
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = (start + end) // 2
            # ! compare with the `end`
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
        
        return nums[start] if nums[start] < nums[end] else nums[end]


# 585/852. 山脉序列中的最大值
# [2020年10月22日]
# https://www.lintcode.com/problem/maximum-number-in-mountain-sequence/description
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        start, end = 0, len(nums)-1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] < nums[mid+1]:
                start = mid
            else:
                end = mid
        return max( nums[start], nums[end] )


# 28. 搜索二维矩阵
# [2020年10月22日]
# https://www.lintcode.com/problem/search-a-2d-matrix/description
class Solution:
    """
    @param matrix: matrix, a list of lists of integers
    @param target: An integer
    @return: a boolean, indicate whether matrix contains target
    """
    def searchMatrix(self, matrix, target):
        if matrix is None or matrix[0] is None: return False
        
        self.n, self.m = len(matrix), len(matrix[0])
        start, end = 0, self.m * self.n -1
        
        while start + 1 < end:
            mid = (start+end) //2
            if self.getVal(matrix, mid) < target:
                start = mid
            else:
                end = mid
                
        return True if  self.getVal(matrix, start) == target or self.getVal(matrix, end) == target else False 

    def getVal(self, matrix, index):
        # error: `self.n` -> `self.m`
        return matrix[index // self.m][index % self.m]

        
# 38. 搜索二维矩阵 II
# [2020年10月22日]
# https://www.lintcode.com/problem/search-a-2d-matrix-ii/description
class Solution:
    """
    @param matrix: A list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicate the total occurrence of target in the given matrix
    """
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]: return 0

        m, n = len(matrix), len(matrix[0])
        i, j = m - 1, 0
        count = 0

        while i >= 0 and j < n: # error:   `j < m` -> 'j < n'
            if matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
            else:
                count += 1
                i -= 1
        return count



# 462. 目标出现总和
# [2020年10月22日]
# https://www.lintcode.com/problem/total-occurrence-of-target/description
class Solution:
    def totalOccurrence(self, A, target):
        # write your code here
        n = len(A) 
        if not n: return 0
        last = self.findLastTargetNum(A, target, n)
        first = self.findFirstTargetNum(A, target, n)
        
        if first == -1 and last == -1:
            return  0
        else:
            return last - first + 1

    def findFirstTargetNum(self, nums, target, n):
        start, end = 0, n -1

        while start + 1 < end:
            mid = (start+end) //2
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[start] == target: return start
        if nums[end] == target: return end
        return -1 

    def findLastTargetNum(self, nums, target, n):
        start, end = 0, n -1

        while start + 1 < end:
            mid = (start+end) //2
            if nums[mid] > target:
                end = mid
            else:
                start = mid
        if nums[end] == target: return end
        if nums[start] == target: return start
        return -1 


# 600/302. 包裹黑色像素点的最小矩形 # TODO
# [2020年10月22日]
# https://www.lintcode.com/problem/smallest-rectangle-enclosing-black-pixels/description
class Solution:
    def minArea(self, image, x, y):
        if not image or not image[0]:
            return 0
        rows, cols = len(image), len(image[0])

        up    = self.bin_search(image, 0, x+1,  True,  True)
        down  = self.bin_search(image, x, rows, True,  False)
        left  = self.bin_search(image, 0, y+1,  False, True)
        right = self.bin_search(image, y, cols, False, False)
        
        return (down-up)*(right-left)

    def check_row(self,image, r):
        for char in image[r]:
            if char == '1':
                return True
        return False

    def check_col(self,image, c):
        for row in range(len(image)):
            if image[row][c] == '1':
                return True
        return False

    def bin_search(self,image, lo, hi, row_col, left_right):
        # FIXME: change to your own style
        while lo < hi:
            mid = (lo + hi) // 2
            if row_col:
                flag = self.check_row(image, mid)
            else:
                flag = self.check_col(image, mid)
        
            if flag:
                if left_right:
                    hi = mid
                else:
                    lo = mid + 1
            else:
                if left_right:
                    lo = mid + 1
                else:
                    hi = mid
        return lo


# 75/162. 寻找峰值
# [2020年10月26日]
# https://www.lintcode.com/problem/find-peak-element/description
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, nums):
        start, end = 0, len(nums)-1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] < nums[mid+1]:
                start = mid
            else:
                end = mid
        return start if nums[start] > nums[end] else end


# 62/33. 搜索旋转排序数组 ️️⭐ # TODO, draw map to solve it 
# [2020年10月26日]
# https://www.lintcode.com/problem/search-in-rotated-sorted-array/description
# https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
class Solution:
    def search(self, nums, target):
        if not nums: return -1 # ! `or not nums[0]` 

        start, end = 0, len(nums)-1
        while start+1 < end:
            mid = (start+end)//2
            if nums[mid] > nums[end]:
                if nums[start] <= target <= nums[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if nums[mid] <= target <= nums[end]:
                    start = mid
                else:
                    end = mid
        if nums[start] == target: return start
        if nums[end] == target: return end
    
        return -1


# 63. 搜索旋转排序数组 II
# DESC What if duplicates are allowed?
# [2020年10月26日]
# https://www.lintcode.com/problem/search-in-rotated-sorted-array-ii/description
class Solution:
    def search(self, nums, target):
        if not nums: return False

        start, end = 0, len(nums)-1
        while start+1 < end:
            # DESC Similiar as previous problem, just keep going if duplicate
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


# 140. 快速幂 # ! 数学原理不熟悉
# [2020年10月26日]
# https://www.lintcode.com/problem/fast-power/description
# https://www.jiuzhang.com/solution/fast-power/#tag-lang-python
class Solution:
    def fastPower(self, a, b, n):
        if n == 0: return 1 % b
        if n == 1: return a % b

        power = self.fastPower(a, b, n // 2)
        power = power * power % b

        if n % 2 == 1:
            power = (power * a) % b
        
        return power


# 428. x的n次幂 # todo 熟悉
# [2020年10月26日]
# https://www.lintcode.com/problem/powx-n/description
# https://www.jiuzhang.com/solutions/powx-n/#tag-lang-python
class Solution:
    def myPow(self, x, n):
        if n < 0 :
            x, n = 1 / x, -n

        ans, tmp = 1, x

        while n != 0:
            if n % 2 == 1:
                ans *= tmp
            tmp *= tmp
            n = n//2
            
        return ans


