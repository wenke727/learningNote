import sys

""" 随课教程预习题 """
# 604. 滑动窗口内数的和
# [2020年10月27日 2021年8月3日]
# https://www.lintcode.com/problem/window-sum/description
# version: prefix sum 
class Solution:
    def winSum(self, nums, k):
        if k == 0: 
            return []
        
        n = len(nums)
        prefix, res = [0] * (n+1), []

        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        for i in range(k, n+1):
            res.append( prefix[i] - prefix[i-k] )

        return res
# version 2
class Solution:
    def winSum(self, nums, k):
        if k == 0: 
            return []
        
        n = len(nums)
        left, right, val_sum, res = 0, k-1, 0, []
        for i in range(k):
            val_sum += nums[i]
        
        while right < n:
            res.append(val_sum)
            right += 1
            if right < n: # ! out of range
                val_sum = val_sum + nums[right] - nums[left]
            left += 1

        return res


# 521. 去除重复元素
# [2020年10月27日 2021年8月3日]
# https://www.lintcode.com/problem/remove-duplicate-numbers-in-array/description
# https://www.jiuzhang.com/solution/remove-duplicate-numbers-in-array/#tag-lang-python
# version 1
class Solution:
    def deduplication(self, nums):
        if not nums: 
            return 0

        dict_nums, res = {}, 0
        for num in nums:
            if num in dict_nums:
                continue
            dict_nums[num] = True
            nums[res] = num
            res += 1

        return res
# version 2
class Solution:
    def deduplication(self, nums):
        if not nums: 
            return 0
        
        nums.sort()
        left = 1
        for right in range(1, len(nums)):
            if nums[right-1] == nums[right]:
                continue
            nums[left] = nums[right]
            left += 1

        return left


# 102. 带环链表
# [2020.11.13 2021年8月3日]
# https://www.lintcode.com/problem/linked-list-cycle/description
class Solution:
    def hasCycle(self, head):
        if head is None: 
            return False

        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False


# 103. 带环链表 II
# [2020年10月27日 2021年8月3日]
# https://www.lintcode.com/problem/linked-list-cycle-ii/description
class Solution:
    def detectCycle(self, head):
        if head is None: 
            return None

        intersect = self.getIntersect(head)
        if intersect is None: 
            return None

        ptr1, ptr2 = head, intersect
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        
        return ptr1

    def getIntersect(self, head):
        slow = fast = head
        # ! Pay attention to the boundary conditions
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return slow

        return None


# 380. 两个链表的交叉 
# [2021年8月4日]
# https://www.lintcode.com/problem/intersection-of-two-linked-lists/description
# https://www.jiuzhang.com/solution/intersection-of-two-linked-lists/#tag-lang-python
# version 1
class Solution:
    def getIntersectionNode(self, headA, headB):
        if not headA or not headA.next or not headB:
            return None 
            
        tailA = headA
        while tailA.next:
            tailA = tailA.next
            
        tailA.next = headB
        
        slow, fast = headA, headA
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next 
            if slow is fast:
                break
            
        if slow is fast:
            slow = headA 
            while slow is not fast:
                slow = slow.next
                fast = fast.next 
            tailA.next = None
            return slow
        
        tailA.next = None
        return None
# version 2
class Solution:
    def getIntersectionNode(self, headA, headB):
        if headA is None or headB is None:
            return None

        size_A = self.get_list_size(headA)
        size_B = self.get_list_size(headB)
        curr_A, curr_B = headA, headB

        diff = abs(size_A - size_B)

        if size_A > size_B:
            for i in range(diff):
                curr_A = curr_A.next
        elif size_B > size_A:
            for i in range(diff):
                curr_B = curr_B.next

        while curr_A and curr_B and curr_A != curr_B:
            curr_A = curr_A.next
            curr_B = curr_B.next

        if curr_A is not None and curr_B is not None:
            return curr_A
        
        return None


    def get_list_size(self, head):
        size = 0
        curr = head
        while curr:
            curr = curr.next
            size += 1

        return size


"""双指针"""
# 539/283. 移动零
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/move-zeroes/description 
# https://leetcode-cn.com/problems/move-zeroes/
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


# 415/125. 有效回文串
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/valid-palindrome/description
class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1

            while left < right and not s[right].isalnum():
                right -= 1
            
            if left < right and s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1

        return True


# 891/680. 有效回文 II
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/valid-palindrome-ii/description
# DESC 给一个非空字符串 s，你最多可以删除一个字符。判断是否可以把它变成回文串
class Solution:
    def validPalindrome(self, s: str):
        left, right = self.twoPointer(s, 0, len(s) - 1)
        if left >= right:
            return True
        return self.isPalindrome(s, left + 1, right) or self.isPalindrome(s, left, right - 1) 

    def isPalindrome(self, s, left, right):
        left, right = self.twoPointer(s, left, right)
        return left >= right

    def twoPointer(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return left, right
            left += 1
            right -= 1
        return left, right


# 607. 两数之和 III-数据结构设计
# [2020年10月27日]
# https://www.lintcode.com/problem/two-sum-iii-data-structure-design/description
# https://www.jiuzhang.com/solution/two-sum-iii-data-structure-design/#tag-lang-python
class TwoSum:
    def __init__(self):
        self.nums = []
        self.is_sorted = False

    def add(self, number):
        self.nums.append(number)
        self.is_sorted = False
        pass

    def find(self, value):
        if len(self.nums) < 2: 
            return False
        
        if not self.is_sorted:
            self.nums.sort()
            self.is_sorted = True
        
        start, end = 0, len(self.nums) - 1
        while start < end:
            val_sum = self.nums[start] + self.nums[end]
            if val_sum == value:
                return True
            elif val_sum < value:
                start += 1
            else:
                end -= 1

        return False


# 608. 两数和 II-输入已排序的数组
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/two-sum-ii-input-array-is-sorted/description
# https://www.jiuzhang.com/solution/two-sum-ii-input-array-is-sorted/#tag-lang-python
class Solution:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        start, end = 0, len(nums) - 1
        while start < end:
            val_sum = nums[start] + nums[end]
            if val_sum == target:
                return [start+1, end+1]
            elif val_sum < target:
                start += 1
            else:
                end -= 1

        return []


# 587. 两数之和 - 不同组成
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/two-sum-unique-pairs/description
# https://www.jiuzhang.com/solutions/two-sum-unique-pairs/#tag-lang-python
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        if not nums or len(nums) < 2: 
            return 0
        
        nums.sort()
        start, end = 0, len(nums) - 1
        count, last_pair = 0, (None, None) 

        while start < end:
            val_sum = nums[start] + nums[end]
            if val_sum == target:
                count += 1
                start += 1
                end -= 1

                # caution, start here was already plus 1
                while start < end and nums[start] == nums[start-1]:
                    start += 1
                while start < end and nums[end] == nums[end+1]:
                    end -= 1

            elif val_sum < target:
                start += 1
            else:
                end -= 1
        return count


# 57/15. 三数之和
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/3sum/description
# https://www.jiuzhang.com/solutions/3sum/#tag-lang-python
class Solution:
    def threeSum(self, nums):
        length = len(nums)
        if nums is None or length < 3: 
            return []

        nums.sort()
        res = []

        for i in range(length - 2):
            # caution: border jedgement
            if i - 1 >= 0 and nums[i] == nums[i-1]:
                continue
            self.findTwoSum(nums, i+1, length-1, -nums[i], res)
        return res
    
    def findTwoSum(self, nums, start, end, target, res):
        while start < end:
            val_sum = nums[start] + nums[end]
            if val_sum == target:
                res.append([-target, nums[start], nums[end]])
                start += 1
                end -= 1
                
                while start < end and nums[start] == nums[start-1]:
                    start += 1
                while start < end and nums[end] == nums[end+1]:
                    end -= 1
            
            elif val_sum < target:
                start += 1
            else:
                end -= 1
        return


# 382. 三角形计数
# [2020年10月27日 2021年8月4日]
# https://www.lintcode.com/problem/triangle-count/description
# https://www.jiuzhang.com/solutions/triangle-count/#tag-lang-python
class Solution:
    def triangleCount(self, nums):
        n = len(nums)
        if n < 3: 
            return -1
        
        nums.sort()
        count = 0
        # ! the maximum value after sorting the nums
        for i in range(2, n):
            left, right = 0, i - 1
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    count += right - left
                    right -= 1
                else:
                    left += 1
        
        return count


# 609. 两数和-小于或等于目标值
# [2020年10月29日 2021年8月4日]
# https://www.lintcode.com/problem/two-sum-less-than-or-equal-to-target/description
# https://www.jiuzhang.com/solutions/two-sum-less-than-or-equal-to-target/#tag-lang-python
class Solution:
    def twoSum5(self, nums, target):
        if not nums: 
            return 0

        nums.sort() 
        left, right = 0, len(nums) - 1
        res = 0
        while  left < right:
            if nums[left] + nums[right] > target:
                right -= 1
            else:
                res += right - left
                left += 1
        
        return res


# 443. 两数之和 II
# [2020年10月29日 2021年8月4日]
# https://www.lintcode.com/problem/two-sum-greater-than-target/description
# https://www.jiuzhang.com/solutions/two-sum-greater-than-target/#tag-lang-python
class Solution:
    def twoSum2(self, nums, target):
        if not nums: 
            return 0

        nums.sort()
        left, right = 0, len(nums) - 1
        res = 0

        while left < right:
            if nums[right] + nums[left] > target:
                res += right - left
                right -= 1
            else:
                left += 1
        return res


# 533. 两数和的最接近值
# [2020年10月30日 2021年8月4日]
# https://www.lintcode.com/problem/two-sum-closest-to-target/description
# https://www.jiuzhang.com/solutions/two-sum-closest-to-target/#tag-lang-python
class Solution:
    """
    @param nums: an integer array
    @param target: An integer
    @return: the difference between the sum and the target
    """
    def twoSumClosest(self, nums, target):
        if not nums: 
            return None

        left, right = 0, len(nums) - 1
        diff = sys.maxsize
        nums.sort()

        while left < right:
            val_sum = nums[left] + nums[right]
            
            if abs(val_sum - target) < diff:
                diff = abs(val_sum - target)       

            if val_sum < target:
                left += 1
            else:
                right -= 1

        return diff


# 59. 最接近的三数之和
# [2020年11月1日 2021年8月4日]
# https://www.lintcode.com/problem/3sum-closest/description
# https://www.jiuzhang.com/solutions/3sum-closest/#tag-lang-python
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """
    def threeSumClosest(self, numbers, target):
        numbers.sort()
        ans = None

        for i in range(len(numbers)):
            left, right = i + 1, len(numbers) - 1
            while left < right:
                sum = numbers[left] + numbers[right] + numbers[i]
                if ans is None or abs(sum - target) < abs(ans - target):
                    ans = sum
                    
                if sum < target:
                    left += 1
                else:
                    right -= 1
        return ans


# 610. 两数和 - 差等于目标值
# [2020年11月13日 2021年8月4日]
# https://www.lintcode.com/problem/two-sum-difference-equals-to-target/description
# https://www.jiuzhang.com/solutions/two-sum-difference-equals-to-target/
class Solution:
    """
    @param nums: an array of Integer
    @param target: an integer
    @return: [num1, num2] (num1 < num2)
    """
    def twoSum7(self, nums, target):
        if not nums: 
            return [-1, -1]
        
        nums.sort()
        target = abs(target)
        j = 1
        for i in range(len(nums)):
            j = max(j, i+1)
            while j < len(nums) and nums[j] - nums[i] < target:
                j += 1
            
            if j >= len(nums):
                break
            if nums[j] - nums[i] == target:
                return [nums[i], nums[j]]
        
        return [-1, -1]


# 58. 四数之和
# [2020年11月13日]
# https://www.lintcode.com/problem/4sum/description
# https://www.jiuzhang.com/solution/4sum/#tag-lang-python
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, nums, target):
        n = len(nums)
        if n < 4: return []
        
        nums.sort()
        res = []
        for i in range(n-4):
            if i > 0 and nums[i] == nums[i-1]:
                continue

            for j in range(i+1, n-3):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                
                pairs = self.findTwoSum( nums, j+1, n-1, target - nums[i] -nums[j] )
                for c, d in pairs:
                    res.append( [nums[i], nums[j], c, d] )
                    
        return res
    
    def findTwoSum(self, nums, left, right, target):
        pairs = []
        
        while left < right:
            if nums[left] + nums[right] == target:
                if not pairs or (nums[left], nums[right]) != pairs[-1]:
                    pairs.append( (nums[left], nums[right]) )
                left += 1
                right -= 1
            elif nums[left] + nums[right] < target:
                left += 1
            else:
                right -= 1
        return pairs


"""Partition Array"""
"""
```python
# ! `left <= right`, not `left < right`
def partitionArray(self, nums, k):
    left, right =  0, len(nums) - 1
    while left <= right:
        while left <= right and con:
            left += 1
        
        while left <= right and con:
            right -= 1
        
        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    return left
```
"""# 31. 数组划分
# [2020年10月29日 2021年8月4日]
# https://www.lintcode.com/problem/partition-array/description
# https://www.jiuzhang.com/solutions/partition-array/#tag-lang-python
class Solution:
    def partitionArray(self, nums, k):
        left, right =  0, len(nums) - 1
        # ! `left < right` -> `left <= right`
        while left <= right:
            while left <= right and nums[left] < k:
                left += 1
            
            while left <= right and nums[right] >= k:
                right -= 1
            
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        return left


# 461. 无序数组K小元素 ⭐⭐
# [2020年10月29日, 2020年11月13日]
# https://www.lintcode.com/problem/kth-smallest-numbers-in-unsorted-array/description
# https://www.lintcode.com/problem/kth-smallest-numbers-in-unsorted-array/description#tag-lang-python
# DESC 对一个数组进行partition的时间复杂度为O(n)。分治，选择一边继续进行partition。所以总的复杂度为T(n) = T(n / 2) + O(n)，总时间复杂度依然为O(n)
class Solution:
    def kthSmallest(self, k, nums):
        if not nums or len(nums) < k: 
            return None

        self.quickSort(nums, 0, len(nums)-1, k-1)        
        return nums[k-1]

    def quickSort(self, nums, start, end, k):
        if start == end:
            return
        
        left, right = start, end
        pivot = (nums[left] + nums[right]) // 2
        
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1

            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        # ! intervls, `>` -> `>=`
        if k >= left:
            self.quickSort(nums, left, end, k)
        
        if k <= right:
            self.quickSort(nums, start, right, k)

        return


# 373. 奇偶分割数组
# [2020年10月29日 2021年8月6日]
# https://www.lintcode.com/problem/partition-array-by-odd-and-even/description
# https://www.jiuzhang.com/solutions/partition-array-by-odd-and-even/#tag-lang-python
class Solution:
    def partitionArray(self, nums):
        left, right = 0, len(nums) - 1

        while left <= right:
            while left <= right and nums[left] % 2 == 1:
                left += 1
        
            while left <= right and nums[right] % 2 == 0:
                right -= 1
        
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        return nums


# 144. 交错正负数
# [2020年11月1日 2021年8月6日]
# https://www.lintcode.com/problem/interleaving-positive-and-negative-numbers/description
# https://www.jiuzhang.com/solutions/interleaving-positive-and-negative-integers/#tag-lang-python
class Solution:
    """
    @param: A: An integer array.
    @return: nothing
    """
    def rerange(self, nums):
        pos = len([ a for a in nums if a >0 ])
        neg = len(nums) - pos
        self.partition(nums, pos > neg)
        self.interleave(nums, pos == neg)
    
    def partition(self, nums, startPositive ):
        flag = 1 if startPositive else -1
        left, right = 0, len(nums) - 1
        
        while left <= right:
            while left <= right and nums[left] * flag > 0:
                left += 1

            while left <= right and nums[right] * flag < 0:
                right -= 1

            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        return 
    
    def interleave(self, A, has_same_length):
        left, right = 1, len(A) - 1
        if has_same_length:
            right = len(A) - 2
            
        while left < right:
            A[left], A[right] = A[right], A[left]
            left, right = left + 2, right - 2

        return


# 49. 字符大小写排序
# [2020年11月1日]
# https://www.lintcode.com/problem/sort-letters-by-case/description
# https://www.jiuzhang.com/solutions/sort-letters-by-case/#tag-lang-python
class Solution:
    def sortLetters(self, chars):
        left, right = 0, len(chars) - 1
        while left <= right:
            while left <= right and chars[left] >= 'a' and chars[left] <= 'z':
                left += 1

            while left <= right and chars[right] >= 'A' and chars[right] <= 'Z':
                right -= 1

            if left <= right:
                chars[left], chars[right] = chars[right], chars[left]
                left += 1
                right -= 1


# 148. 颜色分类
# [2020年10月30日 2021年8月6日]
# https://www.lintcode.com/problem/sort-colors/description
# https://www.jiuzhang.com/solutions/sort-colors/#tag-lang-python
class Solution:
    def sortColors(self, nums):
        left, cur, right = 0, 0, len(nums)-1

        # ! `<=` not  `<`
        while cur <= right:
            if nums[cur] == 0:
                nums[left], nums[cur] = nums[cur], nums[left]
                left += 1
                cur += 1
            elif nums[cur] == 2:
                nums[right], nums[cur] = nums[cur], nums[right]
                right -= 1
            else:
                cur += 1


# 143. 排颜色 II ⭐⭐
# [2020年10月30日 2021年8月6日]
# https://www.lintcode.com/problem/sort-colors-ii/
# https://www.jiuzhang.com/solutions/sort-colors-ii/#tag-lang-python
class Solution:
    def sortColors2(self, colors, k):
        self.sort(colors, 1, k, 0, len(colors)-1)

        return colors

    def sort(self, nums, color_from, color_to, index0, index1):
        if color_from == color_to or index0 == index1:
            return 
        
        left, right = index0, index1
        pivot = (color_from + color_to) // 2
        
        while left <= right:
            # nums[left] < pivot
            while left <= right and nums[left] <= pivot: 
                left += 1

            while left <= right and nums[right] > pivot:
                right -= 1
            
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        self.sort( nums, color_from, pivot, index0, right )
        # pivot+1, not pivot
        self.sort( nums, pivot+1, color_to, left,  index1 ) 


"""ladder 161 习题"""
# 1343. 两字符串和
# [2020年11月13日]
# https://www.lintcode.com/problem/sum-of-two-strings/description
# https://www.jiuzhang.com/solution/sum-of-two-strings/#tag-lang-python
class Solution:
    def SumofTwoStrings(self, A, B):
        result = ""
        lenA, lenB = len(A), len(B)
        i, j = lenA - 1, lenB - 1

        while i >= 0 and j >= 0:
            temp = int(A[i]) + int(B[j])
            # temp = ord(A[i]) - 2 * ord('0') + ord(B[j])
            result = str(temp) + result
            i -= 1
            j -= 1

        if i >= 0:
            result = A[0 : i + 1] + result
        if j >= 0:
            result = B[0 : j + 1] + result

        return result


# 56. 两数之和
# https://www.lintcode.com/problem/two-sum/description
# https://www.jiuzhang.com/solution/two-sum/#tag-lang-python
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        if not numbers: return [-1, -1]
        
        nums = [ (number, index) for index, number in enumerate(numbers) ]
        nums = sorted(nums)
        
        left, right = 0, len(nums) - 1
        while left < right:
            if nums[left][0] + nums[right][0] > target:
                right -= 1
            elif nums[left][0] + nums[right][0] < target:
                left += 1
            else:
                return sorted([nums[left][1], nums[right][1]])
        
        return [-1, -1]


# 1870. 全零子串的数量
# [2020年11月15日]
# https://www.lintcode.com/problem/number-of-substrings-with-all-zeroes/description
class Solution:
    """
    @param str: the string
    @return: the number of substrings 
    """
    def stringCount(self, str):
        count = 0
        length = 0

        str = str + '1'
        for right in range(len(str)):
            if str[right] == '1':
                if length == 0:
                    continue
                count += length * (length+1) / 2
                length = 0
            else:
                length += 1
            
        return int(count)


# 1375. 至少K个不同字符的子串 # TODO
# https://www.lintcode.com/problem/substring-with-at-least-k-distinct-characters/description
# https://www.jiuzhang.com/solution/substring-with-at-least-k-distinct-characters/#tag-lang-python
class Solution:
    """
    @param s: a string
    @param k: an integer
    @return: the number of substrings there are that contain at least k distinct characters
    """
    def kDistinctCharacters(self, s, k):
        left = 0
        counter = {}
        answer = 0

        for right in range(len(s)):
            counter[s[right]] = counter.get(s[right], 0) + 1
            while left <= right and len(counter) >= k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
                    
            if len(counter) == k - 1 and left > 0 and s[left - 1] not in counter:
                answer += left

        return answer


# 1246. 替换后的最长重复字符
# https://www.lintcode.com/problem/longest-repeating-character-replacement/description
# https://www.jiuzhang.com/solution/longest-repeating-character-replacement/#tag-lang-python
# version: sliding window
class Solution:
    def characterReplacement(self, s, k):
        if not s: return 0
        if k >= len(s): return len(s)
        
        counter = {}
        left = 0
        result = 0
        maxlength = 0
        
        for right in range(len(s)):
            counter[s[right]] = counter.get(s[right], 0) + 1
            maxlength = max(maxlength, counter[s[right]])

            if right - left + 1 - maxlength > k:
                counter[s[left]] -= 1
                left += 1
            else:
                result = max(result, right - left + 1)        
        return result

class Solution:
    """
    @param s: a string
    @param k: a integer
    @return: return a integer
    """
    def characterReplacement(self, s, k):
        counter = {}
        answer = 0
        j = 0
        # max_freq记录出现最多的字符数量
        max_freq = 0
        for i in range(len(s)):
            # 当j作为下标合法 且 最少需要被替换的字母数目<=k
            while j < len(s) and j - i - max_freq <= k:
                counter[s[j]] = counter.get(s[j], 0) + 1 
                # 更新出现最多的字符数量
                max_freq = max(max_freq, counter[s[j]])
                j += 1 
            
            # 如果替换 除出现次数最多的字母之外的其他字母 的数目>k,
            # 说明有一个不能换，答案与j-i-1进行比较；
            # 否则说明直到字符串末尾替换数目都<=k，可以全部换掉 
            # 答案与子串长度j-i进行比较
            if j - i - max_freq > k:
                answer = max(answer, j - 1 - i)
            else:
                answer = max(answer, j - i) 
                
            # 起点后移一位，当前起点位置的字母个数-1
            counter[s[i]] -= 1
        return answer




