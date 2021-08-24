import sys

# 390. 找峰值 II # TODO 手打一遍
# [2021年3月11日]
# https://www.lintcode.com/problem/find-peak-element-ii/description
# https://www.jiuzhang.com/solution/find-peak-element-ii/#tag-lang-python
# version O(m+n)
class Solution:
    # REF: http://courses.csail.mit.edu/6.006/spring11/lectures/lec02.pdf
    def findPeakII(self, A):
        if len(A) == 0 or len(A[0]) == 0: 
            return [-1, -1]

        left, up, right, down = 0, 0, len(A[0]) - 1, len(A) - 1
        while left + 1 < right or up + 1 < down:
            if right - left > down - up:
                c = (left + right) // 2
                r = self.findColumnPeak(A, c, up, down)
            
                if self.isPeak(A, r, c):
                    return [r, c]
                elif A[r][c] < A[r][c - 1]:
                    right = c
                else:
                    left = c
            else:
                r = (up + down) // 2
                c = self.findRowPeak(A, r, left, right)
            
                if self.isPeak(A, r, c):
                    return [r, c]
                elif A[r][c] < A[r - 1][c]:
                    down = r
                else:
                    up = r

        for r in [left, right]:
            for c in [up, down]:
                if self.isPeak(A, r, c):
                    return [r, c]
        return [-1, -1]

    def isPeak(self, A, r, c):
        return A[r][c] > max(A[r][c-1], A[r][c+1], A[r-1][c], A[r+1][c])

    def findColumnPeak(self, A, c, up, down):
        value = max(A[r][c] for r in range(up, down + 1))
        for r in range(up, down + 1):
            if A[r][c] == value:
                return r

    def findRowPeak(self, A, r, left, right):
        value = max(A[r][c] for c in range(left, right + 1))
        for c in range(left, right + 1):
            if A[r][c] == value:
                return c

# version 二分法
class Solution:
    def findPeakII(self, A):
        n = len(A)
        up, bottom = 0, n - 1

        while up + 1 < bottom:
            mid = (bottom + up) // 2
            index = self.find_maxCol(A, mid)

            #若上一行位置比当前位置值大，则下边界上移
            #若下一行位置比当前位置值大，则上边界下移
            #否则该位置为峰值，直接返回答案
            if A[mid][index] < A[mid - 1][index]: 
                bottom = mid
            elif A[mid][index] < A[mid + 1][index]: 
                up = mid
            else:
                return [mid, index]

        #比较上下边界上最大值，取较大的位置返回答案
        bottom_index = self.find_maxCol(A, bottom)
        up_index     = self.find_maxCol(A, up)
        if A[up][up_index] < A[bottom][bottom_index]:
            return [bottom, bottom_index]
        else:
            return [up, up_index]

    def find_maxCol(self, A, row):
        #find_maxCol用于找到当前行最大值所在列
        col, m = 0, len(A[0])
        for i in range(1,m):
            if A[row][col] < A[row][i]:
                col = i
        return col

# version 递归版本/二分版本
class Solution:
    def findPeakII(self, A):
        if not A or not A[0]:  return None
        return self.find_peak(A, 0, len(A) - 1, 0, len(A[0]) - 1)
        
    def find_peak(self, matrix, top, bottom, left, right):
        if top + 1 >= bottom and left + 1 >= right:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    if self.is_peak(matrix, row, col):
                        return [row, col]
            return [-1, -1]
        
        if bottom - top < right - left:
            col = (left + right) // 2
            row = self.find_col_peak(matrix, col, top, bottom)
            if self.is_peak(matrix, row, col):
                return [row, col]
            if matrix[row][col - 1] > matrix[row][col]:
                return self.find_peak(matrix, top, bottom, left, col - 1)
            return self.find_peak(matrix, top, bottom, col + 1, right)
            
        row = (top + bottom) // 2
        col = self.find_row_peak(matrix, row, left, right)
        if self.is_peak(matrix, row, col):
            return [row, col]
        if matrix[row - 1][col] > matrix[row][col]:
            return self.find_peak(matrix, top, row - 1, left, right)
        return self.find_peak(matrix, row + 1, bottom, left, right)
        
    def is_peak(self, matrix, x, y):
        return matrix[x][y] == max(
            matrix[x][y],
            matrix[x - 1][y],
            matrix[x][y - 1],
            matrix[x][y + 1],
            matrix[x + 1][y],
        )

    def find_row_peak(self, matrix, row, left, right):
        peak_val = -sys.maxsize
        peak = None
        for col in range(left, right + 1):
            if matrix[row][col] > peak_val:
                peak_val = matrix[row][col]
                peak = col
        return peak
        
    def find_col_peak(self, matrix, col, top, bottom):
        peak_val = -sys.maxsize
        peak = None
        for row in range(top, bottom + 1):
            if matrix[row][col] > peak_val:
                peak_val = matrix[row][col]
                peak = row
        return peak


# 141. 对x开根
# [2020年10月24日 2021年03月17日 2021年8月13日]
# https://www.lintcode.com/problem/sqrtx/description
class Solution:
    def sqrt(self, x):
        start, end = 0, x
        
        while start + 1 < end:
            mid = (start + end) // 2
            if self.pow(mid) > x:
                end = mid
            else:
                start = mid

        if self.pow(end) <= x: 
            return end
        
        return start

    def pow(self, x):
        return x * x


# 586. 对x开根II
# [2020年10月24日 2021年03月17日 2021年8月13日]
# https://www.lintcode.com/problem/sqrtx-ii/description
class Solution:
    def sqrt(self, x):
        if x >= 1:
            start, end = 1, x
        else:
            start, end = x, 1
        
        while end - start > 1e-10:
            mid = (start + end) / 2
            if mid * mid < x:
                start = mid
            else:
                end = mid
                
        return start


# 183. 木材加工
# [2020年10月24日 2021年03月17日 2021年8月13日]
# https://www.lintcode.com/problem/wood-cut/description
# https://www.jiuzhang.com/solution/wood-cut/#tag-lang-python
class Solution:
    def woodCut(self, L, k):
        if not L: 
            return 0

        # caution: start with 1 not 0
        start, end = 1, min(max(L), sum(L) // k)
        if start > end: 
            return 0
        
        while start + 1 < end:
            mid = (start + end) // 2
            if self.get_pieces(L, mid) >= k:
                start = mid
            else:
                end = mid
                
        if self.get_pieces(L, end) >= k: 
            return end
        if self.get_pieces(L, start) >= k: 
            return start
        
        return 0
        
    def get_pieces(self, L, length):
        return sum(l // length for l in L)


# 633. 寻找重复的数
# [2020年10月24日 2021年03月17日 2021年8月14日]
# https://www.lintcode.com/problem/find-the-duplicate-number/description
# https://www.jiuzhang.com/solution/find-the-duplicate-number/#tag-lang-python
class Solution:
    def findDuplicate(self, nums):
        start, end = 1, len(nums) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if self.smaller_than_or_equal_to(nums, mid) > mid:
                end = mid
            else:
                start = mid
                
        if self.smaller_than_or_equal_to(nums, start) > start:
            return start
            
        return end
        
    def smaller_than_or_equal_to(self, nums, target):
        count = 0
        for val in nums:
            if val <= target:
                count += 1
        return count


# 437. 书籍复印
# [2021年8月14日]
# https://www.lintcode.com/problem/copy-books/description
# https://www.jiuzhang.com/solution/copy-books/#tag-lang-python
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        if not pages: 
            return 0
            
        start, end = max(pages), sum(pages)
        while start + 1 < end:
            mid = (start + end) // 2
            if self.get_least_people(pages, mid) <= k:
                end = mid
            else:
                start = mid
                
        if self.get_least_people(pages, start) <= k:
            return start
            
        return end
        
    def get_least_people(self, pages, time_limit):
        count = 0
        time_cost = 0 
        for page in pages:
            if time_cost + page > time_limit:
                count += 1
                time_cost = 0
            time_cost += page
            
        return count + 1


# 868. 子数组的最大平均值
# https://www.lintcode.com/problem/maximum-average-subarray/description
# https://www.jiuzhang.com/solution/maximum-average-subarray/#tag-lang-python
class Solution:
    def findMaxAverage(self, nums, k):
        max_k_sum = running_k_sum = sum(nums[:k])
        for i in range(1, len(nums)):
            if i + k <= len(nums):
                running_k_sum += nums[i+k-1] - nums[i-1]
                max_k_sum = max(max_k_sum, running_k_sum)
                
        return max_k_sum / k


""" Sweep-Line """
# 391. 数飞机
# [2021年03月18日 2021年4月13日 2021年8月14日]
# https://www.lintcode.com/problem/number-of-airplanes-in-the-sky/description
class Solution:
    def countOfAirplanes(self, airplanes):
        tasks = []
        for i in airplanes:
            tasks.append((i.start, 1))
            tasks.append((i.end,  -1))
        tasks = sorted(tasks)

        tmp, ans = 0, 0
        for idx, cost in tasks:
            tmp += cost
            ans = max(ans, tmp)
        return ans


# 821. 时间交集
# [2021年03月18日 2021年4月13日 2021年8月14日]
# https://www.lintcode.com/problem/time-intersection/description?_from=ladder&&fromId=4
# https://www.jiuzhang.com/solution/time-intersection/#tag-lang-python
# version 扫描线
class Solution:
    def timeIntersection(self, a, b):
        events = []
        for interval in a + b:
            events.append((interval.start, True))
            events.append((interval.end, False))
        
        events.sort(key = lambda interval : (interval[0], interval[1]))
        
        output = []
        num_online, online_tick = 0, 0
        for tick, online in events:
            if online:
                num_online += 1 
                online_tick = tick 
            else:
                if num_online == 2:
                    output.append(Interval(online_tick, tick))
                num_online -= 1 
        
        return output
# version 二分法
class Solution:
    def timeIntersection(self, A, B):
        m, n, output = len(A), len(B), []
        i, j = 0, 0
        while i < m and j < n:
            a, b = A[i], B[j]
            
            if min(a.end, b.end) >= max(a.start, b.start):
                output.append(Interval(max(a.start, b.start), min(a.end, b.end)))
            
            if a.end <= b.end:
                i += 1
            if a.end >= b.end:
                j += 1
        
        return output


# 131. 大楼轮廓 # TODO
# [2021年8月14日]
# https://www.lintcode.com/problem/the-skyline-problem/description
# https://www.jiuzhang.com/solution/the-skyline-problem/#tag-lang-python
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
        if item not in self.hash: return
            
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
            parent = (index - 1) // 2
            if self._smaller(self.heap[parent], self.heap[index]):
                break
            self._swap(parent, index)
            index = parent
    
    def _sift_down(self, index):
        if index is None:
            return
        while index * 2 + 1 < self.size:
            smallest = index
            left = index * 2 + 1
            right = index * 2 + 2
            
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
        
class Solution:
    """
    @param buildings: A list of lists of integers
    @return: Find the outline of those buildings
    """
    def buildingOutline(self, buildings):
        points = []
        for index, (start, end, height) in enumerate(buildings):
            points.append((start, height, index, True))
            points.append((end, height, index, False))
        points = sorted(points)
        
        maxheap = HashHeap(desc=True)
        intervals = []
        last_position = None
        for position, height, index, is_start in points:
            max_height = maxheap.top()[0] if maxheap.size else 0
            self.merge_to(intervals, last_position, position, max_height)
            
            if is_start:
                maxheap.push((height, index))
            else:
                maxheap.remove((height, index))
            last_position = position

        return intervals
        
    def merge_to(self, intervals, start, end, height):
        if start is None or height == 0 or start == end:
            return
        
        if not intervals:
            intervals.append([start, end, height])
            return
        
        _, prev_end, prev_height = intervals[-1]
        if prev_height == height and prev_end == start:
            intervals[-1][1] = end
            return
        
        intervals.append([start, end, height])


# 850. 员工空闲时间
# [2021年03月25日]
# https://www.jiuzhang.com/solution/employee-free-time/
import heapq
class Solution:
    def employeeFreeTime(self, schedule):
        heap, res = [], []
        for item in schedule:
            for i in range(0, len(item), 2):
                heapq.heappush(heap, (item[i], 0))
                heapq.heappush(heap, (item[i+1], 1))
        
        count, n = 0, len(heap)
        while n > 1:
            left = heapq.heappop(heap)
            right = heap[0]

            if left[1] == 0:
                count += 1
            else:
                count -= 1
            
            if left[1] == 1 and right[1] ==0 and count == 0:
                res.append( Interval(left[0], right[0]) )
            
            n = len(heap)
        
        return res


# 我的日程安排表 I
# https://www.jiuzhang.com/solution/my-calendar-i/
class MyCalendar:
    def __init__(self):
        self.events = []        

    def book(self, start, end):
        s, e = (start, 1), (end, -1)
        self.events.append(s)
        self.events.append(e)
        self.events.sort()

        num_of_events, max_concurrent_event = 0, -1
        for _, delta in self.events:
            num_of_events += delta
            max_concurrent_event = max(max_concurrent_event, num_of_events)

            if max_concurrent_event > 1:
                self.events.remove(s)
                self.events.remove(e)

                return False
        
        return True


""" Deque """
# 362. 滑动窗口的最大值
# [2021年3月19日 2021年8月14日]
# https://www.lintcode.com/problem/sliding-window-maximum/description
# DESC 单调队列
from collections import deque
class Solution:
    """
    @param: nums: A list of integers
    @param: k: An integer
    @return: The maximum number inside the window at each moving
    """
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


"""exercise""" # TODO

# 414. 两个整数相除
# https://www.lintcode.com/problem/divide-two-integers/description
# https://www.jiuzhang.com/solution/divide-two-integers/#tag-lang-python
class Solution(object):
    def divide(self, dividend, divisor):
        INT_MAX = 2147483647
        if divisor == 0:
            return INT_MAX
        neg = dividend > 0 and divisor < 0 or dividend < 0 and divisor > 0
        a, b = abs(dividend), abs(divisor)
        ans, shift = 0, 31
        while shift >= 0:
            if a >= b << shift:
                a -= b << shift
                ans += 1 << shift
            shift -= 1
        if neg:
            ans = - ans
        if ans > INT_MAX:
            return INT_MAX
        return ans


# 919. 会议室 II
# [2021年4月13日]
# https://www.lintcode.com/problem/meeting-rooms-ii/description
# https://www.jiuzhang.com/solution/meeting-rooms-ii/#tag-lang-python
class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    def minMeetingRooms(self, intervals):
        points = []
        for interval in intervals:
            points.append((interval.start, 1))
            points.append((interval.end, -1))
        points.sort()

        meeting_rooms, ongoing_meetings = 0,  0
        for _, delta in sorted(points):
            ongoing_meetings += delta
            meeting_rooms = max(meeting_rooms, ongoing_meetings)
            
        return meeting_rooms


# 438. 书籍复印 II
# https://www.lintcode.com/problem/copy-books-ii/description
# https://www.jiuzhang.com/solution/copy-books-ii/#tag-lang-python
class Solution:
    """
    @param n: An integer
    @param times: an array of integers
    @return: an integer
    """
    def copyBooksII(self, n, times):
        l = 1 
        r = min(times) * n 
        while l < r:
            mid = (l + r) // 2
            if self.ok(n, times, mid):
                r = mid 
            else:
                l = mid + 1 
        return l 
    
    def ok(self, n, times, tm):
        num = 0
        for i in times:
            num += tm // i
        return n <= num

class Solution:
    # @param n: an integer
    # @param times: a list of integers
    # @return: an integer
    def copyBooksII(self, n, times):
        # write your code here
        k = len(times)
        ans = 0
        eachTime = []
        totalTime = []
        for i in range(k):
            self.heapAdd(eachTime, totalTime, times[i], 0)
        for i in range(n):
            ans = totalTime[0]
            x = eachTime[0]
            self.heapDelete(eachTime, totalTime)
            self.heapAdd(eachTime, totalTime, x, ans+x)
        ans = 0
        for i in range(len(totalTime)):
            ans = max(ans, totalTime[i])
        return ans

    def heapAdd(self, eachTime, totalTime, et, tt):
        eachTime.append(et)
        totalTime.append(tt)
        n = len(eachTime)-1
        while n > 0 and eachTime[n]+totalTime[n] < eachTime[(n-1)//2]+totalTime[(n-1)//2]:
            eachTime[n], eachTime[(n-1)//2] = eachTime[(n-1)//2], eachTime[n]
            totalTime[n], totalTime[(
                n-1)//2] = totalTime[(n-1)//2], totalTime[n]
            n = (n-1)//2

    def heapDelete(self, eachTime, totalTime):
        n = len(eachTime)-1
        if n >= 0:
            eachTime[0] = eachTime[n]
        if n >= 0:
            totalTime[0] = totalTime[n]
        if len(eachTime) > 0:
            eachTime.pop()
        if len(totalTime) > 0:
            totalTime.pop()
        n = 0
        while n*2+1 < len(eachTime):
            t = n*2+1
            if t+1 < len(eachTime) and eachTime[t+1]+totalTime[t+1] < eachTime[t]+totalTime[t]:
                t += 1
            if eachTime[n]+totalTime[n] <= eachTime[t]+totalTime[t]:
                break
            eachTime[n], eachTime[t] = eachTime[t], eachTime[n]
            totalTime[n], totalTime[t] = totalTime[t], totalTime[n]
            n = t


# 617. 子数组的最大平均值 II
# https://www.lintcode.com/problem/maximum-average-subarray-ii/description
# https://www.jiuzhang.com/solution/maximum-average-subarray-ii/#tag-lang-python
class Solution:
    """
    @param: nums: an array with positive and negative numbers
    @param: k: an integer
    @return: the maximum average
    """
    def maxAverage(self, nums, k):
        if not nums:
            return 0
            
        start, end = min(nums), max(nums)
        while end - start > 1e-5:
            mid = (start + end) / 2
            if self.check_subarray(nums, k, mid):
                start = mid
            else:
                end = mid
                
        return start
        
    def check_subarray(self, nums, k, average):
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num - average)
            
        min_prefix_sum = 0
        for i in range(k, len(nums) + 1):
            if prefix_sum[i] - min_prefix_sum >= 0:
                return True
            min_prefix_sum = min(min_prefix_sum, prefix_sum[i - k + 1])
            
        return False


# 156 · 合并区间
# [2021年4月12日]
# https://www.lintcode.com/problem/156/
class Solution:
    def merge(self, intervals):
        intervals = sorted(intervals, key=lambda x: x.start)
        result = []

        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
                
        return result


# 833 · 进程序列
# [2021年4月12日]
# https://www.lintcode.com/problem/833/
class Solution:
    def numberOfProcesses(self, logs, queries):
        events = sorted([(log.start, 1) for log in logs] + 
                        [(log.end, -1) for log in logs] +
                        [(q, 0) for q in queries], 
                        key = lambda e: (e[0], -abs(e[1]), e[1]))
        
        running, counts = 0, {}
        for t, delta in events:
            running += delta
            if delta == 0:
                counts[t] = running
        
        return [counts[q] for q in queries]


# 30 · 插入区间
# [2021年4月12日]
# https://www.lintcode.com/problem/30/
class Solution:
    """
    Insert a new interval into a sorted non-overlapping interval list.
    @param intevals: Sorted non-overlapping interval list
    @param newInterval: The new interval.
    @return: A new sorted non-overlapping interval list with the new interval.
    """
    def insert(self, intervals, newInterval):
        res = []
        insertPos = 0
        
        # 定位到区间集与待插入的区间开始重合的部分，然后开始求交集。 交集一直延伸到相交区间的最末端
        for interval in intervals:
            if interval.end < newInterval.start:
                res.append(interval)
                insertPos += 1
            elif interval.start > newInterval.end:
                res.append(interval)
            else:
                newInterval.start = min(interval.start, newInterval.start)
                newInterval.end   = max(interval.end, newInterval.end)
        
        res.insert(insertPos, newInterval)
    
        return res


# 903 · 范围加法
# [2021年4月13日]
# https://www.lintcode.com/problem/903/
class Solution:
    """
    @param length: the length of the array
    @param updates: update operations
    @return: the modified array after all k operations were executed
    """
    def getModifiedArray(self, length, updates):
        res = [0 for i in range(length)]
        operation = res + [0]

        for start, end, val in updates:
            operation[start] += val
            operation[end+1] -= val

        for idx in range(length):
            if idx == 0:
                res[idx] = operation[idx]
                continue
            res[idx] = res[idx-1] + operation[idx]

        return res 

