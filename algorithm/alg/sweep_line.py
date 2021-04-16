"""
* 扫描问题的思路
    1. 事件往往是以区间的形式存在
    2. 区间两端代表事件的开始和结束
    3. 需要排序
"""
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


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


# 30 · 插入区间
# [2021年4月12日]
# https://www.lintcode.com/problem/30/
# DESC 给出一个无重叠的按照区间起始端点排序的区间列表。
# DESC 在列表中插入一个新的区间，你要确保列表中的区间仍然有序且不重叠
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


# 391. 数飞机
# [2021年03月18日 2021年4月13日]
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
# [2021年03月18日 2021年4月13日]
# https://www.lintcode.com/problem/time-intersection/description?_from=ladder&&fromId=4
# https://www.jiuzhang.com/solution/time-intersection/#tag-lang-python
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


# 903 · 范围加法
# [2021年4月13日]
# https://www.lintcode.com/problem/903/
# desc 每个更新操作表示为一个三元组：[startIndex, endIndex, inc]
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


# 919. 会议室 II
# [2021年4月13日]
# https://www.lintcode.com/problem/meeting-rooms-ii/description
# https://www.jiuzhang.com/solution/meeting-rooms-ii/#tag-lang-python
# DESC 给定一系列的会议时间间隔intervals，找到所需的最小的会议室数量
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


# 1064 · 我的日程表 II
# https://www.lintcode.com/problem/1064/
import bisect
class MyCalendarTwo(object):
  def __init__(self):
    self.attr = []
    self.count = []

  def book(self, start, end):
    start_idx = bisect.bisect_right(self.attr, start)
    self.attr.insert(start_idx, start)
    self.count.insert(start_idx, 1)

    end_idx = bisect.bisect_left(self.attr, end)
    self.attr.insert(end_idx, end)
    self.count.insert(end_idx, -1)
    
    res, count = True, 0
    for c in self.count:
      count += c
      if count >= 3:
        res = False
        break
    
    if not res:
      self.attr.pop(end_idx)
      self.count.pop(end_idx)
      self.attr.pop(start_idx)
      self.count.pop(start_idx)
    
    return res


# 1241. 寻找右区间 · Find Right Interval
# https://www.lintcode.com/problem/find-right-interval/
# https://www.jiuzhang.com/solution/find-right-interval/
class Solution:
    """
    @param intervals: a list of intervals
    @return: return a list of integers
    """
    def findRightInterval(self, intervals):
        r = [-1] * len(intervals)
        intervals = [(intervals[i], i) for i in range(len(intervals))]
        intervals.sort(key = lambda x:x[0].start)
        
        for i in range(len(intervals)):
            u = intervals[i][0]
            lo,hi = i + 1, len(intervals) - 1
            while(lo < hi):
                med = (lo + hi) >> 1
                if intervals[med][0].start >= u.end:
                    hi = med
                else:
                    lo = med + 1
            r[intervals[i][1]] = intervals[lo][1] if lo < len(intervals) and intervals[lo][0].start >= u.end else -1

        return r

class Solution:
    def findRightInterval(self, intervals):
        length = len(intervals)
        res = [-1] * length
        intervals = [(intervals[i],i) for i in range(length)]
        intervals.sort(key=lambda x: x[0].start)

        for i in range(length):
            cur = intervals[i][0]
            start, end = 0, length - 1
            while start + 1 < end:
                mid = (start + end) >> 1
                if intervals[mid][0].start >= cur.end:
                    end = mid
                else:
                    start = mid
            
            tmp = start if intervals[start][0].start >= cur.end else end

            res[intervals[i][1]] = intervals[tmp][1] if end < length and intervals[tmp][0].start >= cur.end else -1

        return res


# 1291 · 运动会
# https://www.lintcode.com/problem/1291/
class Solution:
    def CheerAll(self, Events):
        order_events = []
        length, now = len(Events), 0

        for i in range(length):
            leave = Events[i][1] - (Events[i][1]-Events[i][0]) // 2 - 1
            order_events.append( [leave, i] )
        order_events.sort()

        for i in range(length):
            leave, index = order_events[i]
            start, end = Events[index]
            now = int(max(now, start))
            if now > leave:
                return -1 
            now += (end-start)//2 + 1
        
        return 1


# 1397 · 覆盖数字
# https://www.lintcode.com/problem/1397/
class Solution:
    """
    @param intervals: The intervals
    @return: The answer
    """
    def digitalCoverage(self, intervals):
        events = collections.Counter()
        for i in intervals:
            events[i.start] += 1
            events[i.end+1] -= 1
        
        cover = 0
        for i in sorted(events):
            cover += events.pop(i)
            events[i] = cover
        
        return max(events, key=events.get)


# 1399 · 拿硬币
# https://www.lintcode.com/problem/1399/
class Solution:
    def takeCoins(self, nums, k):
        n = len(nums)
        lo, hi = 0, k-1
        
        max_coins = running_coins = sum(nums[lo:hi+1])
        while hi != -1:
            running_coins -= nums[hi]
            lo -= 1 
            hi -= 1 
            running_coins += nums[lo]
            
            max_coins = max(max_coins, running_coins)
            
        return max_coins
        

# 1897 · 会议室 3
# https://www.lintcode.com/problem/1897/
class Solution:
    """
    @param intervals: the intervals
    @param rooms: the sum of rooms
    @param ask: the ask
    @return: true or false of each meeting
    """
    def meetingRoomIII(self, intervals, rooms, asks):
        time = [0] * 500001
        max_num = 0
        for interval in intervals:
            time[interval[0]] += 1
            time[interval[1]] -= 1
            max_num = max(max_num, time[interval[1]])

        for i in range(0, len(asks)):
            max_num = max(max_num, asks[i][1])

        last = time[0]
        available = [0] * (max_num+1)
        available[0] = 1 if last < rooms else 0
        for i in range(1, len(available)):
            curr = last + time[i]
            if curr < rooms:
                available[i] = available[i - 1] + 1
            else:
                available[i] = available[i - 1]
            last = curr

        results = []
        for ask in asks:
            result = available[ask[1] - 1] - available[ask[0] - 1] >= ask[1] - ask[0]
            results.append(result)
        return results


# 850 · 员工空闲时间
# https://www.lintcode.com/problem/850/
from heapq import heappush, heappop
class Solution:
    def employeeFreeTime(self, schedule):
        heap, res = [], []
        for item in schedule:
            for i in range(0, len(item), 2):
                heappush(heap, (item[i], 1))
                heappush(heap, (item[i+1], 0))
        
        count, n = 0, len(heap)
        while n > 1:
            left = heappop(heap)
            right = heap[0]

            if left[1]:
                count += 1
            else:
                count -= 1
            
            if left[1] == 0 and right[1] == 1 and count == 0:
                res.append( Interval(left[0], right[0]) )
            
            n = len(heap)
        
        return res


# 1063 · 我的日历III
# https://www.lintcode.com/problem/1063/
import collections
class MyCalendarThree:
    def __init__(self):
        self.mapper = {}

    def book(self, start, end):
        if start not in self.mapper: self.mapper[start] = 0
        if end not in self.mapper: self.mapper[end] = 0

        self.mapper[start] += 1
        self.mapper[end] -= 1

        s = sorted(self.mapper.items(), key=lambda x:x[0])
        now, max_val = 0, 0
        for key, val in s:
            max_val = max(max_val, now)
            now += val

        return max_val 


# 1379 · 最长场景
# https://www.lintcode.com/problem/1379/
class Solution:
    def getLongestScene(self, str):
        # Write your code here
        if not str:
            return 0
        
        pos = {}
        
        for i in range(len(str)):
            char = str[i]
            if char not in pos:
                # [left_pos, right_pos]
                pos[char] = [i, i]
            else:
                pos[char][1] = i
        
        intervals = sorted(pos.values(), key = lambda x: x[0])
        stack = []
        max_l = 0
        
        for left, right in intervals:
            if not stack or stack[-1][1] < left:
                stack.append([left, right])
            else:
                stack[-1][1] = max(stack[-1][1], right)
            
            max_l = max(max_l, stack[-1][1] - stack[-1][0] + 1)
        
        return max_l


# 1450 · 矩形面积 II
# https://www.lintcode.com/problem/1450/


# 131 · 大楼轮廓
# https://www.lintcode.com/problem/131/
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
            points.append( (start, height, index, True) )
            points.append( (end,   height, index, False) )
        points = sorted( points )

        maxHeap = HashHeap(desc=True)
        intervals = []
        last_position = None

        for position, height, index, is_start in points:
            max_height = maxHeap.top()[0] if maxHeap.size > 0 else 0
            self.merge_to( intervals, last_position, position, max_height )
            if is_start:
                maxHeap.push((height, index))
            else:
                maxHeap.remove((height, index))
            last_position = position
        
        return intervals

    def merge_to(self, intervals, start, end, height):
        if start is None or height ==0 or start == end: 
            return 

        if not intervals:
            intervals.append( [start, end, height] )
            return
        
        _, prev_end, prev_height = intervals[-1]
        if prev_height == height and prev_end == start:
            intervals[-1][1] = end

            return 
        
        intervals.append([start, end, height])









