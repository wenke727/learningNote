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


# 寻找右区间 · Find Right Interval
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
            lo,hi = i + 1,len(intervals) - 1
            while(lo < hi):
                med = (lo + hi) >> 1
                if intervals[med][0].start >= u.end:
                    hi = med
                else:
                    lo = med + 1
            r[intervals[i][1]] = intervals[lo][1] if lo < len(intervals) and intervals[lo][0].start >= u.end else -1
        return r


# 1291 · 运动会
# https://www.lintcode.com/problem/1291/



# 1397 · 覆盖数字
# https://www.lintcode.com/problem/1397/


# 1399 · 拿硬币
# https://www.lintcode.com/problem/1399/


# 1897 · 会议室 3
# https://www.lintcode.com/problem/1897/


# 850 · 员工空闲时间
# https://www.lintcode.com/problem/850/


# 1063 · 我的日历III
# https://www.lintcode.com/problem/1063/


# 1379 · 最长场景
# https://www.lintcode.com/problem/1379/


# 1450 · 矩形面积 II
# https://www.lintcode.com/problem/1450/


# 131 · 大楼轮廓
# https://www.lintcode.com/problem/131/







