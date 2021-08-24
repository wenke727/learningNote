import sys

# 135/39. 数字组合
# [2020年11月5日 2021年8月7日  2021年8月23日]
# https://www.lintcode.com/problem/combination-sum/description
# https://www.jiuzhang.com/solutions/combination-sum/#tag-lang-python
class Solution:
    def combinationSum(self, candidates, target):
        if candidates is None:
            return []
        
        candidates.sort()
        result = []

        self.dfs(candidates, target, 0, [], result)

        return result


    def dfs(self, nums, target, index, combination, res):
        if target == 0:
            res.append(combination[:])
            return
        
        for i in range(index, len(nums)):
            if nums[i] > target:
                continue
            
            if i > 0 and nums[i] == nums[i-1]:
                continue

            combination.append(nums[i])
            self.dfs(nums, target-nums[i], i, combination, res)
            combination.pop()


# 153/40. 数字组合 II
# [2020年11月5日 2021年8月7日]
# https://www.lintcode.com/problem/combination-sum-ii/description
# https://www.jiuzhang.com/solution/combination-sum-ii/#tag-lang-python
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, candidates, target):
        candidates.sort()
        results = [] 
        self.dfs( candidates, target, 0, [], [0] * len(candidates), results  )

        return results

    def dfs(self, nums, target, index, combination, use, results):
        if target == 0:
            results.append(combination[:])
            return 
        
        for i in range(index, len(nums)):
            if target < nums[i]:
                continue
            
            if i == 0 or nums[i] != nums[i-1] or use[i-1]==1:
                combination.append(nums[i])
                use[i] = 1
                self.dfs(nums, target- nums[i], i+1, combination, use, results)
                use[i] = 0
                combination.pop()


# 90. k数和 II
# [2020年11月5日 2021年8月7日]
# https://www.lintcode.com/problem/k-sum-ii/description
# https://www.jiuzhang.com/solution/k-sum-ii/#tag-lang-python
class Solution:
    """
    @param: A: an integer array
    @param: k: a postive integer <= length(A)
    @param: targer: an integer
    @return: A list of lists of integer
    """
    def kSumII(self, A, k, target):
        A.sort()
        results = []
        self.dfs( A, k, target, 0, [], results )
        
        return results

    def dfs(self, nums, k, target, index, combination, reults):
        if k == 0 and target == 0:
            reults.append( combination[:] )
            return
        
        if k == 0 or target <= 0:
            return
        
        for i in range(index, len(nums)):
            combination.append( nums[i] )
            # 不允许重复，因此`i+1`
            self.dfs(nums, k-1, target- nums[i], i+1, combination, reults)
            combination.pop()       


# 680. 分割字符串
# [2020年11月5日 2021年8月7日]
# https://www.lintcode.com/problem/split-string/description
# https://www.jiuzhang.com/solution/split-string/#tag-lang-python
class Solution:
    def splitString(self, s):
        result = []
        self.dfs( s, 0, [], result )

        return result

    def dfs(self, s, start, combination, result):
        if start >= len(s):
            result.append( combination[:] )
            return
        
        for i in range(2):
            if i + start >= len(s):
                continue

            combination.append( s[start: start+i+1] )
            self.dfs( s, start+i+1, combination, result )
            combination.pop()


# 136. 分割回文串
# [2020年11月5日 2021年8月7日]
# https://www.lintcode.com/problem/palindrome-partitioning/description
# https://www.jiuzhang.com/solutions/palindrome-partitioning/#tag-lang-python
class Solution:
    def partition(self, s):
        result = []
        self.dfs(s, [], result)

        return result

    def dfs( self, s, combination, result):
        if len(s) == 0:
            result.append( combination[:] )
            return
        
        for i in range(1, len(s)+1):
            prefix = s[:i]

            if not self.is_palindrome(prefix):
                continue
            
            combination.append( prefix )
            self.dfs( s[i:], combination, result )
            combination.pop()

    def is_palindrome(self, s):
        return s == s[::-1]


""" Memoization Search 记忆化搜索 """
# 192/44. 通配符匹配 ⭐
# [2020年11月5日, 2020年11月12日 2021年8月7日]
# https://www.lintcode.com/problem/wildcard-matching/description
# https://www.jiuzhang.com/solution/wildcard-matching/#tag-lang-python
#version dfs
class Solution:
    def isMatch(self, source, pattern):
        return self.dfs(source, 0, pattern, 0, {})
    
    def dfs(self, s, i, p, j, memo):
        if (i, j) in memo:
            return memo[(i,j)]

        if len(s) == i:
            return self.is_empty(p, j)
        if len(p) == j:
            return False
        
        if p[j] != '*':
            matched = self.is_match_char(s[i], p[j]) and self.dfs(s, i+1, p, j+1, memo)
        else:
            matched = self.dfs(s, i+1, p, j, memo) or self.dfs(s, i, p, j+1, memo)

        memo[(i,j)] = matched

        return matched

    def is_match_char(self, s, p):
        return s == p or p == '?'
    
    def is_empty(self, p, j):
        for index in range(j, len(p)):
            if p[index] != "*":
                return False

        return True
#version dp
class Solution:
    """
    @param s: A string 
    @param p: A string includes "?" and "*"
    @return: is Match?
    """
    def isMatch(self, s, p):
        if s is None or p is None: return False

        n, m = len(s), len(p)
        dp = [  [False] * (m+1) for _ in range(n+1) ]

        # initilization
        dp[0][0] = True
        for i in range(1, m+1):
            dp[0][i] = dp[0][i-1] and p[i-1] == "*"
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                if p[j-1] == '*':
                    # ! `*` 参与匹配 / 不参与匹配
                    dp[i][j] = dp[i-1][j] or dp[i][j-1] 
                else:
                    dp[i][j] = dp[i-1][j-1] and ( s[i-1] == p[j-1] or p[j-1] == "?" )
        
        return dp[n][m]


# 154/10. 正则表达式匹配 ⭐
# [2020年11月5日, 2020年11月12日 2021年8月7日]
# https://www.lintcode.com/problem/regular-expression-matching/description
# https://www.jiuzhang.com/solution/regular-expression-matching/#tag-lang-python
class Solution:
    """
    @param s: A string 
    @param p: A string includes "." and "*"
    @return: A boolean
    """
    def isMatch(self, source, pattern):
        return self.dp(source, 0, pattern, 0, {})
    
    def dp(self, s, i, p, j, memo):
        if (i,j) in memo:
            return memo[(i,j)]

        if len(s) == i:
            return self.is_empty(p[j:])
        if len(p) == j:
            return False

        if j+1 < len(p) and p[j+1] == '*':
            # case0: match more than one char; case 1: mathced zero char
            case0 = self.is_match_char(s[i], p[j]) and self.dp(s, i+1, p, j, memo)
            case1 = self.dp(s, i, p, j+2, memo)
            matched = case0 or case1
        else:
            matched = self.is_match_char(s[i], p[j]) and self.dp(s, i+1, p, j+1, memo)
        
        memo[(i,j)] = matched

        return matched

    def is_match_char( self, s, p ):
        return s == p or p == '.'

    def is_empty(self, pattern):
        if len(pattern) % 2 == 1: 
            return False

        for i in range(1, len(pattern), 2):
            if pattern[i] != '*':
                return False
        
        return True  


# 582/140. 单词拆分II
# [2020年11月5日, 2020年11月12日 2021年8月7日]
# https://www.lintcode.com/problem/word-break-ii/description
# https://www.jiuzhang.com/solution/word-break-ii/#tag-lang-python
class Solution:
    """
    @param: s: A string
    @param: wordDict: A set of words.
    @return: All possible sentences.
    """
    def wordBreak(self, s, wordDict):
        return self.dfs(s, wordDict, {})
    
    def dfs( self, s, wordDict, memo):
        if s in memo: 
            return memo[s]

        if len(s) == 0: 
            return []
        
        partitions= []
        for i in range( 1, len(s)+1 ):
            prefix = s[:i]
            if prefix not in wordDict:
                continue

            sub_partitions = self.dfs( s[i:], wordDict, memo )
            for item in sub_partitions:
                partitions.append( prefix + ' ' + item )
        
        # caution
        if s in wordDict:
            partitions.append(s)
            
        memo[s] = partitions

        return partitions


# 683. Word Break III ⭐⭐
# [2020年11月12日 2021年8月7日]
# https://www.lintcode.com/problem/word-break-iii/description
# 给出一个单词表和一条去掉所有空格的句子，根据给出的单词表添加空格, 返回可以构成的句子的数量, 保证构成的句子中所有的单词都可以在单词表中找到.
class Solution:
    """
    @param: : A string
    @param: : A set of word
    @return: the number of possible sentences.
    """
    def wordBreak3(self, s, dict):
        max_length, lower_dict = self.initialize(dict)

        return self.memo_search(s.lower(), 0, max_length, lower_dict, {})
        
    def memo_search(self, s, index, max_length, lower_dict, memo):
        if index == len(s):
            return 1
        
        if index in memo:
            return memo[index]
        
        memo[index] = 0
        for i in range(index, len(s)):
            if i + 1 - index > max_length:
                break

            word = s[index: i+1]
            if word not in lower_dict:
                continue
            
            memo[index] += self.memo_search(s, i + 1, max_length, lower_dict, memo)
            
        return memo[index]
        
    def initialize(self, dict):
        max_length = 0
        lower_dict = set()
        for word in dict:
            max_length = max(max_length, len(word))
            lower_dict.add(word.lower())
        
        return max_length, lower_dict


# 272. Climbing Stairs II
# https://www.lintcode.com/problem/climbing-stairs-ii/description
# https://www.jiuzhang.com/solution/climbing-stairs-ii/#tag-lang-python
class Solution:
    """
    @param n: An integer
    @return: An Integer
    """
    def climbStairs2(self, n):
        if n <= 1:
            return 1
        if n == 2:
            return 2
        a, b, c = 1, 1, 2
        for i in range(3, n + 1):
            a, b, c = b, c, a + b + c
        return c


"""DP 动态规划"""
# 76/300. Longest Increasing Subsequence
# [2020年11月13日 2021年1月2日]
# https://www.lintcode.com/problem/longest-increasing-subsequence/description
# https://www.jiuzhang.com/solution/longest-increasing-subsequence/#tag-lang-python
# DESC 给定一个整数序列，找到最长上升子序列（LIS），返回LIS的长度
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


# 602. 俄罗斯套娃信封
# [2021年1月2日]
# https://www.lintcode.com/problem/russian-doll-envelopes/description
# https://www.jiuzhang.com/solution/russian-doll-envelopes/#tag-lang-python
class Solution:
    """
    @param: envelopes: a number of envelopes with widths and heights
    @return: the maximum number of envelopes
    """
    def maxEnvelopes(self, envelopes):
        if not envelopes:
            return 0
        
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        
        n = len(envelopes)
        lis = [float('inf')] * (n + 1)
        lis[0] = -float('inf')
        
        longest = 0
        for (_, h) in envelopes:
            index = self.first_gte(lis, h)
            lis[index] = h
            longest = max(longest, index)
            
        return longest
    
    def first_gte(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid
        if nums[start] >= target:
            return start
        return end
# version: myself
class Solution:
    def maxEnvelopes(self, envelopes):
        if not envelopes: return 0
        
        envelopes.sort( key = lambda x: (x[0], -x[1]))
        
        res = []
        for (_, w) in envelopes:
            if len(res) == 0 or w > res[-1]:
                res.append( w )
                continue
            
            insert_pos = self.binary_search_left( res, w )
            res[insert_pos] = w
        
        return len(res)
    
    def binary_search_left(self, nums, target):
        start, end = 0, len(nums)-1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid
        
        # ! notice
        if target <= nums[start]: return start
        if target <= nums[end]: return end
        
        return len(nums)


# 603/368. Largest Divisible Subset
# [2020年11月13日]
# https://www.lintcode.com/problem/largest-divisible-subset/description
# https://www.jiuzhang.com/solution/largest-divisible-subset/#tag-lang-python
# version 1
class Solution:
    """
    @param nums: a set of distinct positive integers
    @return: the largest subset A
    """
    def largestDivisibleSubset(self, nums):
        if not nums: return []
            
        nums = sorted(nums)

        dp, prev = {}, {}
        for num in nums:
            dp[num] = 1
            prev[num] = -1
        
        last_num = nums[0]
        for num in nums:
            for factor in self.get_factors(num):
                if factor not in dp:
                    continue

                if dp[num] < dp[factor] + 1:
                    dp[num] = dp[factor] + 1
                    prev[num] = factor
            
            if dp[num] > dp[last_num]:
                last_num = num
        
        return self.get_path(prev, last_num)

    def get_factors(self, num):
        # 不是 for 循环所有比他小的数，而是直接 for 循环他的因子
        if num == 1: return []

        factor, res = 1, []
        while factor * factor <= num:
            if num % factor == 0:
                res.append(factor)

                if factor * factor != num and factor != 1:
                    res.append(num // factor)
            factor += 1
        return res
    
    def get_path(self, prev, last_num):
        path = []
        while last_num != -1:
            path.append(last_num)
            last_num = prev[last_num]
        return path[::-1]
# version 2
class Solution:
    # @param {int[]} nums a set of distinct positive integers
    # @return {int[]} the largest subset 
    def largestDivisibleSubset(self, nums):
        # Write your code here
        n = len(nums)
        dp = [1] * n
        father = [-1] * n

        nums.sort()
        m, index = 0, -1
        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if 1 + dp[j] > dp[i]:
                        dp[i] = dp[j] + 1
                        father[i] = j

            if dp[i] >= m:
                m = dp[i]
                index = i

        result = []
        for i in range(m):
            result.append(nums[index])
            index = father[index]

        return result


# 109. Triangle
# [2020年11月13日]
# https://www.lintcode.com/problem/triangle/description
# https://www.jiuzhang.com/solution/triangle/#tag-lang-python
# version DP + DFS
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        return self.divide_conquer(triangle, 0, 0, {})
       
    def divide_conquer(self, triangle, x, y, memo):
        if x == len(triangle):
            return 0
            
        if (x, y) in memo:
            return memo[(x, y)]

        left = self.divide_conquer(triangle, x + 1, y, memo)
        right = self.divide_conquer(triangle, x + 1, y + 1, memo)
        
        memo[(x, y)] = min(left, right) + triangle[x][y]
        return memo[(x, y)]
# version 自顶向下, 非滚动数组
class Solution:
    def minimumTotal(self, triangle):
        n = len(triangle)
        
        dp = [[0] * n for _ in range(n)]
        
        dp[0][0] = triangle[0][0]
        for i in range(1, n):
            dp[i][0] = dp[i-1][0] + triangle[i][0]
            dp[i][i] = dp[i-1][i-1] + triangle[i][i]
   
        for i in range(1, n):
            for j in range(1, i):
                dp[i][j] = min( dp[i-1][j-1], dp[i-1][j] ) + triangle[i][j]

        return min( dp[n-1] )
# version 自顶向下
class Solution:
    def minimumTotal(self, triangle):
        n = len(triangle)
        
        # state: dp[i][j] 代表从 0, 0 走到 i, j 的最短路径值
        dp = [[0] * n, [0] * n]
        
        dp[0][0] = triangle[0][0]
        
        # function: dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
        for i in range(1, n):
            dp[i % 2][0] = dp[(i - 1) % 2][0] + triangle[i][0]
            dp[i % 2][i] = dp[(i - 1) % 2][i - 1] + triangle[i][i]
            for j in range(1, i):
                dp[i % 2][j] = min(dp[(i - 1) % 2][j], dp[(i - 1) % 2][j - 1]) + triangle[i][j]
               
        return min(dp[(n - 1) % 2])
# version 自底向上的动态规划
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        n = len(triangle)
        
        # state: dp[i][j] 代表从 i,j  走到最底层的最短路径值
        dp = [[0] * n, [0] * n]
        
        # initialize: 初始化终点（最后一层）
        for i in range(n):
            dp[(n - 1) % 2][i] = triangle[n - 1][i]
            
        # dp[i][j] = min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle[i][j]
        for i in range(n - 2, -1, -1):
            for j in range(i + 1):
                dp[i % 2][j] = min(dp[(i + 1) % 2][j], dp[(i + 1) % 2][j + 1]) + triangle[i][j]
                
        return dp[0][0]


# 630. Knight Shortest Path II
# [2020年11月13日]
# https://www.lintcode.com/problem/knight-shortest-path-ii/description
# https://www.jiuzhang.com/solution/knight-shortest-path-ii/#tag-lang-python
DIRECTIONS = [(-1, -2),(1, -2),(-2, -1),(2, -1),]
class Solution:
    def shortestPath2(self, grid):
        if not grid or not grid[0]: 
            return -1
        
        n, m = len(grid), len(grid[0])
        dp = [[sys.maxsize] * m for _ in range(n)]
        dp[0][0] = 0
        
        # ! i, j 遍历顺序 <- 骑士只能从左边走到右边
        for j in range(m):
            for i in range(n):
                if grid[i][j]:
                    continue
           
                for dx, dy in DIRECTIONS:
                    x, y = i + dx, j + dy
                    if not (0 <= x < n and 0 <= y < m):
                        continue

                    dp[i][j] = min(dp[i][j], dp[x][y] + 1)

        return -1 if dp[n-1][m-1] == sys.maxsize else dp[n-1][m-1] 
# version 滚动数组优化
class Solution:
    # @param {boolean[][]} grid a chessboard included 0 and 1
    # @return {int} the shortest path
    def shortestPath2(self, grid):
        if not grid or not grid[0]:
            return -1
        
        n, m = len(grid), len(grid[0])
        
        # state: dp[i][j % 3] 代表从 0,0 跳到 i,j 的最少步数
        dp = [[float('inf')] * 3 for _ in range(n)]

        # initialize: 0,0 是起点
        dp[0][0] = 0
        
        # function
        for j in range(1, m):
            for i in range(n):
                dp[i][j % 3] = float('inf')
                if grid[i][j]:
                    continue
                for delta_x, delta_y in DIRECTIONS:
                    x, y = i + delta_x, j + delta_y
                    if 0 <= x < n and 0 <= y < m:
                        dp[i][j % 3] = min(dp[i][j % 3], dp[x][y % 3] + 1)

        # answer
        if dp[n - 1][(m - 1) % 3] == float('inf'):
            return -1
        return dp[n - 1][(m - 1) % 3]

