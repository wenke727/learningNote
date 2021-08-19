import sys

"""
区间类Dp

共性就是求[0, n-1]这样一个区间
逆向思维分析，从大到小

逆向 => 分治
"""
# 476/877 石子归并
# [2020年11月6日, 2020年11月24日, 2021年3月31日]
# https://www.lintcode.com/problem/stone-game/description
# https://www.jiuzhang.com/solution/stone-game/#tag-lang-python
# DESC 记忆化搜索：从大到小，先考虑 0 ~ n-1 合并的总费用
# version: DFS
class Solution:
    def stoneGame(self, piles):
        return self.dfs(piles, 0, len(piles)-1, {})

    def dfs(self, piles, start, end, memo):
        if (start, end) in memo: 
            return memo[(start, end)]
        if start >= end: 
            return 0

        cost = sum(piles[start : end+1])
        mimimun = sys.maxsize

        for mid in range( start, end):
            left  = self.dfs(piles, start,   mid, memo)
            right = self.dfs(piles, mid + 1, end, memo)
            mimimun = min( mimimun, left + cost + right )
        
        memo[(start, end)] = mimimun

        return mimimun
# version 
class Solution:
    """
    @param A: An integer array
    @return: An integer
    """
    def stoneGame(self, A):
        n = len(A)
        if n < 2:
            return 0
            
        # dp[i][j] => minimum cost merge from i to j
        dp = [[0] * n for _ in range(n)]
        # range_sum[i][j] => A[i] + A[i + 1] ... + A[j]
        range_sum = self.get_range_sum(A)
            
        # ? enumerate the range size first, start point second 
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = sys.maxsize
                for mid in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid + 1][j] + range_sum[i][j])
        
        return dp[0][n - 1]
                    
    def get_range_sum(self, A):
        n = len(A)
        range_sum = [[0] * n for _ in range(len(A))]
        for i in range(n):
            range_sum[i][i] = A[i]
            for j in range(i + 1, n):
                range_sum[i][j] = range_sum[i][j-1] + A[j]
        return range_sum


# 168/312. 吹气球 ✨✨✨
# [2020年11月6日 2021年3月31日]
# https://www.lintcode.com/problem/burst-balloons/description
# https://www.jiuzhang.com/solution/burst-balloons/#tag-lang-python
# version 2
class Solution:
    def maxCoins(self, nums):
        if not nums:
            return 0
        
        nums = [1, *nums, 1]

        return self.dfs(nums, 0, len(nums)-1, {})
    
    def dfs(self, nums, start, end, memo):
        # memo[i][j] 代表搓破气球i和j之间（开区间）的所有气球，可以获得的最高分数
        if (start, end) in memo:
            return memo[(start, end)]
        
        if start == end:
            return 0
        
        best = 0
        for mid in range(start+1, end):
            left  = self.dfs(nums, start, mid, memo)
            right = self.dfs(nums, mid, end, memo)
            best = max(
                best,  
                left + right + nums[start]*nums[mid]*nums[end]
            )
        
        memo[(start, end)] = best
        return best  
# version 1
class Solution:
    def maxCoins(self, nums):
        if not nums:
            return 0
            
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        # TODO traverse type
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                for k in range(i, j + 1):
                    left = nums[i-1] if i > 0 else 1
                    right = nums[j + 1] if j < n - 1 else 1
                    dp[i][j] = max(dp[i][j], dp[i][k - 1] + dp[k + 1][j] + left * nums[k] * right)
    
        return dp[0][n - 1]
# version 3
class Solution:
    def maxCoins(self, nums):
        if not nums: return 0
            
        nums = [1, *nums, 1]
        n = len(nums)
        
        dp = [[0] * n for _ in range(n)]

        for i in range(n - 1, -1, -1):
            for j in range(i + 2, n):
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] +  dp[k][j] + nums[i] * nums[k] * nums[j])
        return dp[0][n - 1]


# 430. 攀爬字符串 
# [2021年3月31日]
# https://www.lintcode.com/problem/scramble-string/description
# https://www.jiuzhang.com/solution/scramble-string/#tag-lang-python
class Solution:
    def isScramble(self, s1, s2):
        return self.dfs(s1, s2, {})
    
    def dfs(self, s1, s2, memo):
        if (s1, s2) in memo: return memo[(s1, s2)]
        
        if len(s1) != len(s2): return False
        
        if s1 == s2: return True
        
        s1_list = list(s1)
        s2_list = list(s2)
        if s1_list.sort() != s2_list.sort():
            return False
        
        # for i in range(len(s1)): # the begin point is 1
        for i in range(1, len(s1)):
            if (self.dfs(s1[:i], s2[:i], memo) and self.dfs(s1[i:], s2[i:], memo)) or \
               (self.dfs(s1[:i], s2[-i:], memo) and self.dfs(s1[i:], s2[:-i], memo)):
                memo[(s1, s2)] = True
                return True
        
        memo[(s1, s2)] = False
        return False


# 741 · 计算最大值II
# https://www.lintcode.com/problem/741/
class Solution:
    """
    @param str: a string of numbers
    @return: the maximum value
    """
    def maxValue(self, str):
        n = len(str)
        dp = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            dp[i][i] = ord(str[i]) - ord('0')

        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                for k in range(i, j):
                    dp[i][j] = max(dp[i][j],  
                                   dp[i][k] + dp[k + 1][j],
                                   dp[i][k] * dp[k + 1][j]
                                )
        
        return dp[0][n - 1]


"""匹配类动态规划"""
# 77/1143. 最长公共子序列
# [2020年11月6日 2021年1月2日 2021年3月31日]
# https://www.lintcode.com/problem/longest-common-subsequence/description
# https://www.jiuzhang.com/solution/longest-common-subsequence/#tag-lang-python
class Solution:
    def longestCommonSubsequence(self, A, B):
        n, m = len(A), len(B)
        dp = [[0] * (m + 1), [0] * (m + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if A[i-1] == B[j-1]:
                    dp[i%2][j] = dp[(i-1)%2][j-1] + 1
                else:
                    dp[i%2][j] = max(dp[i%2][j-1], dp[(i-1) % 2][j], dp[(i-1) % 2][j-1])
        
        return dp[n % 2][m]
class Solution:
    def longestCommonSubsequence(self, A, B):
        n, m = len(A), len(B)
        dp = [[0] * (m + 1) for _ in range(n+1)]
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[(i-1)][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1], dp[(i-1)][j], dp[(i-1)][j-1])
        
        return dp[n][m]


# 119. 编辑距离 
# [2020年11月6日 2021年1月2日 2021年1月4日 2021年3月31日]
# https://www.lintcode.com/problem/edit-distance/description
# https://www.jiuzhang.com/solution/edit-distance/#tag-lang-python
class Solution:
    def minDistance(self, word1, word2):
        n, m = len(word1), len(word2)
        dp = [[0] *(m+1) for i in range(2)]

        for j in range( m+1 ):
            dp[0][j] = j
        
        for i in range(1, n+1):
            dp[i%2][0] = i
            for j in range(1, m+1):
                d, u, l = dp[(i-1)%2][j-1], dp[(i-1)%2][j], dp[i%2][j-1]
                
                if word1[i-1] == word2[j-1]:
                    dp[i%2][j] = min(d, u+1, l+1)
                else:
                    dp[i%2][j] = min(d, u, l) + 1
        
        return dp[n % 2][m]

# version: 入门
class Solution:
    def minDistance(self, word1, word2):
        n, m = len(word1), len(word2)
        
        dp = [ [0] * (m+1) for _ in range(n+1) ]
        for j in range(m+1):
            dp[0][j] = j
        for i in range(n+1):
            dp[i][0] = i
        
        for i in range(n):
            for j in range(m):
                d, u, l = dp[i][j], dp[i][j+1], dp[i+1][j]
                
                if word1[i] == word2[j]:
                    dp[i+1][j+1] = min(d, u+1, l+1)
                else:
                    dp[i+1][j+1] = min(d, u, l) + 1
        
        return dp[n][m]

# version 带打印功能  DFS + memoization version，求出所有具体的操作方式 
class Node:
    def __init__(self, value, choice):
        self.value = value
        # 0 None, 1 replace, 2 delete, 3 insert
        self.choice = choice
    
    def __str__(self):
        return str(self.value) 

class Solution:
    def minDistance(self, word1, word2):
        n, m = len(word1), len(word2)
        
        dp = [ [Node(0, -1) for i in range(m+1) ] for _ in range(n+1) ]

        for j in range(m+1):
            dp[0][j].value = j
        for i in range(n+1):
            dp[i][0].value = i
        
        for i in range(n):
            for j in range(m):
                d, u, l = dp[i][j], dp[i][j+1], dp[i+1][j]
                if word1[i] == word2[j]:
                    dp[i+1][j+1] =  Node(d.value, 0)
                else:
                    dp[i+1][j+1] = self.minNode( d, u, l )
        
        self.print_result(word1, word2, dp)
            
        return dp[n][m].value
        
    def minNode( self, d: Node, u: Node, l: Node ):
        # 0 None, 1 replace, 2 delete, 3 insert
        # ! val = min( d.value, u.value, l.value ) + 1
        val = min( d.value, u.value, l.value ) 
        if val == d.value:
            return Node(val+1, 1)
        if val == u.value:
            return Node(val+1, 2)
        return Node(val+1, 3)
        
    def print_result( self, word1, word2, dp ):
        rows, cols = len(dp), len(dp[0])
        i , j = rows-1, cols-1
        print( f"change {word1} to {word2}:" )
        
        info = []
        while i != 0 and j != 0:
            c1, c2 = word1[i-1], word2[j-1]
            choice = dp[i][j].choice
            info.append( f"word1[{i-1}]:" )
            if choice == 0:
                info.append( f'\tskip {c1}, [{i},{j}]' )
                i-=1; j-=1
            elif choice == 1:
                info.append( f'\treplace {c1} with {c2}, [{i},{j}]' )
                i-=1; j-=1
            elif choice == 2:
                info.append( f'\tdelete {c1}, [{i},{j}]' )
                i-=1
            else:
                info.append( f"\tinsert {c2}, [{i},{j}]" )
                j-=1
        
        while i > 0:
            info.append( f"word1[{i-1}]:" )
            info.append( f'\tdelete {word1[i-1]}, [{i},{j}]' )
            i-=1
            
        while j > 0:
            info.append( f"word1[0]:" )
            info.append( f'\tinsert {word2[j-1]}, [{i},{j}]' )
            j-=1    

        for i in info[::-1]:
            print(i)  


# 623. K步编辑 # TODO
# https://www.lintcode.com/problem/k-edit-distance/description
# https://www.jiuzhang.com/solution/k-edit-distance/#tag-lang-python
class TrieNode:
    def __init__(self):
        self.children = [None for i in range(26)]
        self.hasWord = False
        self.str = None
    
    @classmethod
    def addWord(cls, root, word):
        node = root
        for letter in word:
            child = node.children[ord(letter) - ord('a')]
            if child is None:
                child = TrieNode()
                node.children[ord(letter) - ord('a')] = child
            node = child
    
        node.hasWord = True
        node.str = word

class Solution:
    # @param {string[]} words a set of strings
    # @param {string} target a target string
    # @param {int} k an integer
    # @return {string[]} output all the stirngs that meet the requirements 
    def kDistance(self, words, target, k):
        root = TrieNode()
        for word in words:
            TrieNode.addWord(root, word)

        result = []
        n = len(target)
        dp = [i for i in range(n + 1)]

        self.find(root, result, k, target, dp)
        return result

    def find(self, node, result, k, target, dp):
        n = len(target)

        if node.hasWord and dp[n] <= k:
            result.append(node.str)

        next = [0 for i in range(n + 1)]

        for i in range(26):
            if node.children[i] is not None:
                next[0] = dp[0] + 1
                for j in range(1, n + 1):
                    if ord(target[j-1]) - ord('a') == i:
                        next[j] = min(dp[j-1], min(next[j-1] + 1, dp[j] + 1))
                    else:
                        next[j] = min(dp[j-1] + 1, min(next[j-1] + 1, dp[j] + 1))

                self.find(node.children[i], result, k, target, next)


# 118. 不同的子序列 ⭐
# [2020年11月6日]
# https://www.lintcode.com/problem/distinct-subsequences/description
# https://www.jiuzhang.com/solution/distinct-subsequences/#tag-lang-python
class Solution:
    """
    @param S: A string
    @param T: A string
    @return: Count the number of distinct subsequences
    """
    def numDistinct(self, S, T):
        n, m = len(S), len(T)
        if n < m:
            return False
        dp = [[0] * (m+1) for _ in range(n+1)]

        for i in range(n+1):
            dp[i][0] = 1
        for j in range(1, m+1):
            dp[0][j] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                if S[i-1] != T[j-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    # not use / use
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
        return dp[n][m]


# 29. 交叉字符串
# https://www.lintcode.com/problem/interleaving-string/description
# https://www.jiuzhang.com/solution/interleaving-string/#tag-lang-python
class Solution:
    """
    @params s1, s2, target: Three strings as description.
    @return: return True if target is formed by the interleaving of
             s1 and s2 or False if not.
    @hint: you can use [[True] * m for i in range (n)] to allocate a n*m matrix.
    """
    def isInterleave(self, s1, s2, target):
        if s1 is None or s2 is None or target is None:
            return False
        if len(s1) + len(s2) != len(target):
            return False

        dp = [[False] * (len(s2) + 1) for i in range(len(s1) + 1)]
        dp[0][0] = True
        for i in range(len(s1)):
            dp[i + 1][0] = s1[:i + 1] == target[:i + 1]
        for i in range(len(s2)):
            dp[0][i + 1] = s2[:i + 1] == target[:i + 1]

        for i in range(len(s1)):
            for j in range(len(s2)):
                dp[i + 1][j + 1] = False
                if s1[i] == target[i + j + 1]:
                    dp[i + 1][j + 1] = dp[i][j + 1]
                if s2[j] == target[i + j + 1]:
                    dp[i + 1][j + 1] |= dp[i + 1][j]
        return dp[len(s1)][len(s2)]

class Solution:
    """
    @param s1: A string
    @param s2: A string
    @param s3: A string
    @return: Determine whether s3 is formed by interleaving of s1 and s2
    """
    def isInterleave(self, s1, s2, s3):
        return self.helper(s1, s2, s3)
        
    def helper(self, s1, s2, s3):
        if s1 == "" and s2 == "" and s3 == "":
            return True
        if s1 == "":
            if s2 == s3:
                return True
            else:
                return False
        if s2 == "":
            if s1 == s3:
                return True
            else:
                return False
        if s1[0] == s2[0] and s1[0] == s3[0]:
            return self.helper(s1[1:], s2, s3[1:]) or self.helper(s1, s2[1:], s3[1:])
        if s1[0] == s3[0]:
            return self.helper(s1[1:], s2, s3[1:])
        if s2[0] == s3[0]:
            return self.helper(s1, s2[1:], s3[1:])
        if s1[0] != s3[0] and s2[0] != s3[0]:
            return False


"""背包类Dp"""
# 92. 背包问题
# [2020年11月6日 2021年4月7日]
# https://www.lintcode.com/problem/backpack/description
# https://www.jiuzhang.com/solution/backpack/#tag-lang-python
# 在n个物品中挑选若干物品装入背包，最多能装多满？假设背包的大小为m
# version 1
class Solution:
    def backPack(self, m, A):
        n = len(A)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True

        for i in range(1, n+1):
            dp[i][0] = True

        for i in range(1, n+1):
            for j in range(1, m+1):
                if j >= A[i-1]:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - A[i-1] ]
                else:
                    dp[i][j] = dp[i-1][j] 
        
        for i in range(m, -1, -1):
            if dp[n][i]:
                return i

        return 0
# version 3
class Solution:
    def backPack(self, m, A):
        A.sort()
        intervals = [[0, 0]]
        for item in A:
            new_intervals = []
            for interval in intervals:
                new_intervals.append([interval[0] + item, interval[1] + item])
                
            intervals = self.merge_intervals(intervals, new_intervals)

        max_size = 0
        for interval in intervals:
            if interval[0] <= m <= interval[1]:
                return m
            if interval[0] > m:
                break
            max_size = max(max_size, interval[1])
        return max_size
            
    def merge_intervals(self, list1, list2):
        i, j = 0, 0
        intervals = []
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                self.push_to_intervals(intervals, list1[i])
                i += 1
            else:
                self.push_to_intervals(intervals, list2[j])
                j += 1
                
        while i < len(list1):
            self.push_to_intervals(intervals, list1[i])
            i += 1
        
        while j < len(list2):
            self.push_to_intervals(intervals, list2[j])
            j += 1
            
        return intervals
        
    def push_to_intervals(self, intervals, interval):
        if not intervals or intervals[-1][1] + 1 < interval[0]:
            intervals.append(interval)
            return
        
        intervals[-1][1] = max(intervals[-1][1], interval[1])


# 125. 背包问题 II
# [2020年11月6日 2020年1月5号 2021年4月8日]
# https://www.lintcode.com/problem/backpack-ii/description
# https://www.jiuzhang.com/solution/backpack-ii/#tag-lang-python
# 有 n 个物品和一个大小为 m 的背包. 给定数组 A 表示每个物品的大小和数组 V
# version 1
class Solution:
    def backPackII(self, m, A, V):
        n = len(A)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(1, n+1):
            dp[i][0] = 0
            for j in range(1, m+1):
                dp[i][j] = dp[i-1][j] 
                if j >= A[i-1]:
                    # ! 仔细品品dp的含义，dp[i][j]: 对于前i个物品，当前书包的容量为w，这个情况下装下的最大价值
                    dp[i][j] = max( dp[i][j], dp[i-1][j - A[i-1]] + V[i-1])

        return dp[n][m]
# version 3
class Solution:
    """
    @param m: An integer m denotes the size of a backpack
    @param A: Given n items with size A[i]
    @param V: Given n items with value V[i]
    @return: The maximum value
    """
    def backPackII(self, m, A, V):
        n = len(A)
        dp = [[0] * (m + 1), [0] * (m + 1)]

        for i in range(1, n + 1):
            dp[i%2][0] = 0
        
            for j in range(1, m + 1):
                dp[i%2][j] = dp[(i-1) % 2][j]
                if A[i-1] <= j:
                    dp[i%2][j] = max( dp[i%2][j], dp[(i-1) % 2][j - A[i-1]] + V[i-1] )
        
        return dp[n % 2][m]


# 562. 背包问题 IV
# [2020年11月7日  2021年4月8日]
# https://www.lintcode.com/problem/backpack-iv/description
# https://www.jiuzhang.com/solution/backpack-iv/#tag-lang-python
# 给出 n 个物品, 以及一个数组, nums[i]代表第i个物品的大小, 保证大小均为正数并且没有重复, 正整数 target 表示背包的大小, 找到能填满背包的方案数
# 每一个物品可以使用无数次
class Solution:
    def backPackIV(self, nums, target):
        if not nums: 
            return 0
            
        n = len(nums)
        dp = [[0] * (target + 1) for _ in range(target+1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            dp[i][0] = 1
            for j in range(1, target + 1):
                dp[i][j] = dp[i-1][j]
                if j >= nums[i-1]:
                    dp[i][j] += dp[i][j - nums[i-1]]

        return dp[n][target]

class Solution:
    def backPackIV(self, nums, target):
        if not nums: 
            return 0
            
        n = len(nums)
        dp = [[0] * (target + 1) , [0] * (target + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            dp[i%2][0] = 1
            for j in range(1, target + 1):
                dp[i%2][j] = dp[(i-1) % 2][j]
                if j >= nums[i-1]:
                    dp[i%2][j] += dp[i%2][j - nums[i-1]]

        return dp[n % 2][target]


# 740 · 零钱兑换 2
# https://www.lintcode.com/problem/740/
class Solution:
    def change(self, amount, coins):
        n = len(coins)

        dp = [ [0] * (amount+1) for _ in range(n+1) ]
        for i in range(n+1):
            dp[i][0] = 1
        
        for i in range(1, n+1):
            for j in range(1, amount+1):
                if j < coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    # dp[i][j] = dp[i-1][j] + dp[i-1][j - coins[i-1]]
                    dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]]
        
        return dp[-1][-1]


# 89. K数之和
# [2020年11月7日 2021年4月8日]
# https://www.lintcode.com/problem/k-sum/description
# https://www.jiuzhang.com/solution/k-sum/#tag-lang-python
# 在这 n 个数里面找出 k 个数，使得这 k 个数的和等于目标数字
# DESC 使用滚动数组的三维背包
class Solution:
    def kSum(self, A, k, target):
        n = len(A)
        dp = [
            [[0] * (target + 1) for _ in range(k + 1)] for i in range(n+1)
        ]

        for i in range(n+1):
            dp[i][0][0] = 1
        
        for i in range(1, n+1):
            for j in range(1, min(k+1, i+1)):
                for s in range(1, target+1):
                    dp[i][j][s] = dp[i-1][j][s]
                    if s >= A[i-1]:
                        dp[i][j][s] += dp[i-1][j-1][s-A[i-1]]
        
        return dp[-1][-1][-1]

class Solution:
    def kSum(self, A, k, target):
        n = len(A)
        dp = [
            [[0] * (target + 1) for _ in range(k + 1)],
            [[0] * (target + 1) for _ in range(k + 1)],
        ]
        
        # dp[i][j][s], 前 i 个数里挑出 j 个数，和为 s
        dp[0][0][0] = 1
        for i in range(1, n + 1):
            dp[i%2][0][0] = 1
            for j in range(1, min(k + 1, i + 1)):
                for s in range(1, target + 1):
                    dp[i%2][j][s] = dp[(i-1) % 2][j][s]

                    if s >= A[i-1]:
                        dp[i%2][j][s] += dp[(i-1) % 2][j-1][s - A[i-1]]
                        
        return dp[n % 2][k][target]


# 91. 最小调整代价 # TODO
# https://www.lintcode.com/problem/minimum-adjustment-cost/description
# https://www.jiuzhang.com/solution/minimum-adjustment-cost/#tag-lang-python
# 给一个整数数组，调整每个数的大小，使得相邻的两个数的差不大于一个给定的整数target，
# 调整每个数的代价为调整前后的差的绝对值，求调整代价之和最小是多少
class Solution:
    """
    @param: A: An integer array
    @param: target: An integer
    @return: An integer
    """
    def MinAdjustmentCost(self, A, target):
        n = len(A)
        # dp[i][j]表示元素A[i]=j时，A[i]与A[i-1]差值不大于target所需要付出的最小代价
        # 初始化为极大值
        dp = [[sys.maxsize] * 101 for _ in range(n)]
        for i in range(n):
            for j in range(1, 101):
                if i == 0:
                    # 临界值：第一个元素A[0]调整为j的代价
                    dp[0][j] = abs(j - A[0])
                else:
                    # left为A[i]=j时，A[i-1]与A[i]差值不大于target的A[i-1]最小值
                    # right为A[i]=j时，A[i-1]与A[i]差值不大于target的A[i-1]最大值
                    left = max(1, j - target)
                    right = min(100, j + target)
                    for k in range(left, right + 1):
                        # 当A[i-1]=k时，答案为A[i-1]=k的代价dp[i-1][k]，加上A[i]=j的调整代价abs(j-A[i])
                        dp[i][j] = min(dp[i][j], dp[i-1][k] + abs(j - A[i]))
        
        mincost = sys.maxsize
        for i in dp[n - 1]:
            mincost = min(mincost, i)
        return mincost

class Solution:
    """
    @param: A: An integer array
    @param: target: An integer
    @return: An integer
    """
    def MinAdjustmentCost(self, A, target):
        ret = float('inf')
        memo = [[-1 for _ in range(81)] for __ in range(101)]
        for n in range(1, 101):
            ret = min(ret, self.dfs(A, 1, n, target, memo) + abs(A[0]-n))
        return ret

    def dfs(self, nums, index, prev, target, memo):
        if index == len(nums):
            return 0
        
        if memo[prev][index] >= 0:
            return memo[prev][index]
        
        min_val = float('inf')
        for n in range(
            max(prev-target, 1), 
            min(prev+target+1, 101)):
            
            min_val = min(min_val, self.dfs(nums, index+1, n, target, memo) + abs(nums[index]-n))
        
        memo[prev][index] = min_val
        return min_val
    