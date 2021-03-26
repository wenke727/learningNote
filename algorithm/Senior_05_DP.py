"""
动态滚动数组的四点要素：
 * 状态
    存储小规模的结果（最优解、Yes/No、Count）
 * 方程
    状态之间是怎么转换的，小的状态 -> 大的状态
 * 初始化
    最极限的小状态是什么来求最大值，起点
 * 答案
    最大的那个状态是什么，终点

"""

"""滚动数组优化"""
# 392/198. 打劫房屋
# [2020年11月5日, 2020年11月23日, 2021年1月7日, 2021年3月20日]
# https://www.lintcode.com/problem/house-robber/description
# https://www.jiuzhang.com/solution/house-robber/#tag-lang-python
class Solution:
    # ! `%2`也是对的
    def houseRobber(self, A):
        if not A:
            return 0
        if len(A) <= 2:
            return max(A)
            
        f = [0] * 3
        f[0], f[1] = A[0], max(A[0], A[1])
        
        for i in range(2, len(A)):
            f[i % 3] = max( f[(i - 1) % 3], (f[(i - 2) % 3] + A[i]) )
            
        return f[(len(A) - 1) % 3]


# 534/213. 打劫房屋 II
# [2020年11月5日, 2020年11月23日, 2021年1月7日, 2021年3月20日]
# https://www.lintcode.com/problem/house-robber-ii/description
# https://www.jiuzhang.com/solution/house-robber-ii/#tag-lang-python
# version myself
class Solution:
    def houseRobber2(self, nums):
        if not nums: return 0

        n = len(nums)
        if n <= 2:
            return max( nums )

        dp = [0] * n

        # case 0
        dp[0], dp[1] = 0, nums[1]
        for i in range(2, n):
            dp[i%3] = max( dp[(i-1) % 3], dp[(i-2) % 3] + nums[i] )
        ans = dp[(n-1) % 3]

        # case 1
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, n):
            dp[i%3] = max( dp[(i-1)%3], dp[(i-2)%3] + nums[i] )
        
        return max( ans, dp[(n-2)%3] )


# 535. 打劫房屋 III
# [2021年3月20日]
# https://www.lintcode.com/problem/house-robber-iii/description
# 当dp不好写的时候就可以考虑memo了, 这里有一个坑需要小心
class Solution:
    """
    @param root: The root of binary tree.
    @return: The maximum amount of money you can rob tonight
    """
    def houseRobber3(self, root):
        rob, not_rob = self.visit(root)
        return max(rob, not_rob)

    def visit(self, root):
        if root is None: return 0, 0

        l_rob, l_no_rob = self.visit(root.left)
        r_rob, r_no_rob = self.visit(root.right)

        rob = r_no_rob + l_no_rob + root.val
        no_rob = max(l_rob, l_no_rob) + max(r_rob, r_no_rob)

        return rob, no_rob


# 111. 爬楼梯
# [2020年11月5日, 2020年11月23日]
# https://www.lintcode.com/problem/climbing-stairs/description
# https://www.jiuzhang.com/solution/climbing-stairs/#tag-lang-python
class Solution:
    """
    @param n: An integer
    @return: An integer
    """
    def climbStairs(self, n):
        # write your code here
        if n == 0:
            return 1
        if n <= 2:
            return n
        result=[1,2]
        for i in range(n-2):
            result.append(result[-2]+result[-1])
        return result[-1]

class Solution:
    def climbStairs(self, n):
        if n == 0: return 0
        res = self.steps(n, {1:1, 2:2})

        return res
    
    def steps(self, n, memo):
        if n in memo: 
            return memo[n]
        
        memo[n] = self.steps( n-1, memo ) + self.steps( n-2, memo )

        return memo[n]


# 436/221. 最大正方形
# [2020年11月5日 2020年03月24日]
# https://www.lintcode.com/problem/maximal-square/description
# https://www.jiuzhang.com/solutions/maximal-square/#tag-lang-python
# version 1
class Solution:
    def maxSquare(self, matrix):
        if not matrix or len(matrix[0]) == 0: return 0

        n, m = len(matrix), len(matrix[0])
        dp = [[0] * (m) for _ in range(n)]

        for i in range(m):
            dp[0][i] = matrix[0][i]
        for i in range(n):
            dp[i][0] = matrix[i][0]

        edge = max(matrix[0])  # case: [["0","1"]]
        for i in range(1, n):
            for j in range(1, m):
                if matrix[i][j]:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i-1][j - 1]) + 1
                else:
                    dp[i][j] = 0
            edge = max(edge, max(dp[i]))
            # edge = max(edge, dp[i][j] ) #! dp为 n x m
        return edge * edge
# version 2: 滚动数组优化
class Solution:
    def maxSquare(self, matrix):
        if not matrix or not matrix[0]: return 0
        n, m = len(matrix), len(matrix[0])
        
        # intialization
        f = [[0] * m, [0] * m]
        for i in range(m):
            f[0][i] = matrix[0][i]
            
        edge = max(matrix[0])
        for i in range(1, n):
            f[i % 2][0] = matrix[i][0]
            for j in range(1, m):
                if matrix[i][j]:
                    f[i % 2][j] = min(f[(i - 1) % 2][j], f[i % 2][j - 1], f[(i - 1) % 2][j - 1]) + 1
                else:
                    f[i % 2][j] = 0
            edge = max(edge, max(f[i % 2]))

        return edge * edge


# 631. Maximal Square II 
# [2020年11月5日 2020年03月24日]
# https://www.lintcode.com/problem/maximal-square-ii/description
# https://www.jiuzhang.com/solution/maximal-square-ii/#tag-lang-python
# DESC 使用九章算法强化班和动态规划专题班中讲过的滚动数组+矩阵型动态规划
# DESC 动态规划，u和l数组分别代表左边三角形的最大值和上方三角形的最大值，而f代表对角线到此点的最大长度。 直接三者求最小值转移即可。
class Solution:
    def maxSquare2(self, matrix):
        if not matrix or not matrix[0]: return 0

        n, m = len(matrix), len(matrix[0])
        dp = [[0] * m for i in range(2) ]
        up = [[0] * m for i in range(2) ]

        for i in range(m):
            dp[0][i] = matrix[0][i]
            up[0][i] = 1 - matrix[0][i]
        
        edge = max(matrix[0])

        for i in range(1, n):
            dp[i % 2][0] = matrix[i][0]
            up[i % 2][0] = 0 if matrix[i][0] else up[(i-1) % 2][0] +1
            left = 1 - matrix[i][0]

            for j in range(1, m):
                if matrix[i][j]:
                    dp[i % 2][j] = min( dp[(i-1)%2][j-1], up[(i-1)%2][j], left ) + 1
                    up[i % 2][j] = 0
                    left = 0
                else:
                    dp[i % 2][j] = 0
                    up[i % 2][j] = up[(i-1) % 2][j] + 1
                    left += 1
            
            edge = max(edge, max(dp[i % 2]) )

        return edge * edge


# 114. 不同的路径
# [2020年03月24日]
# https://www.lintcode.com/problem/unique-paths/description
# https://www.jiuzhang.com/solution/unique-paths/#tag-lang-python
class Solution:
    def uniquePaths(self, m, n):
        dp = [[0] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        return dp[m - 1][n - 1]


# 110. 最小路径和
# [2020年03月24日]
# DESC 给定一个只含非负整数的m*n网格，找到一条从左上角到右下角的可以使数字和最小的路径。
# https://www.lintcode.com/problem/minimum-path-sum/description
# https://www.jiuzhang.com/solution/minimum-path-sum/#tag-lang-python
class Solution:
    def minPathSum(self, grid):
        if not grid: return 0

        m, n = len(grid), len(grid[0])
        f = [[0 for j in range(0, n)] for i in range(0, m)]

        f[0][0] = grid[0][0]
        for i in range(1, m):
            f[i][0] = f[i - 1][0] + grid[i][0]
        for j in range(1, n):
            f[0][j] = f[0][j - 1] + grid[0][j]

        # 状态转移
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = min( f[i-1][j], f[i][j-1] ) + grid[i][j]

        return f[m - 1][n - 1]


# 119. 编辑距离 # TODO
# [2021年1月4日]
# DESC 给出两个单词word1和word2，计算出将word1 转换为word2的最少操作次数
# https://www.lintcode.com/problem/edit-distance/description 
# https://www.jiuzhang.com/solution/edit-distance/#tag-lang-python
class Solution:
    def minDistance(self, word1, word2):
        n, m = len(word1), len(word2)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            f[i][0] = i
        for j in range(m + 1):
            f[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if word1[i - 1] == word2[j - 1]:
                    # equal, delete, add
                    f[i][j] = min( f[i - 1][j - 1], f[i - 1][j] + 1, f[i][j - 1] + 1 )
                else:
                    f[i][j] = min( f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
                        
        return f[n][m]

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
        # self.print_result(word1, word2, dp)
            
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
        for i in range(len(dp)):
            res = []
            for j in dp[i]:
                res.append( j.value) 
            print(res)
        print()
        for i in range(len(dp)):
            res = []
            for j in dp[i]:
                res.append( j.choice) 
            print(res)
        
        
        rows, cols = len(dp), len(dp[0])
        i , j = rows-1, cols-1
        print( f"change {word1} to {word2}:" )
        
        while i != 0 and j != 0:
            c1, c2 = word1[i-1], word2[j-1]
            choice = dp[i][j].choice
            print( f"word1[{i-1}]:" )
            if choice == 0:
                print( f'\tskip {c1}, [{i},{j}]' )
                i-=1; j-=1
            elif choice == 1:
                print( f'\treplace {c1} with {c2}, [{i},{j}]' )
                i-=1; j-=1
            elif choice == 2:
                print( f'\tdelete {c1}, [{i},{j}]' )
                i-=1
            else:
                print( f"\tinsert {c2}, [{i},{j}]" )
                j-=1
        
        while i > 0:
            print( f"word1[{i-1}]:" )
            print( f'\tdelete {word1[i-1]}, [{i},{j}]' )
            i-=1
            
        while j > 0:
            print( f"word1[0]:" )
            print( f'\tinsert {word2[j-1]}, [{i},{j}]' )
            j-=1
            

"""记忆化搜索"""
# [2021年3月22日]
# https://www.lintcode.com/problem/longest-continuous-increasing-subsequence/
class Solution:
    """
    @param A: An array of Integer
    @return: an integer
    """
    def longestIncreasingContinuousSubsequence(self, A):
        # write your code here
        size = len(A)
        if size < 1:
            return 0 
            
        if size < 2:
            return 1 
            
        dp1, dp2 = 1, 1 
        
        glomax = 0 
        
        for i in range(1, size):
            dp1 = dp1 + 1 if A[i] > A[i - 1] else 1 
            dp2 = dp2 + 1 if A[i] < A[i - 1] else 1 
            glomax = max(glomax, max(dp1, dp2))
            
        return glomax

# 398. 最长上升连续子序列 II
# [2020年11月6日]
# https://www.lintcode.com/problem/longest-continuous-increasing-subsequence-ii/description
# https://www.jiuzhang.com/solutions/longest-continuous-increasing-subsequence-ii/#tag-lang-python
# version 动态数组 BFS
class Solution:
    def longestContinuousIncreasingSubsequence2(self, matrix):
        if not matrix or not matrix[0]: return 0

        n, m = len(matrix), len(matrix[0])
        points, memo = [], {}
        # DESC 序列性动态规划,我们把二维矩阵打散成为一位数组，数组中每个元素记录二维矩阵中的坐标和高度。 然后把一位数组按照高度排序
        for i in range(n):
            for j in range(m):
                points.append( (matrix[i][j], i, j) )
        points.sort()

        for i in range( len(points) ):
            key = (points[i][1], points[i][2])
            memo[key] = 1

            for dx, dy in [ (1,0), (-1,0), (0,1), (0,-1) ]:
                x_nxt, y_nxt = key[0] + dx, key[1] + dy
                if not self.inside(matrix, x_nxt, y_nxt):
                    continue

                if (x_nxt, y_nxt) in memo and matrix[x_nxt][y_nxt] < points[i][0]:
                    memo[key] = max( memo[key], memo[(x_nxt, y_nxt)] + 1 )
        
        return max(memo.values())

    def inside(self, matrix, x, y):
        return 0 <= x < len(matrix) and 0 <= y < len(matrix[0])
# version classical: dfs
class Solution:
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]: return 0

        n, m = len(matrix), len(matrix[0])
        memo = {}
        longest = 0
        for i in range(n):
            for j in range(m):
                longest = max(longest, self.dfs(matrix, i, j, memo))
        return longest

    def dfs(self, matrix, x, y, memo):
        if (x,y) in memo:
            return memo[(x,y)]

        longest = 1 
        for dx, dy in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
            x_nxt, y_nxt = x + dx, y + dy
            if not self.inside(matrix, x_nxt, y_nxt) or matrix[x_nxt][y_nxt] >= matrix[x][y]:
                continue
            longest = max( longest, self.dfs( matrix, x_nxt, y_nxt, memo ) + 1 )

        memo[(x,y)] = longest
        return longest

    def inside(self, matrix, x, y):
        return 0 <= x < len(matrix) and 0 <= y < len(matrix[0])


"""博弈类DP"""
# 394. 硬币排成线
# [2020年11月5日]
# https://www.lintcode.com/problem/coins-in-a-line/description
# https://www.jiuzhang.com/solution/coins-in-a-line/#tag-lang-python
# version 1
class Solution:
    def firstWillWin(self, n):
        return self.dfs(n, {})

    def dfs( self, n, memo ):
        if n in memo:
            return memo[n]
            
        if n == 0: 
            return False

        if n == 1 or n == 2:
            return True

        case1 = self.dfs(n-2, memo) and self.dfs(n-3, memo)
        case2 = self.dfs(n-3, memo) and self.dfs(n-4, memo)
        res = case1 or case2
        
        memo[n] = res
        return res
# version 2
class Solution:
    def firstWillWin(self, n):
        dp = [False, True, True]
        for i in range(3, n + 1):
            dp[i%3] = not dp[(i-1) % 3] or not dp[(i-2) % 3]
        return dp[n % 3]


# 395. 硬币排成线 II # TODO
# https://www.lintcode.com/problem/coins-in-a-line-ii/description
# https://www.jiuzhang.com/solution/coins-in-a-line-ii/#tag-lang-python
# version 普通方式
class Solution:
    """
    @param values: a vector of integers
    @return: a boolean which equals to true if the first player will win
    """
    def firstWillWin(self, values):
        size = len(values);
        if size <= 2:
            return True
        
        dp = [0] * (size + 1)
        Sum = 0
        dp[size - 1] = values[size - 1] # i=len-1时,只有一个可以拿
        dp[size - 2] = values[size - 1] + values[size - 2] # i = len-2,有两个可拿，直接拿走
        dp[size - 3] = values[size - 2] + values[size - 3] # 当i=len-3的时候，剩下最后三个，这时候如果拿一个，对方就会拿走两个，所以这次拿两个
        Sum += (values[size - 1] + values[size - 2] + values[size - 3])
        # 当i = len-4以及以后的情况中，显然可以选择拿一个或者拿两个两种情况，我们自然是选择拿最多的那个作为`dp`的值
        for i in range(size - 4,-1,-1):
            Sum += values[i]
            dp[i] = max(values[i] + min(dp[i + 2], dp[i + 3]), # 只拿一个,那么对手可能拿两个或者一个，对手肯定是尽可能多拿，所以我们要选择尽可能小的那个
                values[i] + values[i + 1] + min(dp[i + 3], dp[i + 4])) # 拿两个，同样的情况
        # 由于硬币总数是确定的，我们比较一下先手的硬币dp[0]和后手的硬币数量sum-dp[0]就能得到答案 
        return dp[0] > Sum - dp[0]
# Version 
class Solution:
    def firstWillWin(self, values):
        if not values: return False
        if len(values) <= 2: return True
            
        n = len(values)
        
        f = [0] * 3
        prefix_sum = [0] * 3
        f[(n - 1) % 3] = prefix_sum[(n - 1) % 3] = values[n - 1]

        # traverse values in reverse order from n-1 to 0
        for i in range(n - 2, -1, -1):
            prefix_sum[i % 3] = prefix_sum[(i + 1) % 3] + values[i]
            f[i % 3] = max(
                values[i] + prefix_sum[(i + 1) % 3] - f[(i + 1) % 3],
                values[i] + values[i + 1] + prefix_sum[(i + 2) % 3] - f[(i + 2) % 3],
            )
        return f[0] > prefix_sum[0] - f[0]
# version 记忆化搜索，自顶向下
class Solution:
    """
    @param values: a vector of integers
    @return: a boolean which equals to true if the first player will win
    """
    def firstWillWin(self, values):
        if not values: return False

        if len(values) <= 2: return True

        first, sencond = self.dfs(values, 0, {})
        return first > sencond

    def dfs(self, values, index, memo):
        if index in memo:
            return memo[index]

        if index >= len(values):
            return 0, 0
        if index == len(values) - 1:
            return values[index], 0
        if index == len(values) - 2:
            return values[index] + values[index+1], 0

        first1, second1 = self.dfs(values, index + 1, memo) 
        first2, second2 = self.dfs(values, index + 2, memo)

        total = values[index] + first1 + second1
        first = max( (values[index] + second1), (values[index] + values[index+1] + second2) )

        memo[index] = (first, total - first)
        return memo[index]


# 396. 硬币排成线 III #TODO⭐️⭐️⭐️
# https://www.lintcode.com/problem/coins-in-a-line-iii/description
# https://www.jiuzhang.com/solution/coins-in-a-line-iii/#tag-lang-python
# version
class Solution:
    """
    @param values: a vector of integers
    @return: a boolean which equals to true if the first player will win
    """
    def firstWillWin(self, values):
        n = len(values)
        if n < 2:
            return True
        # dp[i][j] -- best total values the player can get in (values[i], values[j])
        dp = [[0 for x in range(n)] for y in range(n)]
        total = sum(values)
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i][j] = values[i]
                elif i+1 == j:
                    dp[i][j] = max(values[i:i+2])
                else:
                    dp[i][j] = max([values[i] + min([dp[i+2][j], dp[i+1][j-1]]), 
                                    values[j] + min([dp[i+1][j-1], dp[i][j-2]])])
        return dp[0][n-1] > total/2
# version 使用记忆化搜索的版本
class Solution:
    def firstWillWin(self, values):
        if not values: return False
        first, second = self.dfs(values, 0, len(values) - 1, {})
        return first > second
        
    def dfs(self, values, left, right, memo):
        if left == right:
            return values[left], 0
        
        if (left, right) in memo:
            return memo[(left, right)]
        
        first1, second1 = self.dfs(values, left + 1, right, memo)
        first2, second2 = self.dfs(values, left, right - 1, memo)
        
        total = first1 + second1 + values[left]
        first = max(values[left] + second1, values[right] + second2)
        
        memo[(left, right)] = first, total - first
        return first, total - first



"""ladder exercise"""

# 191/152. 乘积最大子序列
# https://www.lintcode.com/problem/maximum-product-subarray/description?_from=ladder&&fromId=4
class Solution:
    """
    @param nums: An array of integers
    @return: An integer
    """
    def maxProduct(self, nums):
        if not nums:
            return None
            
        global_max = prev_max = prev_min = nums[0]
        for num in nums[1:]:
            if num > 0:
                curt_max = max(num, prev_max * num)
                curt_min = min(num, prev_min * num)
            else:
                curt_max = max(num, prev_min * num)
                curt_min = min(num, prev_max * num)
            
            global_max = max(global_max, curt_max)
            prev_max, prev_min = curt_max, curt_min
            
        return global_max

# 676. 解码方式 II
# https://www.lintcode.com/problem/decode-ways-ii/description?_from=ladder&&fromId=4
# https://www.jiuzhang.com/solution/decode-ways-ii/#tag-lang-python
class Solution(object):
    def numDecodings(self, s):
        if s == None: return 0
        mod = 1000000007
        n = len(s)

        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            if s[i - 1] == '*':
                dp[i] = (dp[i] + 9 * dp[i - 1]) % mod
                if i >= 2:
                    t = 0
                    if s[i - 2] == '*':
                        dp[i] = (dp[i] + 15 * dp[i - 2]) % mod
                    elif s[i - 2] == '1':
                        dp[i] = (dp[i] + 9 * dp[i - 2]) % mod
                    elif s[i - 2] == '2':
                        dp[i] = (dp[i] + 6 * dp[i - 2]) % mod
            else:
                if s[i - 1] >= '1' and s[i - 1] <= '9':
                    dp[i] = (dp[i] + dp[i - 1]) % mod
                if i >= 2:
                    if s[i - 2] == '*':
                        t = 0
                        if s[i - 1] >= '0' and s[i - 1] <= '6':
                            dp[i] = (dp[i] + 2 * dp[i - 2]) % mod
                        elif s[i - 1] >= '7' and s[i - 1] <= '9':
                            dp[i] = (dp[i] + dp[i - 2]) % mod
                    else:
                        twoDigits = int(s[i - 2 : i])
                        if twoDigits >= 10 and twoDigits <= 26:
                            dp[i] = (dp[i] + dp[i - 2]) % mod
        return dp[n]

class Solution:
    
    def numDecodings(self, encoded) -> int:
        
        mod_base = 1000000007
        if not encoded: return 0 
        if encoded[0] is '0': return 0 
        if len(encoded) == 1: return 9 if encoded[0] is '*' else 1 
        
        def num_of_ways_to_separate(a, b):
            
            if a is not '*' and b is not '*':
                if b is '0': 
                    if a is '1' or a is '2':
                        return 0
                    raise ValueError
                return 1 
            
            if a is '*' and b is '*':
                return 9
                
            if b is '*':
                return 9
                
            if a is '*':
                if b is '0':
                    return 0
                return 1
            
        def num_of_ways_to_combine(a, b):
            
            if a is not '*' and b is not '*':
                if a is '1':
                    return 1 
                if a is '2' and ord('0') <= ord(b) <= ord('6'):
                    return 1
        
                return 0
            
            if a is '*' and b is '*':
                return 9 + 6
                
            if a is '*':
                if ord('0') <= ord(b) <= ord('6'):
                    return 2
                return 1
                
            if b is '*':
                if a is '1':
                    return 9
                if a is '2':
                    return 6
                    
                return 0
        
        n_2 = 1 
        if encoded[0] is '*':
            n_1 = 9 
        else:
            n_1 = 1
        
        try:
            
            for i in range(1, len(encoded)):
                
                n_0 = 0 
                
                multiplier = num_of_ways_to_separate(encoded[i-1], encoded[i])
                n_0 += n_1 * multiplier
                multiplier = num_of_ways_to_combine(encoded[i-1], encoded[i])
                n_0 += n_2 * multiplier
                
                n_0 %= mod_base
                
                n_2, n_1 = n_1, n_0
                
        except ValueError:
            return 0

        return n_0



