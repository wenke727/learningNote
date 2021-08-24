# 355 · 大楼间穿梭
# https://www.lintcode.com/problem/355/
class Solution:
    """
    @param heights: the heights of buildings.
    @param k: the vision.
    @param x: the energy to spend of the first action.
    @param y: the energy to spend of the second action.
    @return: the minimal energy to spend.
    """
    def shuttleInBuildings(self, heights, k, x, y):
        next_position = self.get_next_position(heights, k)
        n = len(heights)
        dp = [float('inf')] * n
        dp[n - 1] = 0
        for i in range(n - 2, -1, -1):
            if i + 1 < n:
                dp[i] = min(dp[i], dp[i + 1] + y)
            if i + 2 < n:
                dp[i] = min(dp[i], dp[i + 2] + y)
            if i in next_position:
                dp[i] = min(dp[i], dp[next_position[i]] + x)

        return dp[0]

    def get_next_position(self, heights, k):
        n = len(heights)
        next_position = {}
        stack = []
        for i in range(n):
            while stack and heights[stack[-1]] < heights[i]:
                if i - stack[-1] <= k:
                    next_position[stack[-1]] = i
                stack.pop()
            stack.append(i)
        
        return next_position
