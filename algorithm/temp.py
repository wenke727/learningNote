class Solution:
    def backPackIV(self, nums, target):
        if not nums: 
            return 0
        
        n = len(nums)
        dp = [[0]*(target+1) for _ in (n+1)]
        dp[0][0] = 1

        for i in range(1, n+1):
            dp[i][0] = 1
            for j in range(1, target+1):
                dp[i][j] = dp[i-1][j]
                if j >= nums[i-1]:
                    dp[i][j] += dp[i][j-nums[i-1]]
        
        return dp[n][target]
        