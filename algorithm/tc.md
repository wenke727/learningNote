# Lintcode

## DP

- [874 · 最大的假期天数](https://www.lintcode.com/problem/874/)

DFS version:

``` python
class Solution:
    """
    @param flights: the airline status from the city i to the city j
    @param days: days[i][j] represents the maximum days you could take vacation in the city i in the week j
    @return: the maximum vacation days you could take during K weeks
    """
    def maxVacationDays(self, flights, days):
        N = len(flights)
        K = len(days[0])

        # start from city 0 and week 0
        return self.dfs(0, 0, flights, days, N, K, {})
    
    def dfs(self, city, week, flights, days, N, K, memo):
        if (city, week) in memo:
            return memo[(city, week)]
        
        if week == K:
            return 0
        
        maxDay = 0
        for i in range(N):
            # we could either stay in current city or fly to other city at current week
            if i == city or flights[city][i] == 1:
                maxDay = max(
                  maxDay, 
                  days[i][week] + self.dfs(i, week+1, flights, days, N, K, memo)
                )
        
        memo[(city, week)] = maxDay

        return maxDay
```

- [286 · 逆序对](https://www.lintcode.com/problem/286/?_from=collection&fromId=208)

给定一个n，一个包含 2^n 个数的数列。再给定一个包含m个数的数组，表示询问m次，对于每个qi，每次要求把这些数每 2^qi 分为一组，然后把每组进行翻转。每次操作后整个序列中的逆序对个数是多少呢？

``` python

```
