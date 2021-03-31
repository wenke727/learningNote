class Solution:
    def firstWillWin(self, values):
        if not values: return False

        first, second = self.dfs(values, 0, len(values), {})

        return first > second
    
    def dfs(self, values, left, right, memo):
        if left == right: return values[left], 0

        if (left, right) in memo: return memo[(left, right)]

        first1, second1 = self.dfs(values, left+1, right, memo)
        first2, second2 = self.dfs(values, left, right-1, memo)

        total = values[left] + first1 + second1
        first = max(
            values[left] + second1,
            values[right] + second2
        )

        memo[(left, right)] = first, total-first

        return first, total-first

        