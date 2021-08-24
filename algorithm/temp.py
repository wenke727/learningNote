class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, candidates, target):
        if candidates is None:
            return []
        
        candidates.sort()
        result = []

        self.dfs(candidates, target, 0, [], result)
    
        return result


    def dfs(self, nums, target, index, combination, res):
        # if index >= len(nums):
            # return
        
        if target == 0:
            res.append(combination[:])
            return

        for i in range(index, len(nums)):
            if nums[i] > target:
                continue
            
            if i > 0 and nums[i] == nums[i-1]:
                continue

            combination.append(nums[i])
            self.dfs(nums, target-nums[i], index+1, combination, res)
            combination.pop()
