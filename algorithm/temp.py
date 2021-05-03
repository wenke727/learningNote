from heapq import heappop, heappush
class Solution:
    def topKFrequentWords(self, words, k):
        word_dict = {}

        for w in words:
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
        
        res = []
        [heappush(res, (-v, k)) for k, v in word_dict.items()]
        return [heappop(res)[1] for _ in range(k)]
