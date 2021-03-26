from typing import Collection


from collections import deque

class Solution:
    """
    @param length: the length of board
    @param connections: the connections of the positions
    @return: the minimum steps to reach the end
    """
    def modernLudo(self, length, connections):
        queue = deque([1])
        dist = {1: 0}

        graph = {}
        for o, d in connections:
            graph[o] = graph.get(o, set())
            graph[o].add(d)
        
        while queue:
            cur = queue.popleft()

            if cur in graph:
                for nxt in graph[cur]:
                    if nxt in dist and dist[nxt] <= dist[cur]:
                        continue

                    queue.append(nxt)
                    dist[nxt] = dist[cur] 

            for i in range(1, 7):
                nxt = cur + i
                if nxt <= length:
                    if nxt in dist and dist[nxt] <= dist[cur] + 1:
                        continue

                    queue.append(nxt)
                    dist[nxt] = dist[cur] + 1


        return dist[length]                    