import sys
class Solution:
    def optimalMatch(self, matrix):
        graph, people, bikes = self._format(matrix)
        return self._kuhn_munkres(graph, people, bikes)

    def _kuhn_munkres(self, graph, left_values, right_values):
        n = len(right_values)
        match = [-1] * n

        for left_node in range(n):
            while True:
                left_visited, right_visited = set(), set()
                left_visited.add(left_node)

                if self._hungarian(graph, left_values, right_values, match, left_visited, right_visited, left_node):
                    break
                
                delta = sys.maxsize
                for l in left_visited:
                    for r in range(len(right_values)):
                        if r in right_visited:
                            continue
                        delta = min(delta, left_values[l] + right_values[r] - graph[l][r])
                
                if delta == sys.maxsize:
                    return -1
                
                for l in left_visited:
                    left_values[l] -= delta
                for r in right_visited:
                    right_values[r] += delta
                
        dis = 0
        for r, l in enumerate(match):
            dis += graph[l][r]
        
        return -dis
    
    def _hungarian(self, graph, left_values, right_values, match, left_visited, right_visited, start_node):
        for r in range(len(right_values)):
            if r in right_visited:
                continue
        
            if left_values[start_node] + right_values[r] == graph[start_node][r]:
                right_visited.add(r)
                if match[r] == -1:
                    match[r] = start_node
                    return True
            
                left_visited.add(match[r])
                if self._hungarian(graph, left_values, right_values, match, left_visited, right_visited, match[r]):
                    match[r] = start_node
                    return True

        return False
    
    def _format(self, matrix):
        left_index_set, right_index_set = [], []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1:
                    left_index_set.append((i,j))
                if matrix[i][j] == 2:
                    right_index_set.append((i,j))
        
        graph = [[None] * len(right_index_set) for _ in range(len(left_index_set))]
        left_values = [-sys.maxsize] * len(left_index_set)
        right_values = [0] * len(right_index_set)

        for l in range(len(left_index_set)):
            for r in range(len(right_index_set)):
                l_x, l_y = left_index_set[l]
                r_x, r_y = right_index_set[r]
                graph[l][r] = -abs(l_x - r_x) - abs(l_y - r_y)
                left_values[l] = max(left_values[l], graph[l][r])

        return graph, left_values, right_values 
