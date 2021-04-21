import sys
import collections
# 1644 · 平面最大矩形
# https://www.lintcode.com/problem/1644/


# 1576 · 最佳匹配
# https://www.lintcode.com/problem/1576/
# https://blog.csdn.net/chenshibo17/article/details/79933191, https://www.cnblogs.com/logosG/p/logos.html
# 给出一个 n * m 的矩阵，矩阵中1表示人的位置，2表示自行车的位置，0表示空地，假设人的位置是(x1,y1)，自行车的位置是(x2,y2)，那么人跟自行车的距离为|x1-x2|+|y1-y2|，一个人只能跟一辆自行车匹配，求如何匹配能使人到自行车的总距离最小，返回这个最小距离。
class Solution:
    def optimalMatch(self, matrix):
        graph, people, bikes = self._format(matrix)
        return self._kuhn_munkres(graph, people, bikes)
    
    def _kuhn_munkres(self, graph, left_values, right_values):
        n = len(left_values)
        match = [-1] * n

        for left_node in range(n):
            while True:
                left_visited, right_visited = set(), set()
                left_visited.add(left_node)

                # 找到一个match，可以直接break; 如果hungarian找不到match, 此时就需要松弛
                if self._hungarian(graph, left_values, right_values, match, left_visited, right_visited, start_node=left_node):
                    break

                delta = sys.maxsize
                # 二重循环: 从左边已经visited的点出发,去找右边**没有在本轮访问过的点**，看看能够松弛的最小cost是多少
                for l in left_visited:
                    for r in range(n):
                        if r in right_visited:
                            continue
                        delta = min(delta, left_values[l] + right_values[r] - graph[l][r])
                
                if delta == sys.maxsize:
                    return -1
                for l in left_visited:
                    left_values[l] -= delta
                for r in right_visited:
                    right_values[r] += delta

        dist = 0
        for r, l in enumerate(match):
            dist += graph[l][r]
        return -dist

    def _hungarian(self, graph, left_values, right_values, match, left_visited, right_visited, start_node ):
        for r in range(len(right_values)):
            if r in right_visited:
                continue

            # 最佳匹配要检查左右点值之和是否和graph相等, 并且此时要放入right visited
            if left_values[start_node] + right_values[r] == graph[start_node][r]:
                right_visited.add(r)
                if match[r] == -1:
                    match[r] = start_node
                    return True

                left_visited.add(match[r])
                if self._hungarian(graph, left_values, right_values, match, left_visited, right_visited, match[r] ):
                    match[r] = start_node
                    return True
        
        return False

    def _format(self, matrix):
        """[summary]

        Args:
            matrix ([type]): 这里1是人，2是自行车，0是空地

        Returns:
            praph[matrix]: 是一个方阵, 行表示人，列表示自行车, graph[i][j]表示距离
            left_values[(x,y), val]: value初始化就是graph[people][bike]
            right_values[(x,y), val]: alue初始化都是0
        """
        left_index_set, right_index_set= [], []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1:
                    left_index_set.append((i, j))
                if matrix[i][j] == 2:
                    right_index_set.append((i, j))

        graph = [[None] * len(right_index_set) for _ in range(len(left_index_set))]
        left_values = [-sys.maxsize] * len(left_index_set)
        right_values = [0] * len(right_index_set)

        for l in range(len(left_index_set)):
            for r in range(len(right_index_set)):
                li, lj = left_index_set[l]
                ri, rj = right_index_set[r]
                graph[l][r] = -abs(li - ri) - abs(lj - rj) 
                left_values[l] = max(left_values[l], graph[l][r] )

        return graph, left_values, right_values


# 1456 · 单词合成问题
# https://www.lintcode.com/problem/1456/
from collections import defaultdict
class Solution:
    """
    @param target: the target string
    @param words: words array
    @return: whether the target can be matched or not
    """
    def matchFunction(self, target, words):
        char_dict = defaultdict(list)
        n = len(words)
        for i in range(n):
            for c in words[i]:
                char_dict[c].append(i)
           
        return self.dfs(target, 0, words, char_dict, set()) 
        
    def dfs(self, target, startIndex, words, char_dict, visited):
        if startIndex == len(target):
            return True
        
        if target[startIndex] not in char_dict:
            return False
        
        for idx_word in char_dict[target[startIndex]]:
            if idx_word in visited:
                continue

            visited.add(idx_word)
            if self.dfs(target, startIndex + 1, words, char_dict, visited):
                return True
            visited.remove(idx_word)    
        return False


# 816 · 旅行商问题 # TODO
# https://www.lintcode.com/problem/816/
# https://www.jiuzhang.com/solution/traveling-salesman-problem/
# DESC 城市和无向道路成本之间的关系为3元组 [A, B, C]（在城市 A 和城市 B 之间有一条路，成本是 C）
# DESC 我们需要从1开始找到的旅行所有城市的付出最小的成本。
# version 如果要保证正确性的最优做法。状态压缩动态规划
class Solution:
    """
    @param n: an integer,denote the number of cities
    @param roads: a list of three-tuples,denote the road between cities
    @return: return the minimum cost to travel all cities
    """
    def minCost(self, n, roads):
        graph = self.construct_graph(roads, n)

        state_size = 1 << n
        dp = [ [float('inf')] * (n + 1) for _ in range(state_size) ]
        dp[1][1] = 0

        for state in range(state_size):
            for i in range(2, n + 1):
                if state & (1 << (i - 1)) == 0:
                    continue
                prev_state = state ^ (1 << (i - 1))
                for j in range(1, n + 1):
                    if prev_state & (1 << (j - 1)) == 0:
                        continue
                    dp[state][i] = min(dp[state][i], dp[prev_state][j] + graph[j][i])
        return min(dp[state_size - 1])
        
    def construct_graph(self, roads, n):
        graph = {
            i: {j: float('inf') for j in range(1, n + 1)}
            for i in range(1, n + 1)
        }
        for a, b, c in roads:
            graph[a][b] = min(graph[a][b], c)
            graph[b][a] = min(graph[b][a], c)
        return graph

class Result:
    def __init__(self):
        self.min_cost = float('inf')

# VERSION DFS + pruning
class Solution:
    """
    @param n: an integer,denote the number of cities
    @param roads: a list of three-tuples,denote the road between cities
    @return: return the minimum cost to travel all cities
    """
    def minCost(self, n, roads):
        graph = self.construct_graph(roads, n)
        result = Result()
        self.dfs(1, n, [1], set([1]), 0, graph, result)
        return result.min_cost
        
    def dfs(self, city, n, path, visited, cost, graph, result):
        if len(visited) == n:
            result.min_cost = min(result.min_cost, cost)
            return
    
        for next_city in graph[city]:
            if next_city in visited:
                continue
            
            if self.has_better_path(graph, path, next_city):
                continue

            visited.add(next_city)
            path.append(next_city)
            self.dfs(
                next_city,
                n,
                path,
                visited,
                cost + graph[city][next_city],
                graph,
                result,
            )
            path.pop()
            visited.remove(next_city)
    
    def construct_graph(self, roads, n):
        graph = {
            i: {j: float('inf') for j in range(1, n + 1)}
            for i in range(1, n + 1)
        }
        for a, b, c in roads:
            graph[a][b] = min(graph[a][b], c)
            graph[b][a] = min(graph[b][a], c)
        return graph

    def has_better_path(self, graph, path, city):
        for i in range(1, len(path)):
            # i-1, i, -1(cur), city(nxt) 遍历顺序的问题
            if graph[path[i - 1]][path[i]] + graph[path[-1]][city] >\
               graph[path[i - 1]][path[-1]] + graph[path[i]][city]:
                return True
        return False

# VERSION 随机算法
# 使用随机化算法，不保证正确性，但是可以处理很大的数据，得到近似答案。 调整策略是交换 i, j 两个点的位置，看看是否能得到更优解 测试中如果失败了可以多跑几次。
RANDOM_TIMES = 1000
class Solution:
    """
    @param n: an integer,denote the number of cities
    @param roads: a list of three-tuples,denote the road between cities
    @return: return the minimum cost to travel all cities
    """
    def minCost(self, n, roads):
        graph = self.construct_graph(roads, n)
        min_cost = float('inf')
        for _ in range(RANDOM_TIMES):
            path = self.get_random_path(n)
            cost = self.adjust_path(path, graph)
            min_cost = min(min_cost, cost)
        return min_cost
        
    def construct_graph(self, roads, n):
        graph = {
            i: {j: float('inf') for j in range(1, n + 1)}
            for i in range(1, n + 1)
        }
        for a, b, c in roads:
            graph[a][b] = min(graph[a][b], c)
            graph[b][a] = min(graph[b][a], c)
        return graph
    
    def get_random_path(self, n):
        import random
        
        path = [i for i in range(1, n + 1)]
        for i in range(2, n):
            j = random.randint(1, i)
            path[i], path[j] = path[j], path[i]
        return path
        
    def adjust_path(self, path, graph):
        n = len(graph)
        adjusted = True
        while adjusted:
            adjusted = False
            for i in range(1, n):
                for j in range(i + 1, n):
                    if self.can_swap(path, i, j, graph):
                        path[i], path[j] = path[j], path[i]
                        adjusted = True
        cost = 0
        for i in range(1, n):
            cost += graph[path[i - 1]][path[i]]
        return cost
    
    def can_swap(self, path, i, j, graph):
        before = self.adjcent_cost(path, i, path[i], graph)
        before += self.adjcent_cost(path, j, path[j], graph)
        after = self.adjcent_cost(path, i, path[j], graph)
        after += self.adjcent_cost(path, j, path[i], graph)
        return before > after
    
    def adjcent_cost(self, path, i, city, graph):
        cost = graph[path[i - 1]][city]
        if i + 1 < len(path):
            cost += graph[city][path[i + 1]]
        return cost


# 1031 · 图可以被二分么？
# https://www.lintcode.com/problem/1031/
class Solution:
    """
    @param graph: the given undirected graph
    @return:  return true if and only if it is bipartite
    """
    def isBipartite(self, graph):
        n = len(graph)
        self.color = [0] * n

        for i in range(n):
            if self.color[i] == 0:
                if not self.colored(i, graph, 1):
                    return False
            
        return True
    
    def colored(self, cur, graph, status):
        self.color[cur] = status

        for nxt in graph[cur]:
            if self.color[nxt] == 0:
                if not self.colored(nxt, graph, -status):
                    return False
            
            if self.color[nxt] == self.color[cur]:
                return False
        
        return True

class Solution:
    def isBipartite(self, graph):
        queue = collections.deque()
        node_sets = [set(), set()]
        visited_nodes = set()
        step = 0

        for i in range(len(graph)):
            if i in visited_nodes:
                continue
                
            if graph[i]:
                queue.append(i)
                visited_nodes.add(i)
        
            while queue:
                queue_size = len(queue)
                for _ in range(queue_size):
                    curr_node = queue.popleft()
                    node_sets[step % 2].add(curr_node)
                    for next_node in graph[curr_node]:
                        if next_node in node_sets[step % 2]:
                            return False
                        if next_node not in visited_nodes:
                            queue.append(next_node)
                            visited_nodes.add(next_node)
                step += 1
        
        return True


# 176 · 图中两个点之间的路线
# https://www.lintcode.com/problem/176/
class Solution:
    """
    @param: graph: A list of Directed graph node
    @param: s: the starting Directed graph node
    @param: t: the terminal Directed graph node
    @return: a boolean value
    """
    def hasRoute(self, graph, s, t):
        queue = collections.deque([s])
        visited = set([s])

        while queue:
            node = queue.popleft()
            if node == t:
                return True
            
            for nxt in node.neighbors:
                if nxt in visited:
                    continue
                queue.append(nxt)
                visited.add(nxt)
        
        return False

# 814 · 无向图中的最短路径
# https://www.lintcode.com/problem/814/
class Solution:
    """
    @param graph: a list of Undirected graph node
    @param A: nodeA
    @param B: nodeB
    @return:  the length of the shortest path
    """
    def shortestPath(self, graph, s, t):
        if s == t: return 1 

        queue = collections.deque([s])
        visited = set([s])
        dis = 0

        while queue:
            dis += 1
            length = len(queue)

            for i in range(length):
                node = queue.popleft()
                
                for nxt in node.neighbors:
                    if nxt == t:
                        return dis
                    if nxt in visited:
                        continue
                    queue.append(nxt)
                    visited.add(nxt)
        
        return -1


# 1430 · 相似字符串组
# https://www.lintcode.com/problem/1430/
from collections import deque
class Solution:
    def numSimilarGroups(self, strs):
        word_set = set(strs)
        visited = set()

        similar_set = 0
        for word in strs:
            if word in visited:
                continue
            similar_set += 1
            self.mark_similar_set(word, word_set, visited)

        return similar_set
    
    def mark_similar_set(self, word, word_set, visited):
        queue = deque([word])
        visited.add(word)

        while queue:
            word = queue.popleft()

            for nxt in self.get_neighbor_words(word, word_set):
                if nxt in visited:
                    continue
                queue.append(nxt)
                visited.add(nxt)

        return

    def get_neighbor_words(self, word, word_set):
        if len(word)**2 < len(word_set):
            return self.get_neighbor_words1(word, word_set)
        
        return self.get_neighbor_words2(word, word_set)

    def get_neighbor_words1(self, word, word_set):
        # O(L^3)
        n = len(word)
        chars = list(word)
        neighbor_words = []
        for i in range(n):
            for j in range(i + 1, n):
                chars[i], chars[j] = chars[j], chars[i]
                anagram = ''.join(chars)
                if anagram in word_set:
                    neighbor_words.append(anagram)
                chars[i], chars[j] = chars[j], chars[i]
        return neighbor_words    

    def get_neighbor_words2(self, word, word_set):
        # O(N*L)
        neighbors = []
        for neighbor in word_set:
            if self.is_similar(word, neighbor):
                neighbors.append(neighbor)
        return neighbors

    def is_similar(self, word1, word2):
        diff = 0
        for i, j in zip(word1, word2):
            if i != j:
                diff += 1
        
        return diff == 2


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
