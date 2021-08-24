from collections import defaultdict, deque
import collections

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


"""二叉树上的宽度优先搜索"""
# 69. 二叉树的层次遍历
# [2020年10月28日 2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/binary-tree-level-order-traversal/description
# https://www.jiuzhang.com/solutions/binary-tree-level-order-traversal/#tag-lang-python
class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        if root is None: 
            return []

        queue, res = deque([root]), []
        while queue:
            level = []
            # range
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            res.append(level)
        
        return res


# 7. 二叉树的序列化和反序列化 # TODO 
# https://www.lintcode.com/problem/serialize-and-deserialize-binary-tree/description
# https://www.jiuzhang.com/solution/serialize-and-deserialize-binary-tree/#tag-lang-python


# 242. 将二叉树按照层级转化为链表
# [2020年11月2日 2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/convert-binary-tree-to-linked-lists-by-depth/description
# https://www.jiuzhang.com/solutions/convert-binary-tree-to-linked-lists-by-depth/#tag-lang-python
# version bfs
class Solution:
    def binaryTreeToLists(self, root):
        if root is None: 
            return []

        result, queue = [], collections.deque([root])
        while queue:
            result.append(self.create_linkedlist(queue))
            nxtQueue = collections.deque([])
            for node in queue:
                if node.left:
                    nxtQueue.append(node.left)
                if node.right:
                    nxtQueue.append(node.right)
            queue = nxtQueue

        return result

    def create_linkedlist(self, nodes):
        dummy = ListNode(0)
        cur = dummy

        for node in nodes:
            list_node = ListNode(node.val)
            cur.next = list_node
            cur = list_node
        
        return dummy.next
# version dfs
class Solution:
    def binaryTreeToLists(self, root):
        result = []
        self.dfs(root, 1, result)
        return result

    def dfs(self, root, depth, result):
        if root is None:
            return

        node = ListNode(root.val)
        if len(result) < depth:
            result.append(node)
        else:
            node.next = result[depth-1]
            result[depth-1] = node
        
        self.dfs(root.right, depth + 1, result)
        self.dfs(root.left, depth + 1, result)


"""图上的宽度优先搜索"""
# 137/133. 克隆图 ⭐
# [2020年10月28日 2021年7月29日 2021年8月21日 2021年8月22日]
# https://www.lintcode.com/problem/137/
class Solution:
    def cloneGraph(self, node):
        root = node
        if node is None: 
            return node

        # clone nodes
        nodes = self.getNodes(node)
        mapping = {node: UndirectedGraphNode(node.label) for node in nodes}

        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)

        return mapping[root]

    def getNodes(self, node):
        queue, nodes = deque([node]), set([node])
        while queue:
            node = queue.popleft()
            for neighbor in node.neighbors:
                if neighbor in nodes:
                    continue
                nodes.add(neighbor)
                queue.append(neighbor)

        return nodes


# 120/127. 单词接龙
# [2020年10月28日 2021年7月30日 2021年8月22日]
# https://www.lintcode.com/problem/word-ladder/description
# https://www.jiuzhang.com/solution/word-ladder/#tag-lang-python
# DESC 找出从start到end的最短转换序列，输出最短序列的长度
# version：BFS
class Solution:
    def ladderLength(self, start, end, dict):
        dict  = set(list(dict) + [start, end])
        queue = deque([start])
        dis   = {start: 1}

        while queue:
            cur = queue.popleft()
            if cur == end:
                return dis[cur]

            for nxt in self.get_next_words(cur, dict):
                if nxt in dis:
                    continue
                dis[nxt] = dis[cur] + 1
                queue.append(nxt)

        return 0

    def get_next_words(self, word, dict):
        words = []
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + char + word[i+1:]
                if char == word[i] or new_word not in dict:
                    continue
                words.append(new_word)
        return words
# version：双向BFS
class Solution:
    def ladderLength(self, start, end, dict):
        dict   = set(list(dict) + [start, end])

        graph  = self.build_graph(dict)
        forward_queue  = deque( [start] )
        backward_queue = deque( [end] )
        forward_set    = set([start])
        backward_set   = set([end])
    
        distance = 1
        while forward_queue and backward_queue:
            distance += 1
            if self.extend_queue( graph, forward_queue, forward_set, backward_set ):
                return distance
            distance += 1
            if self.extend_queue(graph, backward_queue, backward_set, forward_set):
                return distance

        return 0
    
    def extend_queue(self, graph, queue, visited, opposite_visited):
        for _ in range(len(queue)):
            word = queue.popleft()
            for nxt_word in graph[word]:
                if nxt_word in visited:
                    continue
                if nxt_word in opposite_visited:
                    return True

                queue.append( nxt_word )
                visited.add( nxt_word )
        
        return False

    def build_graph(self, dict):
        graph = {}
        for word in dict:
            graph[word] = self.get_next_words(word, dict)
        return graph


    def get_next_words(self, word, dict):
        words = []
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + char + word[i+1:]
                if char == word[i] or new_word not in dict:
                    continue
                words.append(new_word)
        
        return words
# version：% method
class Solution:
    def ladderLength(self, start, end, dict):
        from collections import deque
        dict  = set( list(dict) + [start, end] )

        queue = deque([start])
        dis   = {start:1}

        gragp = self.build_graph(dict)
        endList = self.get_next_words(end)

        while queue:
            node = queue.popleft()
            if node == end:
                return dis[node]

            for nxt in self.get_next_words(node):
                if nxt in endList:
                    return dis[nxt] + 1

                if nxt in dis:
                    continue
                dis[nxt] = dis[node] + 1
                queue.append(nxt)
        return 0

    def get_next_words(self, word):
        words = []
        for i in range(len(word)):
            words.append(word[:i] + "*" + word[i+1:])
        return words
    
    def build_graph(self, dict):
        gragp = {}
        for i in range(len(dict)):
            for j in range(len(i)):
                key = i[:j] + "*" + i[j+1:]
                temp = gragp.get(key, [])
                temp.append( i )
                gragp[key] = temp
        return gragp


# 121. 单词接龙 II
# [2021年7月31日]
# https://www.lintcode.com/problem/word-ladder-ii/description
# https://www.jiuzhang.com/solution/word-ladder-ii/#tag-lang-python
# DESC 给出两个单词（start和end）和一个字典，找出`所有`从start到end的最短转换序列
# DESC see `chapter_07_DFS_Permutation` 


# 433/200. 岛屿的个数
# [2020年11月2日 2021年7月31日 2021年8月22日]
# https://www.lintcode.com/problem/number-of-islands/description
# https://www.jiuzhang.com/solutions/number-of-islands/#tag-lang-python
DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
class Solution:
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0
        
        isLands, visited = 0, set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i, j) in visited or grid[i][j] == 0:
                    continue
                self.bfs(grid, i, j, visited)
                isLands += 1
        
        return isLands

    def bfs(self, grid, x, y, visited):
        queue = deque([(x, y)])
        visited.add((x,y))

        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                nxt_x, nxt_y = x+dx, y+dy
                if not self.isValid(grid, nxt_x, nxt_y, visited):
                    continue
                queue.append((nxt_x, nxt_y))
                visited.add((nxt_x, nxt_y))

    def isValid(self, grid, x, y, visited):
        n, m = len(grid), len(grid[0])
        if not (0<=x<n and 0<=y<m) or (x,y) in visited:
            return False
        
        return grid[x][y]


# 804 · 不同岛屿的数量II ⭐⭐
# [2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/804/description?_from=collection&fromId=208
# https://www.jiuzhang.com/solution/number-of-distinct-islands-ii/
CHANGES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
class Solution:
    def numDistinctIslands2(self, grid):
        isLands = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    continue
                shape = self.dfs(grid, i, j)
                isLands.add(self.cononical(shape))
        
        return len(isLands)
    
    def dfs(self, grid, x, y):
        if not grid[x][y]:
            return []
        
        grid[x][y] = 0
        shape = [(x,y)]

        for dx, dy in DIRECTIONS:
            nxt_x, nxt_y = x + dx, y + dy
            if not (0 <= nxt_x < len(grid) and 0 <= nxt_y < len((grid[0]))):
                continue
            shape += self.dfs(grid, nxt_x, nxt_y)
        
        return shape
    
    def cononical(self, shape):
        def _encoding(shape):
            x, y = shape[0]
            return ";".join(f"{i-x},{j-y}" for i, j in shape)
        
        shapes = [[[a*i, b*j] for i, j in shape ] for a, b in CHANGES]
        shapes += [[(j,i) for i, j in shape] for shape in shapes]

        return min( _encoding(sorted(shape)) for shape in shapes )


# 611. 骑士的最短路线 ⭐
# [2020年11月2日 2021年7月31日 2021年8月22日]
# https://www.lintcode.com/problem/knight-shortest-path/description
# https://www.jiuzhang.com/solutions/knight-shortest-path/#tag-lang-python
# version bfs
DIRECTIONS = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2),]
class Solution:
    def shortestPath(self, grid, source, destination):
        queue = deque([(source.x, source.y)])
        distance = {(source.x, source.y):0}

        while queue:
            x, y = queue.popleft()
            if (x, y) == (destination.x, destination.y):
                return distance[(x,y)]
            
            for dx, dy in DIRECTIONS:
                nxt = (x+dx, y+dy)
                if nxt in distance:
                    continue
                if not self.is_valid(*nxt, grid):
                    continue

                distance[nxt] = distance[(x,y)] + 1
                queue.append(nxt)
        
        return -1

    
    def is_valid(self, x, y ,grid):
        n, m = len(grid), len(grid[0])

        if 0 <= x < n and 0<= y < m and not grid[x][y]:
            return True
        return False

# version：A*
from heapq import heappop, heappush
DIRECTIONS = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2),]
class Solution:
    def shortestPath(self, grid, src, dst):
        self.dst = dst

        queue = [(self.h_val(src.x, src.y), 0, src.x, src.y)]
        distance = {(src.x, src.y): 0}
    
        while queue:
            _, dis, x, y = heappop(queue)
            if (x, y) == (dst.x, dst.y):
                return distance[(x,y)]

            for dx, dy in DIRECTIONS:
                nxt_x, nxt_y = x + dx, y + dy
                if not self.is_valid(nxt_x, nxt_y, grid, distance):
                    continue

                h = self.h_val(nxt_x, nxt_y)
                nxt_dis = distance[(nxt_x, nxt_y)] = dis + 1
                heappush(queue, (nxt_dis+h, nxt_dis, nxt_x, nxt_y))

        return -1


    def h_val(self, x0, y0):
        return (abs(self.dst.x - x0) + abs(y0 - self.dst.y))/3


    def is_valid(self, x, y ,grid, visited):
        if (x,y) in visited:
            return False

        n, m = len(grid), len(grid[0])

        if 0 <= x < n and 0<= y < m and not grid[x][y]:
            return True

        return False      

# version: two-bfs
DIRECTIONS = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2),]
class Solution:
    def shortestPath(self, grid, source, destination):
        if not grid or not grid[0]:
            return -1
            
        n, m = len(grid), len(grid[0])
        if grid[destination.x][destination.y]:
            return -1
        if n * m == 1:
            return 0
        if source.x == destination.x and source.y == destination.y:
            return 0
        
        forward_queue = collections.deque([(source.x, source.y)])
        forward_set = set([(source.x, source.y)])

        backward_queue = collections.deque([(destination.x, destination.y)])
        backward_set = set([(destination.x, destination.y)])
        
        distance = 0
        while forward_queue and backward_queue:
            if len(forward_queue) < len(backward_queue):
                distance += 1
                if self.extend_queue(forward_queue, forward_set, backward_set, grid):
                    return distance
                    
                distance += 1
                if self.extend_queue(backward_queue, backward_set, forward_set, grid):
                    return distance
            else:
                distance += 1
                if self.extend_queue(backward_queue, backward_set, forward_set, grid):
                    return distance       
                             
                distance += 1
                if self.extend_queue(forward_queue, forward_set, backward_set, grid):
                    return distance
        return -1
                
    def extend_queue(self, queue, visited, opposite_visited, grid):
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                new_x, new_y = (x + dx, y + dy)
                if not self.is_valid(new_x, new_y, grid, visited):
                    continue
                if (new_x, new_y) in opposite_visited:
                    return True
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))
                
        return False
        
    def is_valid(self, x, y ,grid, visited):
        n, m = len(grid), len(grid[0])

        if 0 <= x < n and 0<= y < m and not grid[x][y] and not (x,y) in visited:
            return True
        return False

# version: two-A*
import heapq
from collections import deque
DIRECTIONS = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2),]
class Solution:
    def shortestPath(self, grid, src, dst):
        if not grid or not grid[0]:
            return -1
            
        n, m = len(grid), len(grid[0])
        if grid[dst.x][dst.y]:
            return -1
        if n * m == 1:
            return 0
        if src.x == dst.x and src.y == dst.y:
            return 0
        
        self.init(grid, src, dst)

        res =  self.searching()
        # print(self.extract_path())

        return res

    def init(self, grid, src, dst):
        self.grid = grid
        self.src = src
        self.dst = dst

        l0 = self.h_val(src.x, src.y, dst.x, dst.y)

        self.queue_forward = []
        self.parent_forward = {(src.x, src.y): (src.x, src.y)}
        self.visited_forward = {(src.x, src.y): 0}

        self.queue_backward = []
        self.parent_backward = {(dst.x, dst.y): (dst.x, dst.y)}
        self.visited_backward = {(dst.x, dst.y): 0}

        heapq.heappush(self.queue_forward, (l0, 0, src.x, src.y))
        heapq.heappush(self.queue_backward, (l0, 0, dst.x, dst.y))

        self.meet = None


    def searching(self):
        while self.queue_forward and self.queue_backward:
            if len(self.queue_forward) < len(self.queue_backward):
                self.extend_queue(self.queue_forward, self.visited_forward, self.visited_backward, self.parent_forward, self.grid)
                if self.meet is not None:
                    break
                    
                self.extend_queue(self.queue_backward, self.visited_backward, self.visited_forward, self.parent_backward, self.grid)
                if self.meet is not None:
                    break
            else:
                self.extend_queue(self.queue_backward, self.visited_backward, self.visited_forward, self.parent_backward, self.grid)
                if self.meet is not None:
                    break 
                             
                self.extend_queue(self.queue_forward, self.visited_forward, self.visited_backward, self.parent_forward, self.grid)
                if self.meet is not None:
                    break
        
        return -1 if self.meet is None else self.visited_backward[self.meet] + self.visited_forward[self.meet]


    def extend_queue(self, queue, visited, opposite_visited, parent, grid):
        _, dis, x, y = heapq.heappop(queue)
        for dx, dy in DIRECTIONS:
            nxt_x, nxt_y = (x + dx, y + dy)
            dis = visited[(x,y)] + 1

            if not self.is_valid(nxt_x, nxt_y, grid, visited):
                continue

            parent[(nxt_x, nxt_y)] = (x,y)
            visited[(nxt_x, nxt_y)] = dis

            if (nxt_x, nxt_y) in opposite_visited:
                self.meet = (nxt_x, nxt_y)
                return (nxt_x, nxt_y)

            h = self.h_val(x, y, nxt_x, nxt_y)                
            heapq.heappush(queue, (dis+h, dis , nxt_x, nxt_y))
                
        return None


    def is_valid(self, x, y ,grid, visited):
        n, m = len(grid), len(grid[0])

        if 0 <= x < n and 0<= y < m and not grid[x][y] and not (x,y) in visited:
            return True
        return False


    def h_val(self, x0, y0, x1, y1):
        return (abs(x1-x0) + abs(y1-y0))/3


    def extract_path(self,):
        # extract path for foreward part
        path_fore = [self.meet]
        s = self.meet

        while True:
            s = self.parent_forward[s]
            path_fore.append(s)
            if s == self.src:
                break

        # extract path for backward part
        path_back = []
        s = self.meet

        while True:
            s = self.parent_backward[s]
            path_back.append(s)
            if s == self.dst:
                break

        return list(reversed(path_fore)) + list(path_back)



"""拓扑排序"""
# 127. 拓扑排序
# [2020年11月2日 2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/topological-sorting/description
# https://www.jiuzhang.com/solutions/topological-sorting/#tag-lang-python
class Solution:
    def topSort(self, graph):
        node_to_indegree = self.get_indegree(graph)

        start_nodes = [n for n in graph if node_to_indegree[n] == 0]
        order, queue = [], collections.deque(start_nodes)

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] -= 1
                if node_to_indegree[neighbor] == 0:
                    queue.append(neighbor)
                
        return order
    
    def get_indegree(self, graph):
        node_to_indegree = {x: 0 for x in graph}

        for node in graph:
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] += 1
                
        return node_to_indegree


# 616/210. 安排课程
# [2020年11月2日 2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/course-schedule-ii/description
# https://www.jiuzhang.com/solution/course-schedule-ii/#tag-lang-python
from collections import defaultdict
class Solution:
    def findOrder(self, numCourses, prerequisites):
        graph = defaultdict(list)
        indegree = {}
        order = []
        
        for dst, src in prerequisites:
            graph[src].append(dst)
            indegree[dst] = indegree.get(dst, 0) + 1
        
        zero_degree_queue = [n for n in range(numCourses) if n not in indegree ]

        while zero_degree_queue:
            cur = zero_degree_queue.pop()
            order.append(cur)

            for nxt in graph[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    zero_degree_queue.append(nxt)
        
        return order if len(order) == numCourses else []


# 615. 课程表
# [2020年11月2日 2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/course-schedule/description
# https://www.jiuzhang.com/solution/course-schedule/#tag-lang-python
class Solution:
    def canFinish(self, numCourses, prerequisites):
        adj_lst, indegree, order = collections.defaultdict(list), {}, []
        
        for course, pre in prerequisites:
            indegree[course] = indegree.get(course, 0) + 1
            adj_lst[pre].append(course)
        
        zero_indgree_queue = [ x for x in range(numCourses) if x not in indegree]
        
        while zero_indgree_queue:
            pre = zero_indgree_queue.pop()
            order.append(pre)

            for course in adj_lst[pre]:
                indegree[course] -= 1
                if indegree[course] == 0:
                    zero_indgree_queue.append(course)
                    
        return len(order) == numCourses


# 892. 外星人词典 ⭐⭐
# [2020年11月2日 2021年8月2日 2021年8月22日]
# https://www.lintcode.com/problem/alien-dictionary/description
# https://www.jiuzhang.com/solution/alien-dictionary/#tag-lang-python
from heapq import heapify, heappop, heappush
class Solution:
    def alienOrder(self, words):
        graph = self.build_graph(words)
        if graph is None:
            return ""

        return self.topological_sort(graph)

    
    def build_graph(self, words):
        graph = {}
        for w in words:
            for c in w:
                graph[c] = set()

        for i in range(len(words)-1):
            j_min = min(len(words[i]), len(words[i+1]))
            for j in range(j_min):
                if words[i][j] != words[i+1][j]:
                    graph[words[i][j]].add( words[i+1][j] )
                    break

                if j ==j_min - 1 and len(words[i]) > len(words[i+1]):
                    return None

        return graph


    def topological_sort(self, graph):
        indegree = {node: 0 for node in graph}
        for node in graph:
            for nxt in graph[node]:
                indegree[nxt] += 1
        
        queue = [node for node in indegree if indegree[node] == 0]
        heapify(queue)

        topo_order = ''
        while queue:
            cur = heappop(queue)
            topo_order += cur

            for nxt in graph[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heappush(queue, nxt)
        
        return topo_order if len(topo_order) == len(graph) else ""


# 605. 序列重构
# https://www.lintcode.com/problem/sequence-reconstruction/description
# https://www.jiuzhang.com/solution/sequence-reconstruction/#tag-lang-python
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        graph = self.build_graph(seqs)
        topo_order = self.topological_sort(graph)
        return topo_order == org
            
    def build_graph(self, seqs):
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
        
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i - 1]].add(seq[i])

        return graph
    
    def get_indegrees(self, graph):
        indegrees = {
            node: 0
            for node in graph
        }
        
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1
                
        return indegrees
        
    def topological_sort(self, graph):
        indegrees = self.get_indegrees(graph)
        
        queue = []
        for node in graph:
            if indegrees[node] == 0:
                queue.append(node)
        
        topo_order = []
        while queue:
            if len(queue) > 1:
                # there must exist more than one topo orders
                return None
                
            node = queue.pop()
            topo_order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(topo_order) == len(graph):
            return topo_order
            
        return None
        


"""
ladder exercise
"""
# 941. 滑动拼图
# https://www.lintcode.com/problem/sliding-puzzle/description
# https://www.jiuzhang.com/solution/sliding-puzzle/#tag-lang-python
from collections import deque 

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
class Solution:
    def slidingPuzzle(self, board) -> int:
        m, n = len(board), len(board[0])
        init_state = [ ['' for _ in range(n)] for _ in range(m) ] 

        for i in range(len(board)):
            for j in range(len(board[0])):
                init_state[i][j] = str(board[i][j])
        
        target_state = [['1', '2', '3'], ['4', '5', '0']]
        source = self.matrix_to_string(init_state)
        target = self.matrix_to_string(target_state)
        
        queue = deque()
        distance = {} 
        
        queue.append(source)
        distance[source] = 0 
        
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                if node == target:
                    return distance[node]
                
                for neighbor in self.get_neighbors(node):
                    if neighbor in distance:
                        continue 
                        
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1 
        return -1 
        
    def matrix_to_string(self, matrix):
        m, n = len(matrix), len(matrix[0])
        seq = [] 
        for i in range(m):
            for j in range(n):
                seq.append(matrix[i][j])
        return "".join(seq)
    
    def get_neighbors(self, state):
        zeroIndex = state.find('0')
        x, y = zeroIndex // 3, zeroIndex % 3 
        neighbors = [] 
        
        for dx, dy in DIRECTIONS:
            seq = list(state)
            nx, ny = x + dx, y + dy 
            if nx < 0 or nx >= 2 or ny < 0 or ny >= 3:
                continue 
            seq[zeroIndex], seq[nx * 3 + ny] = seq[nx * 3 + ny], seq[zeroIndex]
            neighbor = "".join(seq)
            neighbors.append(neighbor)
        return neighbors


# 794. 滑动拼图 II # TODO
# https://www.lintcode.com/problem/sliding-puzzle-ii/description?_from=ladder&&fromId=161
# https://www.jiuzhang.com/solution/sliding-puzzle-ii/#tag-lang-python
# DESC 细节1: 每一个状态是一个矩阵，矩阵使用二维List表示， List是可变类型， 不可加入set或作为dict的key. 解决办法：将矩阵转化为一个字符串
# DESC 细节2: 由于将矩阵转化为字符串，如何求neighbors? 解决办法：首先找到’0‘在字符串中的index, 记作zeroIndex, 再将zeroIndex转化为其在矩阵中的坐标，记作x, y 根据x, y 可以得知“0”可以和四个方向的元素进行交换， 从而求出neighbor
# version 1
from collections import deque 
DIRECTIONS = [ (0, 1), (0, -1), (1, 0), (-1, 0)  ]
class Solution:
    """
    @param init_state: the initial state of chessboard
    @param final_state: the final state of chessboard
    @return: return an integer, denote the number of minimum moving
    """
    def minMoveStep(self, init_state, final_state):
        source = self.matrix_to_string(init_state)
        target = self.matrix_to_string(final_state)
        
        queue = deque()
        distance = {} 
        
        queue.append(source)
        distance[source] = 0 
        
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                if node == target:
                    return distance[node]

                for neighbor in self.get_neighbors(node):
                    if neighbor in distance:
                        continue 
                    
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1 
        return -1 
        
    def matrix_to_string(self, matrix):
        seq = [] 
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                c = str(matrix[i][j])
                seq.append(c)
        return "".join(seq)
        
        
    def get_neighbors(self, seq):
        neighbors = [] 
        zeroIndex = seq.find("0")

        x = zeroIndex // 3 
        y = zeroIndex % 3 
        
        for dx, dy in DIRECTIONS:
            seq_list = list(seq)
            nx, ny = x + dx, y + dy 
            
            if nx < 0 or nx >= 3 or ny < 0 or ny >= 3:
                continue 
            
            seq_list[zeroIndex], seq_list[nx * 3 + ny] = seq_list[nx * 3 + ny], seq_list[zeroIndex]
            neighbor = "".join(seq_list)
            neighbors.append(neighbor)
        return neighbors

# version: A*
import heapq
class Solution:
    """
    @param init_state: the initial state of chessboard
    @param final_state: the final state of chessboard
    @return: return an integer, denote the number of minimum moving
    """
    def minMoveStep(self, init_state, final_state):
        start, end = [], []
        
        for i in range(3):
            for j in range(3):
                start.append(str(init_state[i][j]))
                end.append(str(final_state[i][j]))

        diff  = self.compare(start, end)    
        queue = [(diff, start)]
        dist  = {''.join(start): 0}

        while queue:
            _, state = heapq.heappop(queue)
            key = ''.join(state)

            for cur in self.findNeighbor(state):
                if cur == end:
                    return dist[key] + 1
                
                cur_key = ''.join(cur)
                if cur_key not in dist:
                    diff = self.compare(cur, end)
                    dist[cur_key] = dist[key] + 1
                    heapq.heappush(queue, (dist[cur_key] + diff, cur) )
        return -1
    
    def findNeighbor(self, state):
        pos = state.index("0")
        res = []
        if pos % 3 != 0:
            temp = state[:]
            temp[pos],temp[pos - 1] = temp[pos - 1],temp[pos]
            res.append(temp)
        if pos % 3 != 2:
            temp = state[:]
            temp[pos],temp[pos + 1] = temp[pos + 1],temp[pos]
            res.append(temp)
        if pos // 3 > 0:
            temp = state[:]
            temp[pos],temp[pos - 3] = temp[pos - 3],temp[pos]
            res.append(temp)
        if pos // 3 < 2:
            temp = state[:]
            temp[pos],temp[pos + 3] = temp[pos + 3],temp[pos]
            res.append(temp)
        return res
        
    def compare(self, s1, s2):
        count = 0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                count += 1
        return count
        
    

# 1179. 朋友圈
# https://www.lintcode.com/problem/friend-circles/description?_from=ladder&&fromId=161
# https://www.jiuzhang.com/solution/friend-circles/#tag-lang-python
# version: BFS
from collections import deque
class Solution:
    """
    @param M: a matrix
    @return: the total number of friend circles among all the students
    """
    def BFS(self, student, M):
        queue = deque([student])

        while queue:
            nxt_queue = deque([])
            for _ in range(len(queue)):
                j = queue.popleft()
                M[j][j] = 2
                for k in range(0, len(M[0])):	#遍历朋友关系
                    if M[j][k] == 1 and M[k][k] == 1:	#如果M[k][k]==1，说明k没被遍历，需要继续搜索
                        nxt_queue.append(k)
            queue = nxt_queue

    def findCircleNum(self, M):
        count = 0
        for i in range(0, len(M)):
            if M[i][i] == 1 :	
                count += 1	
                self.BFS(i, M) 
        return count

# version: dfs
class Solution:
    def dfs(self, x, M, visisted):
        for i in range(len(M)):
            if (M[x][i] == 1 and visisted[i] == False):
                visisted[i] = True
                self.dfs(i, M, visisted)

    def begindfs(self, M):
        n = len(M)
        ans = 0
        visisted = {}
        for i in range(n):
            visisted[i] = False

        for i in range(n):
            if (visisted[i] == False):
                ans += 1
                visisted[i] = True
                self.dfs(i, M, visisted)
        return ans

    def findCircleNum(self, M):
        # Write your code here
        ansdfs = self.begindfs(M)
        return ansdfs

# version: Union find
class UnionFind:
    def __init__(self, n):
        self.father = { i : i for i in range(n)}
        self.count = n 
        
    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        
        if root_a != root_b:
            self.father[root_b] = root_a 
            self.count -= 1 
            
    def find(self, point):
        path = [] 
        while point != self.father[point]:
            path.append(point)
            point = self.father[point]
            
        for p in path:
            self.father[p] = point 
        return point 

class Solution:
    def findCircleNum(self, M):
        if not M or not M[0]:
            return 0 
        
        n = len(M)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    uf.union(i, j)
        return uf.count


# 787. 迷宫
# https://www.lintcode.com/problem/the-maze/description?_from=ladder&&fromId=161
# https://www.jiuzhang.com/solution/the-maze/#tag-lang-python
# DESC： determine whether the ball could stop at the destination
import heapq
class Solution:
    """
    @param maze: the maze
    @param start: the start
    @param destination: the destination
    @return: whether the ball could stop at the destination
    """
    def hasPath(self, maze, start, destination):
        # write your code here
        if start == destination:
             return False

        heap    = [ (0, start[0], start[1])]
        visited = set([(start[0], start[1])])
        while heap:
            cur_len, cur_x, cur_y = heapq.heappop(heap)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt_x, nxt_y, steps = self.get_next(maze, cur_x, cur_y, dx, dy)
                
                if [nxt_x, nxt_y] == destination:
                    return True
                
                if (nxt_x, nxt_y) not in visited:
                    heapq.heappush(heap, (cur_len + steps, nxt_x, nxt_y))
                    visited.add((nxt_x, nxt_y))
        return False
        
    def get_next(self, maze, i, j, dx, dy):
        cur_x, cur_y = i, j
        while 0 <= i < len(maze) and 0 <= j < len(maze[0]) and maze[i][j] == 0:
            i, j = i + dx, j + dy
        return i - dx, j - dy, abs(i - dx - cur_x) + abs(j - dy - cur_y)


# 788. 迷宫II
# https://www.lintcode.com/problem/the-maze-ii/description?_from=ladder&&fromId=161
# https://www.jiuzhang.com/solution/the-maze-ii/#tag-lang-python
# version BFS
import heapq
class Solution:
    def shortestDistance(self, maze, start, destination):
        if start == destination:
             return 0

        heap    = [ (0, start[0], start[1])]
        visited = set([(start[0], start[1])])
        while heap:
            cur_len, cur_x, cur_y = heapq.heappop(heap)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt_x, nxt_y, steps = self.get_next(maze, cur_x, cur_y, dx, dy)
                
                if [nxt_x, nxt_y] == destination:
                    return cur_len + steps
                
                if (nxt_x, nxt_y) not in visited:
                    heapq.heappush(heap, (cur_len + steps, nxt_x, nxt_y))
                    visited.add((nxt_x, nxt_y))
        return -1
        
    def get_next(self, maze, i, j, dx, dy):
        cur_x, cur_y = i, j
        while 0 <= i < len(maze) and 0 <= j < len(maze[0]) and maze[i][j] == 0:
            i, j = i + dx, j + dy
        return i - dx, j - dy, abs(i - dx - cur_x) + abs(j - dy - cur_y)
# version DFS
class Solution:
    """
    @param maze: the maze
    @param start: the start
    @param destination: the destination
    @return: the shortest distance for the ball to stop at the destination
    """
    def shortestDistance(self, maze, start, destination):
        self.m, self.n = len(maze), len(maze[0])
        self.paths = []
        visited = set([(start[0], start[1])])
        
        self.dfs(maze, start, destination, 0, visited)
        if len(self.paths) == 0:
            return -1
        return min(self.paths)
    
    def dfs(self, maze, start, destination, step, visited):
        if start[0] == destination[0] and start[1] == destination[1]:
            self.paths.append(step)
            return
        
        for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = start[0], start[1]
            # for each move need to reset the step
            starting_step = step
            while (x >= 0 and x < self.m and y >= 0 and y < self.n and maze[x][y] != 1):
                x += move[0]
                y += move[1]
                starting_step += 1

            x -= move[0]
            y -= move[1]
            starting_step -= 1
            
            if (x, y) not in visited:
                visited.add((x, y))
                self.dfs(maze, [x, y], destination, starting_step, visited)
                visited.remove((x, y))


# 624. 移除子串
# https://www.lintcode.com/problem/remove-substrings/description?_from=ladder&&fromId=161
# https://www.jiuzhang.com/solution/remove-substrings/#tag-lang-python
class Solution:
    """
    @param: s: a string
    @param: dict: a set of n substrings
    @return: the minimum length
    """
    def minLength(self, s, dict):
        queue = collections.deque([s])
        visited = set([s])
        answer = len(s)

        while queue:
            cur_str = queue.popleft()
            answer = min(answer, len(cur_str))
            for sub_str in self.find_substrings(cur_str, dict):
                if sub_str not in visited:
                    visited.add(sub_str)
                    queue.append(sub_str)
        return answer
        
    def find_substrings(self, s, dict):
        results = []
        for word in dict:
            found = s.find(word)
            while found != -1:
                substring = s[:found] + s[found + len(word):]
                results.append(substring)
                found = s.find(word, found + 1) # searching start from found + 1
        return results

class Solution:
    def minLength(self, s, word_dict):
        # write your code here
        visited = set([s])
        self.min_len = len(s)
        self.dfs(s, word_dict, visited)
        return self.min_len

    def dfs(self, s, word_dict, visited):
        if s == '':
            return 0
            
        all_next = []
        # Find all substrings after remove one item in word_dict
        for item in word_dict:
            pos = s.find(item, 0)
            while pos != -1:
                all_next.append(s[: pos] + s[pos + len(item): ])
                pos = s.find(item, pos + len(item))
        
        for next_s in all_next:
            if next_s in visited:
                continue
            self.min_len = min(self.min_len, len(next_s))
            visited.add(next_s)
            self.dfs(next_s, word_dict, visited)



# 531. 六度问题
# https://www.lintcode.com/problem/six-degrees/description?_from=ladder&&fromId=161
# https://www.jiuzhang.com/solution/six-degrees/#tag-lang-python
from  collections import deque
class Solution:
    '''
    @param {UndirectedGraphNode[]} graph a list of Undirected graph node
    @param {UndirectedGraphNode} s, t two Undirected graph nodes
    @return {int} an integer
    '''
    def sixDegrees(self, graph, s, t):
        dis = {}
        queue = deque([s])
        dis[s] = 0

        while queue:
            x = queue.popleft()
            if x == t:
                return dis[x]

            for y in x.neighbors:
                if y not in dis:
                    dis[y] = dis[x] + 1
                    queue.append(y)
        return -1
# version: two BFS
class Solution:
    def sixDegrees(self, graph, s, t):
        degree = 0
        if s is t: return degree 
        
        visitedS, visitedT = {s}, {t}
        queueS = collections.deque([s])
        queueT = collections.deque([t])
        
        while queueS and queueT:
            degree += 1 
            for _ in range(len(queueS)):
                node = queueS.popleft()
                for neighbor in node.neighbors:
                    if neighbor in visitedT:
                        return degree 
                    if neighbor in visitedS:
                        continue
                    visitedS.add(neighbor)
                    queueS.append(neighbor)
                
            degree += 1 
            for _ in range(len(queueT)):
                node = queueT.popleft()
                for neighbor in node.neighbors:
                    if neighbor in visitedS:
                        return degree 
                    if neighbor in visitedT:
                        continue
                    visitedT.add(neighbor)
                    queueT.append(neighbor)
                
        return -1


# 1029· 寻找最便宜的航行旅途（最多经过k个中转站）
# [2021年4月8日]
# https://www.lintcode.com/problem/1029/
from heapq import heappop, heappush
class Solution:
    def build_graph(self, flights):
        graph = {}
        for start, end, cost in flights:
            if start not in graph:
                graph[start] = [(cost, end)]
            else:
                graph[start].append( (cost, end) )

        return graph

    def findCheapestPrice(self, n, flights, src, dst, K):
        """
            @param n: a integer
            @param flights: a 2D array
            @param src: a integer
            @param dst: a integer
            @param K: a integer
            @return: return a integer
        """
        graph = self.build_graph(flights)
        if src not in graph: return -1

        heap = []
        for cost, nxt in graph[src]:
            heappush(heap, (cost, nxt, 0))
        
        while heap:
            cost, cur, level = heappop(heap)
            if level > K:
                continue
            if cur == dst:
                return cost
            
            if cur in graph:
                for nxt_cost, nxt_stop in graph[cur]:
                    heappush(heap, (cost+nxt_cost, nxt_stop, level+1))
        
        return -1

import heapq
class Solution:
    def networkDelayTime(self, times, N, K):
        if not times: return 0
            
        graph = collections.defaultdict(list)
        for s, e, cost in times:
            graph[s].append((e, cost))

        dist, seen = {}, set()
        queue = [(0, K)]
        
        while queue:
            cost, cur = heapq.heappop(queue)
            if cur in seen: continue
            
            dist[cur] = cost
            seen.add(cur)
            
            for nxt, time in graph[cur]:
                if nxt in seen:
                    continue
                
                if nxt in dist and dist[nxt] < time + cost:
                    continue
                
                dist[nxt] = time + cost
                heapq.heappush(queue, (time + cost, nxt))
                
        if len(dist) != N:
            return -1
            
        return max(dist.values())