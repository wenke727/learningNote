'''
Union find

* Template:
class ConnectingGraph:
    def __init__(self, n):
        self.father = {}
        for i in range(n + 1):
            self.father[i] = i

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            self.father[roota] = rootb

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]

* Trie Tree

from collections import OrderedDict
class TrieNode:
    def __init__(self):
        self.children = OrderedDict()
        self.isWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for letter in word:
            node.children[letter] = node.children.get(letter, TrieNode())
            node = node.children[letter]
        node.isWord = True

    def search(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                return False
            node = node.children[letter]
        
        return node.isWord

    def startsWith(self, prefix):
        node = self.root
        for letter in prefix:
            if letter not in node.children:
                return False
            node = node.children[letter]
        
        return True

'''
# 589. 连接图  
# [2020年11月8日 2021年2月22日 2021年8月11日 2021年8月23日]
# https://www.lintcode.com/problem/connecting-graph/description
# https://www.jiuzhang.com/solutions/connecting-graph/#tag-lang-python
class ConnectingGraph:
    def __init__(self, n):
        self.father = {}
        for i in range(n + 1):
            self.father[i] = i

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            self.father[roota] = rootb

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]


# 590. 连接图 II  
# [2020年11月9日 2021年2月22日 2021年8月11日]
# https://www.lintcode.com/problem/connecting-graph-ii/description
# https://www.jiuzhang.com/solutions/connecting-graph-ii/#tag-lang-python
class ConnectingGraph2:
    """
    有三处错误： 1） nodenum合并；2）query访问的是根节点的值，而不是当前节点的值；3）find 函数是dfs，需要有返回
    """
    def __init__(self, n):
        self.father  = {}
        self.nodenum = {}
        for i in range(n + 1):
            self.father[i] = i
            self.nodenum[i] = 1 

    def connect(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            # Caution: The father is root_b not root_a. self.nodenum[root_a] += self.nodenum[root_b]
            self.nodenum[root_b] += self.nodenum[root_a]

    def query(self, a):
        """
        return: 图中含 a 的联通区域内节点个数
        """
        return self.nodenum[self.find(a)]

    def find(self,x):
        if self.father[x] == x:
            return x
        
        self.father[x] = self.find( self.father[x] )
        return self.father[x]

    def find2(self,x):
        x2 = x
        if self.father[x] == x:
            return x
        
        while self.father[x] != x:
            x = self.father[x]

        while x2 != x:
            temp = self.father[x2]
            self.father[x2] = x
            x2 = temp
        
        return x


# 591. 连接图 III 
# [2020年11月9日 2021年2月22日 2021年8月11日]
# https://www.lintcode.com/problem/connecting-graph-iii/description
# https://www.jiuzhang.com/solutions/connecting-graph-iii/#tag-lang-python
class ConnectingGraph3:
    def __init__(self, n):
        self.father = {}
        self.count = n
        for i in range(n+1):
            self.father[ i ] = i


    def connect(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a  != root_b:
            self.father[root_a] = root_b
            self.count -= 1


    def query(self):
        return self.count


    def find(self,x):
        if self.father[x] == x:
            return x
        
        self.father[x] = self.find( self.father[x] )
        return self.father[x]


# 433/200. 岛屿的个数
# [2020年11月9日 2021年2月22日 2021年8月11日]
# https://www.lintcode.com/problem/number-of-islands/; https://leetcode-cn.com/problems/number-of-islands/
# https://www.jiuzhang.com/solutions/number-of-islands/#tag-lang-python
class UnionFind:
    def __init__(self, n):
        self.father = {}
        self.ans = 0
        for i in range(n):
            self.father[i] = i

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            self.father[roota] = rootb
            self.ans -= 1

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]

class Solution:
    def numIslands(self, grid):
        if not grid or not grid[0]: 
            return 0

        n, m = len(grid), len(grid[0])
        uf = UnionFind(n*m)

        for i in range(n):
            for j in range(m):
                if not grid[i][j]:
                    continue
                uf.ans += 1

                if i + 1 < n and grid[i+1][j]:
                    uf.connect( i*m + j, (i+1)*m + j )
                if j +1 < m and grid[i][j+1]:
                    uf.connect( i*m + j, i*m + (j+1) )
        
        return uf.ans                


# 434. 岛屿的个数II 
# [2020年11月9日 2021年2月22日 2021年8月11日]
# https://www.lintcode.com/problem/number-of-islands-ii/description
# https://www.jiuzhang.com/solutions/number-of-islands-ii/#tag-lang-python
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class UnionFind:
    def __init__(self, n):
        self.father = {}
        self.ans = 0
        for i in range(n):
            self.father[i] = i

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            self.father[roota] = rootb
            self.ans -= 1

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]

class Solution:
    def numIslands2(self, n, m, operators):
        res, island = [], set()
        uf = UnionFind(n*m)

        for op in operators:
            x, y = op.x, op.y
            if (x, y) in island:
                res.append(uf.ans)
                continue
            
            island.add((x, y))
            uf.ans += 1

            for dx, dy in DIRECTIONS:
                nxt_x, nxt_y = x+dx, y+dy
                if (nxt_x, nxt_y) in island:
                    uf.connect(x*m+y, nxt_x*m+nxt_y)
            
            res.append(uf.ans)

        return res


# 178/261. 图是否是树 
# [2020年11月9日 2021年2月23日 2021年8月11日]
# https://www.lintcode.com/problem/graph-valid-tree/description; https://leetcode-cn.com/problems/graph-valid-tree/
# https://www.jiuzhang.com/solutions/graph-valid-tree/#tag-lang-python
class UnionFind:
    def __init__(self, n):
        self.father = {}
        self.ans = 0
        for i in range(n):
            self.father[i] = i

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            self.father[roota] = rootb
            self.ans -= 1

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]

class Solution:
    def validTree(self, n, edges):
        if len(edges) != n-1:
            return False

        uf = UnionFind(len(edges)+1)
        uf.ans = len(edges)+1

        for a, b in edges:
            uf.connect(a, b)

        return uf.ans == 1


# 477. 被围绕的区域 ⭐
# [2020年11月9日 2021年2月23日 2021年8月11日]
# https://www.lintcode.com/problem/surrounded-regions/description https://leetcode-cn.com/problems/surrounded-regions/
# https://www.jiuzhang.com/solutions/surrounded-regions/#tag-lang-python
# DESC 给一个二维的矩阵，包含 'X' 和 'O', 找到所有被 'X' 围绕的区域，并用 'X' 替换其中所有的 'O'
# version: Union find
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class UnionFind:
    def __init__(self, n):
        self.father = {i:i for i in range(n)}

    def connect(self, a, b):
        roota, rootb = self.find(a), self.find(b)
        if roota != rootb:
            # cautions
            self.father[min(roota, rootb)] = max(roota, rootb)

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self,x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        return self.father[x]

class Solution:
    def surroundedRegions(self, board):
        if not board or not board[0]:
          return []
          
        m, n = len(board), len(board[0])
        if m <= 2 or n <=2: 
            return board

        uf = UnionFind(m*n+1)
        dummy = m*n

        for i in range(m):
            for j in range(n):
                if board[i][j] == 'X':
                    continue

                # case: O
                if i in (0, m-1) or j in (0, n-1):
                    uf.connect(i*n+j, dummy)
                
                for dx, dy in DIRECTIONS:
                    nxt_x, nxt_y = i + dx, j + dy
                    if not ( 0<nxt_x<m and 0<nxt_y<n ):
                        continue
                    if board[nxt_x][nxt_y] == 'O':
                        uf.connect(i*n + j, nxt_x*n + nxt_y ) 

        for x in range(m):
            for y in range(n):
                if board[x][y] == "O" and uf.find(x * n + y) != dummy:
                    board[x][y] = "X"
# version: 高频题班
class Solution:
    def surroundedRegions(self, board):
        if not any(board): 
            return

        n, m = len(board), len(board[0])
        # obtian the boundary of board
        queue = [ ij for k in range(max(n,m)) for ij in ((0, k), (n-1, k), (k, 0), (k, m-1))]
        
        while queue:
            i, j = queue.pop()
            if 0 <= i < n and 0 <= j < m and board[i][j] == 'O':
                # print( i, j, board[i][j] )
                board[i][j] = 'W'
                queue += (i, j-1), (i, j+1), (i-1, j), (i+1, j)

        board[:] = [['XO'[c == 'W'] for c in row] for row in board]


# 1070/721. 账户合并 ⭐⭐
# [2020年10月21日 2020年11月9日 2021年2月23日 2021年8月11日]
# https://www.lintcode.com/problem/accounts-merge/description
# https://www.jiuzhang.com/solution/accounts-merge/#tag-lang-python
# DESC: 1) emails_to_ids; 2) ids union; 3) get_id_to_mail(key: find root user id)
class UnionFind:
    def __init__(self, n):
        self.father = {i:i for i in range(n)}

    def find(self, x):
        if self.father[x] == x:
            return x

        self.father[x] = self.find(self.father[x])
        
        return self.father[x]

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)

        if root_a != root_b:
            self.father[root_a] = root_b

class Solution:
    def accountsMerge(self, accounts):
        uf = UnionFind(len(accounts))

        email_to_ids = self.get_email_to_ids(accounts)

        for _, ids in email_to_ids.items():
            root_id = ids[0]
            for id in ids[1:]:
                uf.union(id, root_id)

        id_to_emails = self.get_id_to_email_set(accounts, uf)

        merge_accounts = []
        for user_id, emails in id_to_emails.items():
            merge_accounts.append( 
                [ accounts[user_id][0], *sorted(emails) ]
             )
        
        return merge_accounts

    def get_id_to_email_set(self, accounts, uf):
        id_to_email_set = {}

        for user_id, record in enumerate( accounts ):
            root_user = uf.find(user_id) # ! key point
            id_to_email_set[root_user] = id_to_email_set.get(root_user, set())

            for email in record[1:]:
                id_to_email_set[root_user].add(email)
        
        return id_to_email_set

    def get_email_to_ids(self, accounts):
        email_to_ids = {}
        for user_id, record in enumerate( accounts ):
            for email in record[1:]:
                email_to_ids[email] = email_to_ids.get( email, [] )
                email_to_ids[email].append(user_id)

        return email_to_ids       


# 629. 最小生成树 ⭐⭐⭐
# [2021年2月23日 2021年8月11日  2021年8月23日]
# https://www.lintcode.com/problem/minimum-spanning-tree/description
# functools.cmp_to_key: 两个参数并比较它们，结果为小于则返回一个负数，相等则返回零，大于则返回一个正数。key function则是一个接受一个参数，并返回另一个用以排序的值的可调用对象
import functools
class Connection:
    def __init__(self, city1, city2, cost):
        self.city1, self.city2, self.cost = city1, city2, cost
def cmp(a, b):
    if a.cost != b.cost:
        return a.cost - b.cost
    
    if a.city1 != b.city1:
        if a.city1 > b.city1:
            return 1
        
        return -1 
    
    if a.city2 > b.city2:
        return 1
    
    return -1

class UnionFind:
    def __init__(self, n) -> None:
        self.father = {i:i for i in range(n)}
    
    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.father[root_b] = root_a

            return True
        
        return False
    
    def find(self, x):
        if self.father[x] == x:
            return x
        
        self.father[x] = self.find(self.father[x])

        return self.father[x]

class Solution:
    def lowestCost(self, connections):
        connections.sort(key=functools.cmp_to_key(cmp))

        graph, _size = self.create_city_graph(connections)
        uf = UnionFind(_size)
        res = []

        for i in connections:
            # if it was connected, the `union` function here would return Fasle
            if uf.union(graph[i.city1], graph[i.city2]):
                res.append(i)
        
        root = uf.find(0)
        for i in range(_size):
            if uf.find(i) != root:
                return []

        return res
    
    
    def create_city_graph(self, connections):
        graph, _size = {}, 0
        for i in connections:
            for p in  [i.city1, i.city2]:
                if p in graph:
                    continue
                graph[p] = _size
                _size += 1
        
        return graph, _size
            

    def printConnections(self, lst):
        for c in lst:
            print( f"[{c.city1}, {c.city2}]: {c.cost}" )


"""Trie"""
# 442/208. 实现 Trie（前缀树）
# [2020年11月9日 2021年2月23日 2021年2月24日 2021年8月11日]
# https://www.lintcode.com/problem/implement-trie-prefix-tree/description https://leetcode-cn.com/problems/implement-trie-prefix-tree/
from collections import OrderedDict
class TrieNode:
    def __init__(self):
        self.children = OrderedDict()
        self.isWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for c in word:
            node.children[c]  = node.children.get(c, TrieNode())
            node = node.children[c] 
        node.isWord = True
    
    def search(self, word):
        node = self.root
        for c in word:
            node = node.children.get(c)
            if node is None:
                return False
        return node.isWord

    def startsWith(self, prefix):
        node = self.root
        for c in prefix:
            node = node.children.get(c)
            if node is None:
                return False
        return True


# 473/211. 单词的添加与查找 
# [2020年11月10日 2021年2月23日 2021年8月11日]
# https://www.lintcode.com/problem/add-and-search-word-data-structure-design/description https://leetcode-cn.com/problems/implement-trie-prefix-tree/
from collections import OrderedDict
class TrieNode:
    def __init__(self):
        self.children = OrderedDict()
        self.isWord = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.isWord = True

    def search(self, word):
        if word is None: 
            return False
        
        return self.dfs_helper(self.root, word, 0)

    def dfs_helper(self, node, word, index):
        if node is None: 
            return False
        if index >= len(word): 
            return node.isWord 

        letter = word[index]
        if letter != '.': 
            return self.dfs_helper(node.children.get(letter), word, index+1)
        
        for i in node.children:
            if self.dfs_helper(node.children.get(i), word, index+1):
                return True
        
        return False


# 132/212. 单词搜索 II 
# [2021年2月23日 2021年8月11日]
# https://www.lintcode.com/problem/word-search-ii/description
# version hashmap
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class Solution:
    def findWords(self, board, words):
        if not board or not board[0]: 
            return []

        self.word_set, self.prefix_set = set(words), set()
        for word in words:
            for i in range(len(word)):
                self.prefix_set.add( word[:i+1] )

        result = set()
        self.m, self.n = len(board), len(board[0])
        for i in range(self.m):
            for j in range(self.n):
                self.dfs(board, i, j, board[i][j], set([(i,j)]), result)

        return list(result)


    def dfs(self, board, x, y, word, visited, result):
        if word not in self.prefix_set: 
            return

        if word in self.word_set:
            result.add(word)

        for dx, dy in DIRECTIONS:
            nxt_x, nxt_y = x+dx, y+dy
            if not self.isValid(nxt_x, nxt_y) or (nxt_x, nxt_y) in visited:
                continue
            
            visited.add((nxt_x, nxt_y))
            self.dfs(board, nxt_x, nxt_y, word + board[nxt_x][nxt_y], visited, result)
            visited.remove((nxt_x, nxt_y))
    
    def isValid(self, x, y):
        return 0 <= x < self.m and 0 <= y < self.n
        
# version 用Trie 版本 # find the difference
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None

class Trie():
    def __init__(self) -> None:
        self.root = TrieNode()
    
    def add(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        
        node.is_word = True
        node.word = word
    
    def find(self, word):
        node = self.root
        for letter in word:
            if node.children.get(letter) is None:
                return None
        
        return node

class Solution:
    def wordSearchII(self, board, words):
        if board is None or not board[0]: 
            return []
        
        trie = Trie()
        for word in words:
            trie.add(word)
        
        res = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, 
                         i, 
                         j, 
                         trie.root.children.get(board[i][j]), 
                         set([(i,j)]), 
                         res
                )
        
        return list(res)

    def dfs(self, board, x, y, node, visited, result):
        if node is None:
            return
        
        if node.is_word:
            result.add(node.word)
        
        for dx, dy in DIRECTIONS:
            nxt_x, nxt_y = x+dx, y+dy
            if not self.inside(board, nxt_x, nxt_y) or (nxt_x, nxt_y) in visited:
                continue
        
            visited.add((nxt_x, nxt_y))
            self.dfs(board, 
                     nxt_x, 
                     nxt_y, 
                     node.children.get(board[nxt_x][nxt_y]), 
                     visited, 
                     result
            )
            visited.remove((nxt_x, nxt_y))

    def inside(self, board, x, y):
        return 0 <= x < len(board) and 0 <= y < len(board[0])


# 634/425. 单词矩阵 / 单词方块 ⭐⭐⭐
# https://www.lintcode.com/problem/word-squares/description
# https://www.jiuzhang.com/solution/word-squares/#tag-lang-python
# Version 1
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word_list = []

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def add(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.word_list.append(word)
        node.is_word = True

    def find(self, word):
        node = self.root
        for c in word:
            node = node.children.get(c)
            if node is None:
                return None
        return node
        
    def get_words_with_prefix(self, prefix):
        node = self.find(prefix)
        return [] if node is None else node.word_list
        
class Solution:
    def wordSquares(self, words):
        trie = Trie()
        for word in words:
            trie.add(word)
        
        results = []
        for word in words:
            self.search(trie, [word], results)
        
        return results
        
    def search(self, trie, combination, ressults):
        cur_row_index, n = len(combination), len(combination[0])

        if cur_row_index == n:
            ressults.append(list(combination))
            return
        
        # ! 剪枝 Pruning, it's ok to remove it, but will be slower
        for row_index in range(cur_row_index, n):
            prefix = ''.join([combination[i][row_index] for i in range(cur_row_index)])
            if trie.find(prefix) is None:
                return
        
        prefix = ''.join([combination[i][cur_row_index] for i in range(cur_row_index)])
        for word in trie.get_words_with_prefix(prefix):
            combination.append(word)
            self.search(trie, combination, ressults)
            combination.pop()

# Version 2
class Solution:
    def initPrefix(self, words):
        for word in words:
            if "" not in self.hash: self.hash[""] = []
            self.hash[""] += [str(word)] 
            
            prefix = ""
            for c in word:
                prefix += c 
                if prefix not in self.hash: self.hash[prefix] = []
                self.hash[prefix] += [str(word)]
                
    def checkPrefix(self, x, nextWord, l):
        for i in range(x + 1, l):
            prefix = ""
            for item in self.path:
                prefix += item[i]
            prefix += nextWord[i]
            if (prefix not in self.hash):
                return False 
        return True
    
    def dfs(self, x, l):
        if x == l:
            self.ans.append(list(self.path))
            return
        
        prefix = ""
        
        for item in self.path:
            prefix += item[x]
        
        for item in self.hash[prefix]:
            if not self.checkPrefix(x, item, l):
                continue
            
            self.path.append(item)
            self.dfs(x + 1, l)
            self.path.pop()
            
    def wordSquares(self, words):
        self.hash = {}
        self.path = []
        self.ans = []
        
        if not words:
            return self.ans
        
        self.initPrefix(words)
        self.dfs(0, len(words[0]))
        
        return self.ans


# 527. 序列化Trie
# https://www.lintcode.com/problem/trie-serialization/
class Solution:
    def serialize(self, root):
        if root is None:
            return ""

        data = ""
        for key, value in root.children.items():
            data += key + self.serialize(value)

        return "<%s>" % data

    def deserialize(self, data):
        if data is None or len(data) == 0:
            return None

        root = TrieNode()
        current = root
        path =[]
        for c in data:
            if c == '<':
                path.append(current)
            elif c == '>':
                path.pop()
            else:
                current = TrieNode()
                if len(path) == 0:
                    print( c, path)
                path[-1].children[c] = current


