# 10. 字符串的不同排列
# [2021年8月7日]
# https://www.lintcode.com/problem/string-permutation-ii/description
# https://www.jiuzhang.com/solutions/string-permutation-ii
class Solution:
    def stringPermutation2(self, s):
        chars = sorted(list(s))
        visited = [False] * len(chars)
        res = []
        self.dfs(chars, visited, [], res)

        return res
    
    def dfs(self, chars, visited, permutation, result):
        if len(chars) == len(permutation):
            result.append( "".join(permutation))
            return 
        
        for i in range(len(chars)):
            if visited[i]:
                continue

            # ! 不能跳过一个a选下一个a. a' a" b; => a' a" b => √; => a" a' b => x
            if i > 0 and chars[i] == chars[i-1] and not visited[i-1]:
                continue

            visited[i] = True
            permutation.append(chars[i])
            self.dfs(chars, visited, permutation, result)
            permutation.pop()
            visited[i] = False


# 33. N皇后问题
# [2021年8月8日]
# https://www.lintcode.com/problem/n-queens/description
# https://www.jiuzhang.com/solutions/n-queens/
class Solution:
    """
    @param: n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n: int):
        boards  = []
        # 用 visited 来标记 列号，横纵坐标之和，横纵坐标之差 有没有被用过
        visited = {'col': set(), 'sum': set(), 'diff': set(), }
        self.dfs( n, [], visited, boards )
        
        return boards
    
    def dfs(self, n, permutation, visited, boards):
        if n == len(permutation):
            boards.append(self.draw(permutation))
        
        row = len(permutation)
        for col in range(n):
            if not self.is_valid(permutation, visited, col):
                continue

            permutation.append(col)
            visited['col'].add(col)
            visited['sum'].add(col+row)
            visited['diff'].add(row-col)

            self.dfs(n, permutation, visited, boards)

            permutation.pop()
            visited['col'].remove(col)
            visited['sum'].remove(col+row)
            visited['diff'].remove(row-col)
    
    def is_valid(self, permutation, visited, col):
        row = len(permutation)
        if col in visited['col']: 
            return False
        if row + col in visited['sum']: 
            return False
        if row - col in visited['diff']: 
            return False
        
        return True
    
    def draw(self, permutation):
        board = []
        n = len(permutation)

        for col in permutation:
            row_string = ''.join( ['Q' if c == col else '.' for c in range(n)] )
            board.append(row_string)
        return board


# 52. 下一个排列 # TODO
# https://www.lintcode.com/problem/next-permutation/description
class Solution:
    def nextPermutation(self, num):
        for i in range(len(num)-2, -1, -1):
            if num[i] < num[i+1]:
                break
        else:
            num.reverse()
            return num
    
        for j in range(len(num)-1, i, -1):
            if num[j] > num[i]:
                num[i], num[j] = num[j], num[i]
                break
        
        for j in range(0, (len(num) - i)//2):
            num[i+j+1], num[len(num)-j-1] = num[len(num)-j-1], num[i+j+1]
        
        return num

# 190. 下一个排列
# https://www.lintcode.com/problem/next-permutation-ii/description

# 197. 排列序号
# https://www.lintcode.com/problem/permutation-index/description

# 198. 排列序号II
# https://www.lintcode.com/problem/permutation-index-ii/description


# 425. 电话号码的字母组合
# [2021年8月8日]
# https://www.lintcode.com/problem/letter-combinations-of-a-phone-number/description
# https://www.jiuzhang.com/solution/letter-combinations-of-a-phone-number//#tag-lang-python
KEYBOARD = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
class Solution:
    """
    @param digits: A digital string
    @return: all posible letter combinations
    """
    def letterCombinations(self, digits: str):
        if not digits:
            return []
        results = []
        self.dfs(digits, 0, [], results)
        
        return results
    
    def dfs(self, digits, index, chars, results):
        if index == len(digits):
            results.append(''.join(chars))
            return 

        for letter in KEYBOARD[digits[index]]:
            chars.append(letter)
            self.dfs(digits, index+1, chars, results)
            chars.pop()


# 828. 字模式
# [2021年8月8日]
# https://www.lintcode.com/problem/word-pattern/description
# https://www.jiuzhang.com/solutions/word-pattern/#tag-lang-python
class Solution:
    def wordPattern(self, pattern, teststr):
        words = teststr.split(" ")
        if len(words) != len(pattern):
            return False
            
        c_2_w, w_2_c = {}, {}
        for c, w in zip(pattern, words):
            if c in c_2_w and c_2_w[c] != w:
                return False
            c_2_w[c] = w
            
            if w in w_2_c and w_2_c[w] != c:
                return False
            w_2_c[w] = c
            
        return True


# 829. 字模式 II ⭐
# [2021年8月8日]
# https://www.lintcode.com/problem/word-pattern-ii/description
# https://www.jiuzhang.com/solutions/word-pattern-ii//#tag-lang-python
class Solution:
    """
    @param pattern: a string,denote pattern string
    @param str: a string, denote matching string
    @return: a boolean
    """
    def wordPatternMatch(self, pattern, string):
        return self.is_match(pattern, string, {}, set())

    def is_match(self, pattern, string, mapping, used):
        if not pattern:
            return not string
        
        char = pattern[0]
        if char in mapping:
            word = mapping[char]
            if not string.startswith(word):
                return False

            return self.is_match(pattern[1:], string[len(word):], mapping, used)
        
        for i in range(len(string)):
            word = string[:i+1]
            if word in used:
                continue
            
            used.add(word)
            mapping[char] = word
            
            if self.is_match(pattern[1:], string[i+1:], mapping, used):
                return True
            
            del mapping[char]
            used.remove(word)
        
        return False


# 120. 单词接龙
# https://www.lintcode.com/problem/word-ladder/description
# https://www.jiuzhang.com/solutions/word-ladder//#tag-lang-python
# DESC `chapter_04_BFS.py`


# 121. 单词接龙 II ⭐
# [2021年8月8日]
# https://www.lintcode.com/problem/word-ladder-ii/description
# https://www.jiuzhang.com/solutions/word-ladder-ii/#tag-lang-python
from collections import deque
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: a list of lists of string
    """
    def findLadders(self, start, end, words):
        words.add(start)
        words.add(end)

        graph = self.build_graph(words)
        distance = self.bfs(end, graph)

        results=[]
        self.dfs(start, end, distance, graph, [start], results)
        
        return results
    
    def build_graph(self, words):
        graph = {}

        for w in words:
            for i in range(len(w)):
                key = w[:i] + '*' + w[i+1:]
                graph[key] = graph.get(key, set())
                graph[key].add(w)

        return graph
    
    def get_nxt_word(self, word, graph):
        lst = []
        for i in range(len(word)):
            key = word[:i] + '*' + word[i+1:]
            for w in graph.get(key, []):
                lst.append(w)
        
        return lst
    
    def bfs(self, end, graph):
        distance = {end:0}
        queue = deque([end])

        while queue:
            cur = queue.popleft()
            for nxt in self.get_nxt_word(cur, graph):
                if nxt in distance:
                    continue
            
                distance[nxt] = distance[cur] + 1
                queue.append(nxt)
        
        return distance
    
    def dfs(self, src, dst, distance, graph, path, result):
        if src == dst:
            result.append(path[:])
            return
        
        for nxt in self.get_nxt_word(src, graph):
            if distance[src] - 1 != distance[nxt]:
                continue

            path.append(nxt)
            self.dfs(nxt, dst, distance, graph, path, result)
            path.pop()


# 123. 单词搜索
# [2021年8月8日]
# https://www.lintcode.com/problem/word-search/description
# https://www.jiuzhang.com/solution/word-search/#tag-lang-python
class Solution:
    def exist(self, board, word):
        if word == []:
            return True

        m = len(board)
        if m == 0:
            return False
     
        n = len(board[0])
        if n == 0:
            return False
     
        visited = [[False for j in range(n)] for i in range(m)]

        # DFS
        for i in range(m):
            for j in range(n):
                if self.dfs(board, word, visited, i, j):
                    return True
     
        return False

    def dfs(self, board, word, visited, row, col):
        if word == '':
            return True

        m, n = len(board), len(board[0])
        if row < 0 or row >= m or col < 0 or col >= n:
            return False

        if board[row][col] == word[0] and not visited[row][col]:
            visited[row][col] = True

            if self.dfs(board, word[1:], visited, row - 1, col) or\
               self.dfs(board, word[1:], visited, row, col - 1) or\
               self.dfs(board, word[1:], visited, row + 1, col) or\
               self.dfs(board, word[1:], visited, row, col + 1):
                return True
            else:
                visited[row][col] = False
        
        return False


# 132. 单词搜索 II
# [2021年8月8日]
# https://www.lintcode.com/problem/word-search-ii/description
# https://www.jiuzhang.com/solution/word-search-ii//#tag-lang-python
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None

class Trie:
    def __init__(self):
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

        result = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.search(board, i, j, trie.root.children.get(board[i][j]), set([(i,j)]), result)

        return list(result)

    def search(self, board, x, y, node, visited, result):
        if node is None: 
            return
        
        if node.is_word:
            result.add(node.word)
        
        for dx, dy in DIRECTIONS:	
            x_, y_ = x + dx, y + dy
            if not self.inside(board, x_, y_) or (x_, y_) in visited:
                continue
            
            visited.add((x_, y_))
            self.search(board, x_, y_, node.children.get(board[x_][y_]), visited, result,)
            visited.remove((x_, y_))
            
    def inside(self, board, x, y):
        return 0 <= x < len(board) and 0 <= y < len(board[0])

