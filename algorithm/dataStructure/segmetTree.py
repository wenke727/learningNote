class SegmentTreeNode:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.left, self.right = None, None


# 201 · 线段树的构造
# https://www.lintcode.com/problem/201/
class Solution:
    """
    @param: start: start value.
    @param: end: end value.
    @return: The root of Segment Tree.
    """
    def build(self, start, end):
        if start > end: return None

        root = SegmentTreeNode(start, end)
        
        if start == end:
            return root
        
        root.left = self.build(start, (start+end)//2)
        root.right = self.build((start+end)//2 + 1, end)

        return root

# 202 · 线段树的查询
# https://www.lintcode.com/problem/202/
class Solution:	
    def query(self, root, start, end):
        if root.start > end or root.end < start:
            return -0x7fffff
        
        if start <= root.start and root.end <= end:
            return root.max
        
        return max(self.query(root.left, start, end),
                   self.query(root.right, start, end)
        )


# 203 · 线段树的修改
# https://www.lintcode.com/problem/203/
class Solution:	
    """
    @param root, index, value: The root of segment tree and 
    @ change the node's value with [index, index] to the new given value
    @return: nothing
    """
    def modify(self, root, index, value):
        if root is None:
            return
        
        if root.start == root.end:
            root.max = value
            return
        
        if root.left.end >= index:
            self.modify(root.left, index, value)
        else:
            self.modify(root.right, index, value)
        
        root.max = max(root.left.max, root.right.max)


# 205 · 区间最小数
class SegmentTree(object):
    def __init__(self, start, end, min=0):
        self.start = start
        self.end = end
        self.min = min
        self.left, self.right = None, None
    
    @classmethod
    def build(cls, start, end, a):
        if start > end:
            return None
        
        if start == end:
            return SegmentTree(start, end, a[start])
        
        node = SegmentTree(start, end, a[start])
        mid = (start + end) // 2
        node.left = cls.build(start, mid, a)
        node.right = cls.build(mid+1, end, a)
        node.min = min(node.left.min, node.right.min)

        return node
    
    @classmethod
    def query(self, root, start, end):
        if root.start > end or root.end < start:
            return 0x7fffff
        
        if start <= root.start and root.end <= end:
            return root.min
        
        return min(self.query(root.left, start, end),
                   self.query(root.right, start, end))

class Solution:	
    def intervalMinNumber(self, A, queries):
        root = SegmentTree.build(0, len(A)-1, A)
        result = []

        for query in queries:
            result.append(SegmentTree.query(root, query.start, query.end))

        return result


# 206 区间求和 I · Interval Sum
# https://www.lintcode.com/problem/206/
class SegmentTree(object):
    def __init__(self, start, end, sum=0):
        self.start = start
        self.end = end
        self.sum = sum
        self.left, self.right = None, None
    
    @classmethod
    def build(cls, start, end, a):
        if start > end:
            return None
        
        if start == end:
            return SegmentTree(start, end, a[start])
        
        node = SegmentTree(start, end, a[start])

        mid = (start + end) // 2
        node.left = cls.build(start, mid, a)
        node.right = cls.build(mid+1, end, a)
        
        l_sum, r_sum = 0, 0 
        if node.left:
            l_sum += node.left.sum
        if node.right:
            r_sum += node.right.sum
        node.sum = l_sum + r_sum

        return node
    
    @classmethod
    def query(self, root, start, end):
        if root.start > end or root.end < start:
            return 0
        
        if start <= root.start and root.end <= end:
            return root.sum
        
        return self.query(root.left, start, end) +  self.query(root.right, start, end)

class Solution:	
    def intervalSum(self, A, queries):
        root = SegmentTree.build(0, len(A)-1, A)
        result = []

        for query in queries:
            result.append(SegmentTree.query(root, query.start, query.end))

        return result


# 247 · 线段树查询 II
# https://www.lintcode.com/problem/247/
class Solution:	
    def query(self, root, start, end):
        if root is None:
            return 0

        if root.start > end or root.end < start:
            return 0
    
        if start <= root.start and root.end <= end:
            return root.count
        
        return self.query(root.left, start, end) + \
               self.query(root.right, start, end)


# 248 · 统计比给定整数小的数的个数
# https://www.lintcode.com/problem/248/
class Solution:
    def countOfSmallerNumber(self, A, queries):
        # build segmeng tree
        root = self.build(0, 10000)
        result = []
        
        # modify count value for each
        for num in A:
            self.modify(root, num, 1)
        
        for i in queries:
            count = 0
            if i > 0:
                count = self.query(root, 0, i - 1)
            result.append(count)
        
        return result
    
    def build(self, start, end):
        if start >= end:
            return SegmentTreeNode(start, end, 0)
        root = SegmentTreeNode(start, end, 0)
        mid = start + (end - start) // 2
        root.left = self.build(start, mid)
        root.right = self.build(mid + 1, end)
        return root
    
    def modify(self, root, index, value):
        if root.start == index and root.end == index:
            root.count += value
            return
        # query
        mid = root.start + (root.end - root.start) // 2
        if index <= mid:
            self.modify(root.left, index, value)
        
        if mid < index:
            self.modify(root.right, index, value)
        root.count = root.left.count + root.right.count
    
    def query(self, root, start, end):
        if start == root.start and end == root.end:
            return root.count
        
        mid = root.start + (root.end - root.start) // 2
        if end <= mid:
            return self.query(root.left, start, end)
        
        if start > mid:
            return self.query(root.right, start, end)
            
        return self.query(root.left, start, mid) + \
            self.query(root.right, mid + 1, end)


# 439 · 线段树的构造 II
# https://www.lintcode.com/problem/439/
class Solution:
    """
    @param A: a list of integer
    @return: The root of Segment Tree
    """
    def build(self, A):
        return self.buildTree(0, len(A)-1, A)

    def buildTree(self, start, end, A):
        if start > end: 
            return None
        
        node = SegmentTreeNode(start, end, A[start])
        if start == end:
            return node
        
        mid = (start + end) // 2
        node.left = self.buildTree(start, mid, A)
        node.right = self.buildTree(mid+1, end, A)

        if node.left is not None and node.left.max > node.max:
            node.max = node.left.max
        if node.right is not None and node.right.max > node.max:
            node.max = node.right.max
        
        return node


# 751 · 约翰的生意
# https://www.lintcode.com/problem/751/
class SegmentTree:
    def __init__(self, start, end, min_val):
        self.start = start
        self.end = end
        self.min_val = min_val
        self.left = None
        self.right = None

class Solution:
    def business(self, A, k):
        result = []
        root = self.build(A, 0, len(A) - 1)

        for i in range(len(A)):
            temp = sys.maxsize
            left = max(0, i - k)
            right = min(len(A) - 1, i + k)
            
            temp = min(temp, self.query(root, i - k, i + k))
            
            result.append(A[i] - temp)
            
        return result
        
    def build(self, A, start, end):
        if start > end:
            return
        if start == end:
            return SegmentTree(start, end, A[start])
        
        mid = (start + end) // 2
        left_node = self.build(A, start, mid)
        right_node = self.build(A, mid + 1, end)
        
        root = SegmentTree(start, end, A[start])
        root.min_val = min(left_node.min_val, right_node.min_val)
        
        root.left = left_node
        root.right = right_node
        
        return root
        
    def query(self, root, start, end):
        if root is None:
            return
        if start > root.end or end < root.start:
            return sys.maxsize
        
        if start <= root.start and end >= root.end:
            return root.min_val
        return min(self.query(root.left, start, end), self.query(root.right, start, end))


