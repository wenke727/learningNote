import sys
"""
整棵树在该问题上的结果 和左右儿子在该问题上的结果之间的联系是什么
void traverse(TreeNode root) { 
    #  前序遍历
    traverse(root.left)
    # 中序遍历 
    traverse(root.right) 
    # 后序遍历
}
"""

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# 597/1120. 具有最大平均数的子树
# https://www.lintcode.com/problem/subtree-with-maximum-average/description
# https://www.jiuzhang.com/solution/subtree-with-maximum-average/#tag-lang-python
# Version Divide Conquer
class Solution:
    def findSubtree2(self, root):
        _, max_root, _, _ = self.dfs(root)
        return max_root

    def dfs(self, root):
        if root is None:
            return -sys.maxsize, None, 0, 0
        
        maxAvg_l, maxTree_l, size_l, sum_l = self.dfs(root.left)
        maxAvg_r, maxTree_r, size_r, sum_r = self.dfs(root.right)

        size_cur = size_l + size_r + 1
        sum_cur  = sum_r + sum_l + root.val # ! `root.val`
        avg_cur  = sum_cur / size_cur

        if maxAvg_l == max( maxAvg_l, maxAvg_r, avg_cur ):
            return maxAvg_l, maxTree_l, size_cur, sum_cur
        if maxAvg_r == max( maxAvg_l, maxAvg_r, avg_cur ):
            return maxAvg_r, maxTree_r, size_cur, sum_cur

        return avg_cur, root, size_cur, sum_cur
# version Traversal  
class Solution:
    def __init__(self):
        self.node = None
        self.sum, self.num = 0, 0 

    def findSubtree2(self, root):
        self.dfs(root)
        return self.node 

    def dfs(self, root):
        if root is None:
            return 0, 0
        
        sum_left, num_left = self.dfs(root.left)
        sum_right, num_right = self.dfs(root.right)
        
        sum = sum_left + sum_right + root.val
        num = num_left + num_right + 1

        if self.node is None or sum / num > self.sum / self.num:
            self.sum, self.num = sum, num
            self.node = root
        
        return sum, num


# 175. 翻转二叉树
# [2020年11月2日]
# https://www.lintcode.com/problem/invert-binary-tree/description
# version divide
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        self.dfs(root)
        return root

    def dfs(self, root):
        if root is None:
            return
        
        root.left, root.right =  root.right,  root.left

        self.dfs(root.left)
        self.dfs(root.right)


# 95. 验证二叉查找树
# [2020年11月2日]
# https://www.lintcode.com/problem/validate-binary-search-tree/description
# https://www.jiuzhang.com/solution/validate-binary-search-tree/#tag-lang-python
# Version Divide Conquer
class Solution:
    def isValidBST(self, root):
        isBSt, _, _ = self.divideConquer( root )
        return isBSt

    def divideConquer(self, root):
        if root is None:
            return True, None, None
        
        isBST_l, minNode_l, maxNode_l = self.divideConquer(root.left)
        isBST_r, minNode_r, maxNode_r = self.divideConquer(root.right)

        if not isBST_l or not isBST_r:
            return False, None, None
        if maxNode_l is not None and maxNode_l.val >= root.val:
            return False, None, None
        if minNode_r is not None and minNode_r.val <= root.val:
            return False, None, None
        
        minNode = minNode_l if minNode_l is not None else root
        maxNode = maxNode_r if maxNode_r is not None else root

        return True, minNode, maxNode
# version Traversal  
class Solution:
    def isValidBST(self, root):
        if root is None: return True

        stack = []
        while root:
            stack.append(root)
            root = root.left
        
        last_node = stack[-1]
        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left

            if stack:
                if stack[-1].val <= last_node.val:
                    return False
                last_node = stack[-1]

        return True

"""
# TODO 二叉树的增删改查
# 参考labuladong 二叉搜索树操作集锦
"""
# 1524. 在二叉搜索树中查找
# https://www.lintcode.com/problem/search-in-a-binary-search-tree/description
class Solution:
    """
    @param root: the tree
    @param val: the val which should be find
    @return: the node
    """
    def searchBST(self, root, val):
        if root == None or root.val == val:
            return root

        if val < root.val:
            return self.searchBST(root.left,val)
        else:
            return self.searchBST(root.right,val)


# 85. 在二叉查找树中插入节点
# https://www.lintcode.com/problem/insert-node-in-a-binary-search-tree/description
# https://www.jiuzhang.com/solution/insert-node-in-a-binary-search-tree/#tag-lang-python
# version: DFS
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: node: insert this node into the binary search tree
    @return: The root of the new binary search tree.
    """
    def insertNode(self, root, node):
        return self.__helper(root, node)
    
    def __helper(self, root, node):     
        # helper函数定义成私有属性   
        if root is None:
            return node

        if node.val < root.val:
            root.left = self.__helper(root.left, node)
        else:
            root.right = self.__helper(root.right, node)
        return root
# version
class Solution:
    """
    @param root: The root of the binary search tree.
    @param node: insert this node into the binary search tree.
    @return: The root of the new binary search tree.
    """
    def insertNode(self, root, node):
        if root is None:
            return node
            
        curt = root
        while curt != node:
            if node.val < curt.val:
                if curt.left is None:
                    curt.left = node
                curt = curt.left
            else:
                if curt.right is None:
                    curt.right = node
                curt = curt.right
        return root


# 87. 删除二叉查找树的节点
# https://www.lintcode.com/problem/remove-node-in-binary-search-tree/description
# https://www.jiuzhang.com/solution/remove-node-in-binary-search-tree/#tag-lang-python
# version: inorder -> build
class Solution:
    """
    @param root: The root of the binary search tree.
    @param value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """    
    ans = []
    def inorder(self, root, value):
        if root is None:
            return

        self.inorder(root.left, value)
        if root.val != value:
            self.ans.append(root.val)
        self.inorder(root.right, value)
    
    def build(self, l, r):
        if l == r:
            node = TreeNode(self.ans[l])
            return node

        if l > r:
            return None

        mid = (l+r) / 2
        node = TreeNode(self.ans[mid])
        node.left = self.build(l, mid-1)
        node.right = self.build(mid+1, r)
        return node
    def removeNode(self, root, value):
        # write your code here
        self.inorder(root, value)
        return self.build(0, len(self.ans)-1)
# version: Divide & Conquer Solution
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """
    def removeNode(self, root, value):
        # null case
        if root is None:
            return None

        # check if node to delete is in left/right subtree
        if value < root.val:
            root.left = self.removeNode(root.left, value)
        elif value > root.val:
            root.right = self.removeNode(root.right, value)
        else:
            # if root is has 2 childs/only one child/leaf node
            if root.left and root.right:
                max = self.findMax(root)
                root.val = max.val
                root.left = self.removeNode(root.left, max.val)
            elif root.left:
                root = root.left
            elif root.right:
                root = root.right
            else:
                root = None

        return root

    # find max node in left subtree of root
    def findMax(self, root):
        node = root.left
        while node.right:
            node = node.right
        return node




"""Tree-based Depth First Search
Tree-based Depth First Search, 考点都是基于树的深度优先搜索
碰到二叉树的问题，就想想整棵树在该问题上的结果 和左右儿子在该问题上的结果之间的联系是什么
"""
"""考察形态"""
# 596. 最小子树
# [2020年11月2日]
# https://www.lintcode.com/problem/minimum-subtree/description
# https://www.jiuzhang.com/solutions/minimum-subtree/#tag-lang-python
class Solution:
    def findSubtree(self, root):
        _, node, _ = self.dfs(root)
        return node
    
    def dfs(self, root):
        if root is None:
            return sys.maxsize, None, 0
        
        min_left,  subtree_left,  sum_left  = self.dfs(root.left)
        min_right, subtree_right, sum_right = self.dfs(root.right)

        sum_root = sum_left + sum_right + root.val

        if min_left == min(min_left, min_right, sum_root):
            return min_left, subtree_left, sum_root
        if min_right == min(min_left, min_right, sum_root):
            return min_right, subtree_right, sum_root
        return sum_root, root, sum_root


# 480. 二叉树的所有路径
# [2020年11月2日, 2020.12.03]
# https://www.lintcode.com/problem/binary-tree-paths/description
# https://www.jiuzhang.com/solutions/binary-tree-paths/#tag-lang-python
# version divide conquer
class Solution:
    def binaryTreePaths(self, root):
        if root is None:
            return []

        # ! 99%的题不需要处理叶子节点的，但这需要，因没有办法用于构建root是叶子节点的结果 
        if root.left is None and root.right is None:
            return [str(root.val)]

        path_left  = self.binaryTreePaths(root.left)
        path_right = self.binaryTreePaths(root.right)
        
        paths = []
        for path in path_right + path_left:
            paths.append( str(root.val) +'->' + path )

        return paths
# version Traversal
class Solution:
    def binaryTreePaths(self, root):
        if root is None: return []

        result = []
        self.dfs( root, [str(root.val)], result )
        return result
    
    def dfs(self, node, path, result):
        if node.left is None and node.right is None:
            result.append('->'.join(path))
            return 
        
        if node.left:
            path.append( str(node.left.val) )
            self.dfs( node.left, path, result )
            path.pop()

        if node.right:
            path.append( str(node.right.val) )
            self.dfs( node.right, path, result )
            path.pop()


# 88. 最近公共祖先
# [2020年11月3日, 2020.12.03, 2020年1月5号]
# https://www.lintcode.com/problem/lowest-common-ancestor-of-a-binary-tree/description
# https://www.jiuzhang.com/solutions/lowest-common-ancestor/#tag-lang-python
class Solution:
    def lowestCommonAncestor(self, root, A, B):
        if root is None:
            return None

        if root is A or root is B:
            return root

        # ! 无脑丢给左右
        left  = self.lowestCommonAncestor(root.left,  A, B)
        right = self.lowestCommonAncestor(root.right, A, B)

        if left is not None and right is not None:
            return root
        if left is not None:
            return left
        if right is not None:
            return right

        return None


# 474. 最近公共祖先 II # TODO 和1的区别
# https://www.lintcode.com/problem/lowest-common-ancestor-ii/description
# https://www.jiuzhang.com/solution/lowest-common-ancestor-ii/#tag-lang-python
class Solution:
    """
    @param: root: The root of the tree
    @param: A: node in the tree
    @param: B: node in the tree
    @return: The lowest common ancestor of A and B
    """
    def lowestCommonAncestorII(self, root, A, B):
        parent_set = set()

        curr = A
        while curr is not None:
            parent_set.add(curr)
            curr = curr.parent

        curr = B
        while curr is not None:
            if curr in parent_set:
                return curr
            curr = curr.parent
        return None


# 578. 最近公共祖先 III # TODO
# https://www.lintcode.com/problem/lowest-common-ancestor-iii/description
# https://www.jiuzhang.com/solution/lowest-common-ancestor-iii/#tag-lang-python
class Solution:
    """
    @param {TreeNode} root The root of the binary tree.
    @param {TreeNode} A and {TreeNode} B two nodes
    @return Return the LCA of the two nodes.
    """ 
    def lowestCommonAncestor3(self, root, A, B):
        a, b, lca = self.helper(root, A, B)
        if a and b:
            return lca
        else:
            return None

    def helper(self, root, A, B):
        if root is None:
            return False, False, None
            
        left_a, left_b, left_node = self.helper(root.left, A, B)
        right_a, right_b, right_node = self.helper(root.right, A, B)
        
        a = left_a or right_a or root == A
        b = left_b or right_b or root == B
        
        if root == A or root == B:
            return a, b, root

        if left_node is not None and right_node is not None:
            return a, b, root
        if left_node is not None:
            return a, b, left_node
        if right_node is not None:
            return a, b, right_node

        return a, b, None


# 376. Binary Tree Path Sum
# [2020.11.30]
# https://www.lintcode.com/problem/binary-tree-path-sum/description
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum(self, root, target):
        result = []
        path = []
        self.dfs(root, path, result, target)
        
        return result

    def dfs(self, root, path, result, target):
        if root is None:
            return
        
        path.append(root.val)
        target -= root.val

        if root.left is None and root.right is None and target==0:
            result.append( path[:] )
        
        self.dfs(root.left, path, result, target)
        self.dfs(root.right, path, result, target)

        path.pop()


# 246. 二叉树的路径和 II
# [2020.11.30, 2020.12.03]
# https://www.lintcode.com/problem/binary-tree-path-sum-ii/description
# https://www.jiuzhang.com/solution/binary-tree-path-sum-ii/#tag-lang-python
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum2(self, root, target):
        result = []
        if root is None: 
            return result
    
        self.dfs(root, [], result, 0, target)
        return result
    
    def dfs(self, root, path, result, l, target):
        if root is None:
            return
        
        path.append(root.val)

        tmp = target
        for i in range(l, -1, -1):
            tmp -= path[i]
            if tmp == 0:
                result.append(path[i:])
        
        self.dfs(root.left, path, result, l+1, target)
        self.dfs(root.right, path, result, l+1, target)

        path.pop()


# 97. Maximum Depth of Binary Tree
# [2020.11.30, 2020.12.03]
# https://www.lintcode.com/problem/maximum-depth-of-binary-tree/description
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxDepth(self, root):
        if root is None:
            return 0

        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
    
        return max(left, right) + 1


# 242. 将二叉树按照层级转化为链表 # TODO
# [2020年11月2日, 2020.12.03]
# https://www.lintcode.com/problem/convert-binary-tree-to-linked-lists-by-depth/description
# https://www.jiuzhang.com/solutions/convert-binary-tree-to-linked-lists-by-depth/#tag-lang-python
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
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
            cur = result[depth-1]
            while cur.next != None:
                cur = cur.next
            cur.next = node
        
        self.dfs(root.left, depth+1, result)
        self.dfs(root.right, depth+1, result)


# 93. 平衡二叉树
# https://www.lintcode.com/problem/balanced-binary-tree/description
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        balanced, _ = self.isValidate(root)
        return balanced

    def isValidate(self, root):
        if root is None:
            return True, 0

        balanced, left_height = self.isValidate(root.left)
        if not balanced:
            return False, 0

        balanced, right_height = self.isValidate(root.right)
        if not balanced:
            return False, 0

        return abs( left_height - right_height ) <= 1, max(left_height, right_height) + 1       


"""二叉树结构变化"""
# 453/114. 将二叉树拆成链表 # FIXME
# [2020年11月3日]
# https://www.lintcode.com/problem/flatten-binary-tree-to-linked-list/description
# https://www.jiuzhang.com/solutions/flatten-binary-tree-to-linked-list/#tag-lang-python
# version divide conquer
class Solution:
    def flatten(self, root):
        self.dfs(root)

    def dfs(self, root):
        if root is None: return root
        
        left  = self.dfs(root.left)
        right = self.dfs(root.right)

        # connect
        if left is not None:
            left.right = root.right
            root.right = root.left
            root.left  = None
        
        # ! return 有先后顺序 先right后left，因left在上一步已处理
        if right is not None:
            return right
        if left is not None:
            return left
        return root
# version Traversal
class Solution:
    last_node = None

    def flatten(self, root):
        if root is None:
            return

        if self.last_node is not None:
            self.last_node.left = None
            self.last_node.right = root
        
        self.last_node = root
        right = root.right
        self.flatten(root.left)
        self.flatten(right)


"""考察形态"""
# 902. BST中第K小的元素 ✨
# [2020年11月3日 2020年01月20日]
# https://www.lintcode.com/problem/kth-smallest-element-in-a-bst/description
# https://www.jiuzhang.com/solution/kth-smallest-element-in-a-bst/#tag-lang-python
class Solution:
    def kthSmallest(self, root, k):
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]

        for _ in range(k):
            node = stack.pop()

            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            
            if not stack:
                return None
        return stack[-1].val


# 86. 二叉查找树迭代器
# [2020年11月3日 2020年01月20日]
# https://www.lintcode.com/problem/binary-search-tree-iterator/description
# https://www.jiuzhang.com/solution/binary-search-tree-iterator/#tag-lang-python
class BSTIterator:
    # non-recursion 的 inorder traversal
    def __init__(self, root):
        dummy = TreeNode(0)
        dummy.right = root
        self.stack = [dummy]
        self.next()

    def hasNext(self, ):
        """
        @return: True if there has next node, or false
        """
        return len(self.stack) > 0

    def next(self, ):
        """
        @return: return next node
        """
        node = self.stack.pop()
        nxt_node = node

        if node.right:
            node = node.right
            while node:
                self.stack.append( node )
                node = node.left
        
        return nxt_node


# 448. 二叉查找树的中序后继 ✨
# [ 2020年01月20日]
# https://www.lintcode.com/problem/inorder-successor-in-bst/description
# https://www.jiuzhang.com/solution/inorder-successor-in-bst/#tag-lang-python
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None:
            return None

        if root.val <= p.val:
            # self.inorderSuccessor(root.right, p)
            return self.inorderSuccessor(root.right, p)
        
        left = self.inorderSuccessor( root.left, p )

        if left != None:
            return left
        else:
            return root


# 95. 验证二叉查找树
# [ 2020年01月20日]
# https://www.lintcode.com/problem/validate-binary-search-tree/description
# https://www.jiuzhang.com/solution/validate-binary-search-tree/#tag-lang-python
class Solution:
    def isValidBST(self, root):
        if root is None: return True

        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]
        last_node = None

        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            if stack:
                if last_node is not None and stack[-1].val <= last_node.val:
                    return False
                last_node = stack[-1]
        return True


# 67. 二叉树的中序遍历
# [2020年01月20日]
# https://www.lintcode.com/problem/binary-tree-inorder-traversal/description
class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]
        inorder = []

        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            if stack:
                inorder.append(stack[-1].val)
        return inorder


# 900. 二叉搜索树中最接近的值
# https://www.lintcode.com/problem/closest-binary-search-tree-value/description
# https://www.jiuzhang.com/solution/closest-binary-search-tree-value/#tag-lang-python
class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @return: the value in the BST that is closest to the target
    """
    def closestValue(self, root, target):
        if root is None: return None

        lower = self.get_lower_bound(root, target)
        upper = self.get_upper_bound(root, target)

        if lower is None: return upper.val
        if upper is None: return lower.val
        
        print(target - lower.val, upper.val - target)
        if  target - lower.val > upper.val - target:
            return upper.val
        return lower.val
    
    def get_lower_bound(self, root, target):
        if root is None: return None

        if target < root.val:
            return self.get_lower_bound(root.left, target)
        
        lower = self.get_lower_bound(root.right, target)

        return lower if lower is not None else root

    def get_upper_bound(self, root, target):
        if root is None: return None

        if target > root.val:
            return self.get_upper_bound(root.right, target)

        upper = self.get_upper_bound(root.left, target)

        return upper if upper is not None else root


# 901/272. 二叉搜索树中最接近的值 II
# https://www.lintcode.com/problem/closest-binary-search-tree-value-ii/description
# https://www.jiuzhang.com/solution/closest-binary-search-tree-value-ii/#tag-lang-python
class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @param k: the given k
    @return: k values in the BST that are closest to the target
    """
    def closestKValues(self, root, target, k):
        if root is None or k == 0: 
            return []

        nums = self.get_inorder( root )
        left = self.find_lower_index( nums )
        right = left + 1
        results = []

        for _ in range(k):
            if self.is_left_closer(nums, left, right, target):
                results.append( nums.left )
                left -= 1
            else:
                results.append( nums.right )
                right += 1
        return results

    def get_inorder(self, root):
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]
        inorder = []

        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            if stack:
                inorder.append( stack[-1].val )
        return inorder
    
    def find_lower_index( self, nums, target ):
        start, end = 0, len(nums) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] < target: # ! 搞错了方向
                end = mid
            else:
                start = mid
        
        if nums[start] <= target:
            return start
        return end
    
    def is_left_closer(nums, left, right, target):
        if left < 0:
            return False
        if right >= len(nums):
            return False
        
        return target - nums[left] < nums[right] - target


# 11. 二叉查找树中搜索区间
# https://www.lintcode.com/problem/search-range-in-binary-search-tree/description
# https://www.jiuzhang.com/solution/search-range-in-binary-search-tree/#tag-lang-python
class Solution:
    """
    @param root: param root: The root of the binary search tree
    @param k1: An integer
    @param k2: An integer
    @return: return: Return all keys that k1<=key<=k2 in ascending order
    """
    def searchRange(self, root, k1, k2):
        if root is None: return []
        results = []

        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]

        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            
            if stack:
                if k1 <= stack[-1].val <= k2:
                    results.append(stack[-1].val)
        return results


# 85. 在二叉查找树中插入节点
# https://www.lintcode.com/problem/insert-node-in-a-binary-search-tree/description
# https://www.jiuzhang.com/solution/insert-node-in-a-binary-search-tree/#tag-lang-python
# DESC 如果它大于当前根节点，则应该在右子树中， 如果它小于当前根节点，则应该在左子树中。 （二叉查找树中保证不插入已经存在的值）
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: node: insert this node into the binary search tree
    @return: The root of the new binary search tree.
    """
    def insertNode(self, root, node):
        # write your code here
        return self.__helper(root, node)
    
     # helper函数定义成私有属性 
    def __helper(self, root, node):     
        if root is None:
            return node

        if node.val < root.val:
            root.left = self.__helper(root.left, node)
        else:
            root.right = self.__helper(root.right, node)
        
        return root

# 87. 删除二叉查找树的节点
# https://www.lintcode.com/problem/remove-node-in-binary-search-tree/description
# https://www.jiuzhang.com/solution/remove-node-in-binary-search-tree/#tag-lang-python
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """
    def removeNode(self, root, value):
        pass


