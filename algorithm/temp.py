from algorithm.chapter_04_BFS import ListNode
import collections


class Solution:
    def binaryTreeToLists(self, root):
        if root is None:
            return []
        
        res = []
        queue = collections.deque([root])

        while queue:
            res.append(self.create_linkedlist(queue))
            nxtQueue = collections.deque([])

            for node in queue:
                if node.left:
                    nxtQueue.append(node.left)
        
                if node.right:
                    nxtQueue.append(node.right)
            
            queue = nxtQueue
        
        return res

    def create_linkedlist(self, nodes):
        dummy = ListNode(0)
        cur = dummy

        for node in nodes:
            p = ListNode(node.val)
            cur.next = p
            cur = p
        
        return dummy.next