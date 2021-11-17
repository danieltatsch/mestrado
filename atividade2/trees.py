# References: 
    # https://www.programiz.com/dsa/avl-tree
    # https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python

import colorama
from   termcolor import colored
import sys

class AvlNode(object):
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1
        self.depth = 0

    def update_depth(self, depth=0):
        if self.left is None and self.right is None:
            self.depth  = 0

        if self.left is not None:
            self.left.update_depth(depth + 1)
        if self.right is not None:
            self.right.update_depth(depth + 1)

        self.depth = depth

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s, %s' % (self.key, self.depth)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s, %s' % (self.key, self.depth)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s, %s' % (self.key, self.depth)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s, %s' % (self.key, self.depth)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

class AVLTree(object):
    # Function to insert a node
    def insert_node(self, root, key):

        # Find the correct location and insert the node
        if not root:
            return AvlNode(key)
        elif key < root.key:
            root.left = self.insert_node(root.left, key)
        elif key > root.key:
            root.right = self.insert_node(root.right, key)
        else:
            return root

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        # Update the balance factor and balance the tree
        balanceFactor = self.getBalance(root)
        if balanceFactor > 1:
            if key < root.left.key:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

        if balanceFactor < -1:
            if key > root.right.key:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        return root

    # Function to delete a node
    def delete_node(self, root, key):

        # Find the node to be deleted and remove it
        if not root:
            return root
        elif key < root.key:
            root.left = self.delete_node(root.left, key)
        elif key > root.key:
            root.right = self.delete_node(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp = self.getMinValueNode(root.right)
            root.key = temp.key
            root.right = self.delete_node(root.right,
                                          temp.key)
        if root is None:
            return root

        # Update the balance factor of nodes
        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)

        # Balance the tree
        if balanceFactor > 1:
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    # Function to perform left rotation
    def leftRotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    # Function to perform right rotation
    def rightRotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    # Get the height of the node
    def getHeight(self, root):
        if not root:
            return 0
        return root.height

    # Get balance factore of the node
    def getBalance(self, root):
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMinValueNode(self, root):
        if root is None or root.left is None:
            return root
        return self.getMinValueNode(root.left)

    def getMaxValueNode(self, root):
        if root is None or root.right is None:
            return root
        return self.getMaxValueNode(root.right)

    def preOrder(self, root):
        if not root:
            return
        print("{0} ".format(root.key), end="")
        self.preOrder(root.left)
        self.preOrder(root.right)

    # Print the tree
    def printHelper(self, currPtr, indent, last):
        if currPtr is not None:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "
            print(currPtr.key)
            self.printHelper(currPtr.left, indent, False)
            self.printHelper(currPtr.right, indent, True)

##############################################################################################

# Node creation
class Node():
    def __init__(self, item):
        self.item = item
        self.parent = None
        self.left = None
        self.right = None
        self.color = 1
        self.depth = 0

    def update_depth(self, depth=0):
        if self.parent is None:
            self.depth  = 0
        
        if self.left is not None:
            input_depth = depth + 1 if isinstance(self.left, Node) else 0
            self.left.update_depth(input_depth)
        if self.right is not None:
            input_depth = depth + 1 if isinstance(self.right, Node) else 0
            self.right.update_depth(input_depth)

        self.depth = depth

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            if isinstance(self, AvlNode):
                self.display()
                return
            else:
                line = '%s, %s RED' % (self.item, self.depth) if self.color == 1 else '%s, %s, BLACK' % (self.item, self.depth)
                # line = colored('(%s, %s)', 'red') % (self.item, self.depth) if self.color == 1 else colored('(%s, %s)', 'cyan') % (self.item, self.depth)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()

            if isinstance(self, AvlNode):
                self.display()
                return
            else:
                s = '%s, %s RED' % (self.item, self.depth) if self.color == 1 else '%s, %s, BLACK' % (self.item, self.depth)
                # s = colored('(%s, %s)', 'red') % (self.item, self.depth) if self.color == 1 else colored('(%s, %s)', 'cyan') % (self.item, self.depth)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()

            if isinstance(self, AvlNode):
                self.display()
                return
            else:
                s = '%s, %s RED' % (self.item, self.depth) if self.color == 1 else '%s, %s, BLACK' % (self.item, self.depth)
                # s = colored('(%s, %s)', 'red') % (self.item, self.depth) if self.color == 1 else colored('(%s, %s)', 'cyan') % (self.item, self.depth)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()

        s = '%s, %s RED' % (self.item, self.depth) if self.color == 1 else '%s, %s, BLACK' % (self.item, self.depth)
        # s = colored('(%s, %s)', 'red') % (self.item, self.depth) if self.color == 1 else colored('(%s, %s)', 'cyan') % (self.item, self.depth)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


class RedBlackTree():
    def __init__(self):
        self.TNULL = Node(0)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL
        self.avl_tree = AVLTree()

    # Preorder
    def pre_order_helper(self, node):
        if node != self.TNULL:
            sys.stdout.write(node.item + " ")
            self.pre_order_helper(node.left)
            self.pre_order_helper(node.right)

    # Inorder
    def in_order_helper(self, node):
        if node != self.TNULL:
            self.in_order_helper(node.left)
            sys.stdout.write(node.item + " ")
            self.in_order_helper(node.right)

    # Postorder
    def post_order_helper(self, node):
        if node != self.TNULL:
            self.post_order_helper(node.left)
            self.post_order_helper(node.right)
            sys.stdout.write(node.item + " ")

    # Search the tree
    def search_tree_helper(self, node, key):
        if node == self.TNULL or key == node.item:
            return node

        if key < node.item:
            return self.search_tree_helper(node.left, key)
        return self.search_tree_helper(node.right, key)

    # Balancing the tree after deletion
    def delete_fix(self, x):
        while not isinstance(x, AvlNode) and x != self.root and x.color == 0:
            if x == x.parent.left:
                s = x.parent.right

                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    s = x.parent.right

                old_left = s.left
                if isinstance(s.left, AvlNode):
                    s.left   = self.TNULL

                elif s.left.color == 0 and s.right.color == 0:
                    s.color = 1
                    x = x.parent

                    if isinstance(old_left, AvlNode):
                        s.left = old_left
                else:
                    old_right = self.TNULL

                    if not isinstance(s, AvlNode) and s.right.color == 0:
                        old_right = s.right

                        if isinstance(s.right, AvlNode):
                            s.right = self.TNULL

                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s = x.parent.right

                    s.color = x.parent.color
                    x.parent.color = 0
                    s.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root

                    if isinstance(old_right, AvlNode):
                        s.right.right = old_right
            else:
                s = x.parent.left

                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    s = x.parent.left

                old_right = s.right

                if isinstance(s.right, AvlNode):
                    s.right = self.TNULL

                if s.right is None:
                    s.right = self.TNULL
                if s.left is None:
                    s.left = self.TNULL

                elif s.right.color == 0 and s.left.color == 0:
                    s.color = 1
                    x = x.parent

                    if isinstance(old_right, AvlNode):
                        s.right = old_right
                else:
                    old_left = self.TNULL

                    if not isinstance(s, AvlNode) and s.left.color == 0:
                        old_left = s.left
                        if not isinstance(s.left, AvlNode):
                            s.left = self.TNULL

                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s = x.parent.left

                    s.color = x.parent.color
                    x.parent.color = 0
                    s.left.color = 0
                    self.right_rotate(x.parent)

                    x = self.root

                    if isinstance(old_left, AvlNode):
                        s.left.left = old_left
        x.color = 0

    def __rb_transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def avl2rb(self, node):
      if node != self.TNULL:
        if node.left is not None and node.left != self.TNULL and isinstance(node.left, Node):
            self.avl2rb(node.left)
        if node.right is not None and node.right != self.TNULL and isinstance(node.right, Node):
            self.avl2rb(node.right)

        avl_root_parent = node

        if isinstance(avl_root_parent, Node):
            if isinstance(avl_root_parent.left, AvlNode):
                if avl_root_parent.depth < 2:
                    old_left = avl_root_parent.left
                    avl_key  = self.avl_tree.getMaxValueNode(old_left).key

                    new        = Node(avl_key)
                    new.parent = avl_root_parent
                    new.right  = self.TNULL

                    new_left_left = self.avl_tree.delete_node(old_left, avl_key)

                    new.left             = new_left_left if new_left_left is not None else self.TNULL
                    new.color            = 0
                    avl_root_parent.left = new

            if isinstance(avl_root_parent.right, AvlNode):
                if avl_root_parent.depth < 2:
                    old_right = avl_root_parent.right
                    avl_key   = self.avl_tree.getMinValueNode(old_right).key

                    new        = Node(avl_key)
                    new.parent = avl_root_parent
                    new.left   = self.TNULL

                    new_right_right = self.avl_tree.delete_node(old_right, avl_key)

                    new.right             = new_right_right if new_right_right is not None else self.TNULL
                    new.color             = 0
                    avl_root_parent.right = new

            self.root.update_depth()

    # Node deletion
    def delete_node_helper(self, node, key):
        z = self.TNULL
        avl_root_parent = None

        while node != self.TNULL:
            if isinstance(node, AvlNode):
                node = self.avl_tree.delete_node(node, key)

                if key < avl_root_parent.item:
                    avl_root_parent.left = node if node is not None else self.TNULL
                else:
                    avl_root_parent.right = node if node is not None else self.TNULL

                return

            avl_root_parent = node

            if node.item == key:
                z = node

                if isinstance(node.right, AvlNode):
                    avl_key   = self.avl_tree.getMinValueNode(node.right).key
                    new_right = self.avl_tree.delete_node(node.right, avl_key)

                    z.item  = avl_key
                    z.right = new_right if new_right is not None else self.TNULL
                    return

                elif isinstance(node.left, AvlNode):
                    avl_key  = self.avl_tree.getMaxValueNode(node.left).key
                    new_left = self.avl_tree.delete_node(node.left, avl_key)

                    z.item = avl_key
                    z.left = new_left if new_left is not None else self.TNULL
                    return

            if node.item <= key:
                node = node.right
            else:
                node = node.left

        if z == self.TNULL:
            print("Cannot find key in the tree")
            return

        y = z

        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif (z.right == self.TNULL):
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        self.root.update_depth()
        self.avl2rb(self.root)

        if y_original_color == 0 and x != self.TNULL:
            self.delete_fix(x)
            self.root.update_depth()
            self.avl2rb(self.root)

    # Balance the tree after insertion
    def fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right

                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    # Printing the tree
    def __print_helper(self, node, indent, last):
        if node != self.TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            s_color = "RED" if node.color == 1 else "BLACK"
            print(str(node.item) + "(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    def preorder(self):
        self.pre_order_helper(self.root)

    def inorder(self):
        self.in_order_helper(self.root)

    def postorder(self):
        self.post_order_helper(self.root)

    def searchTree(self, k):
        return self.search_tree_helper(self.root, k)

    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    def maximum(self, node):
        while node.right != self.TNULL:
            node = node.right
        return node

    def successor(self, x):
        if x.right != self.TNULL:
            return self.minimum(x.right)

        y = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    def predecessor(self,  x):
        if (x.left != self.TNULL):
            return self.maximum(x.left)

        y = x.parent
        while y != self.TNULL and x == y.left:
            x = y
            y = y.parent

        return y

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL and y.left is not None:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL and y.right is not None:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key):
        node = Node(key)
        node.parent = None
        node.item = key
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = 1

        y = None
        x = self.root
        avl_root_parent = None

        while x != self.TNULL:
            y = x

            if isinstance(y, Node):
                avl_root_parent = y

                if node.item < x.item:
                    x = x.left
                else:
                    x = x.right

            else:
                y = self.avl_tree.insert_node(y, key)

                if node.item < avl_root_parent.item:
                    avl_root_parent.left = y
                else:
                    avl_root_parent.right = y

                self.root.update_depth()
                return

            if x is None:
                break

        node.parent = y

        if y is not None and y.depth >= 2:

            if isinstance(y, Node): # se o pai eh RB
                avl_node = None
                avl_node = self.avl_tree.insert_node(avl_node, key)

                if node.item < y.item:
                    y.left = avl_node
                else:
                    y.right = avl_node


                # y.color = 0
                self.root.update_depth()
                return

        if y is None:
            self.root = node
        elif node.item < y.item:
            y.left = node
        else:
            y.right = node

        if node.parent == None:
            node.color = 0
            return

        if node.parent.parent == None:
            return

        self.fix_insert(node)
        self.root.update_depth()

    def get_root(self):
        return self.root

    def delete_node(self, item):
        self.delete_node_helper(self.root, item)
        self.root.update_depth()

    def print_tree(self):
        self.__print_helper(self.root, "", True)

    def display(self):
        colorama.init()

        lines, *_ = self.root._display_aux()
        for line in lines:
            print(line)