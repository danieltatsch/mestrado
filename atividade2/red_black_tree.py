###################################### RULES ######################################
# 1 - Nodes must be red or black
# 2 - Root is always black
# 3 - Leaves are always black
# 4 - If a node is red then all its children are black
# 5 - If a node is red then its parent is black
# 6 - Every path from a node to its leaves must have the same number of black nodes
# 7 - Every new node inserted in the tree is red by default
###################################################################################
import colorama
from   termcolor import colored

red   = 0
black = 1 # cyan

class RBTree:
    def __init__(self, key=None, value=None, parent=None, color=black, left=None, right=None):
        self.key    = key
        self.value  = value
        self.color  = color
        self.parent = parent
        self.left   = left
        self.right  = right

    def insert(self, key, value):
        if self.key is None:
            self.key   = key
            self.value = value
        elif key < self.key:
            if self.left is None:
                self.left = RBTree(key, value, self, red)

                if self.color is black:
                    print("ok")
                else:
                    self.left.update_color()
            else:
                self.left.insert(key, value)
        elif key > self.key:
            if self.right is None:
                self.right = RBTree(key, value, self, red)
                
                if self.color is black:
                    print("ok")
                else:
                    self.right.update_color()
            else:
                self.right.insert(key, value)

    def update_color(self):
        pass

    def get(self, key):
        node = self
        while node is not None:
            if node.key == key:
                return node.value

            node = node.right if node.key < key else node.left

        return None

    def get_recursive(self, key):
        return (None                          if self.key is None                           else
                self.value                    if self.key == key                           else
                self.right.get_recursive(key) if self.key < key and self.right is not None else
                self.left.get_recursive(key)  if self.key > key and self.left  is not None else
                None)

    def __getitem__(self, key):
        return self.get_recursive(key)

    def rotate_left(self):
        self.left  = RBTree(self.key, self.value, self.left, self.right.left)
        self.key   = self.right.key
        self.value = self.right.value
        self.right = self.right.right

    def rotate_right(self):
        self.right = RBTree(self.key, self.value, self.left.right, self.right)
        self.key   = self.left.key
        self.value = self.left.value
        self.left  = self.left.left

# https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python

    def display(self):
        colorama.init()

        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = colored('(%s, %s)', 'red') % (self.key, self.value) if self.color is red else colored('(%s, %s)', 'cyan') % (self.key, self.value)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = colored('(%s, %s)', 'red') % (self.key, self.value) if self.color is red else colored('(%s, %s)', 'cyan') % (self.key, self.value)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = colored('(%s, %s)', 'red') % (self.key, self.value) if self.color is red else colored('(%s, %s)', 'cyan') % (self.key, self.value)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = colored('(%s, %s)', 'red') % (self.key, self.value) if self.color is red else colored('(%s, %s)', 'cyan') % (self.key, self.value)
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
