class AvlTree:
    def __init__(self, data=None, left=None, right=None):
        self.data   = data
        self.left   = left
        self.right  = right
        self.bf     = 0
        self.height = 0

    def insert(self, data):
        if self.data is None:
            self.data = data
        elif data < self.data:
            if self.left is None:
                self.left = AvlTree(data)
            else:
                self.left.insert(data)
        elif data > self.data:
            if self.right is None:
                self.right = AvlTree(data)
            else:
                self.right.insert(data)

    def update_height(self):
        height_left  = 1
        height_right = 1

        if self.left is None and self.right is None:
            self.height = 1
            return 1
        if self.left is not None:
            height_left = self.left.update_height()
        if self.right is not None:
            height_right = self.right.update_height()

        self.height = 1 + max(height_left, height_right)

        return self.height

    def update_bf(self):
        height_left  = 0
        height_right = 0

        if self.left is None and self.right is None:
            self.bf = 0
        if self.left is not None:
            self.left.update_bf()
            height_left = self.left.height
        if self.right is not None:
            self.right.update_bf()
            height_right = self.right.height

        self.bf = height_left - height_right

    def get_data(self, data):
        node = self
        while node is not None:
            if node.data == data:
                return node.data

            node = node.right if node.data < data else node.left

        return None

    def get_data_recursive(self, data):
        return (None                                if self.data is None                           else
                self.data                           if self.data == data                           else
                self.right.get_data_recursive(data) if self.data < data and self.right is not None else
                self.left.get_data_recursive(data)  if self.data > data and self.left  is not None else
                None)

        # if self.data is None:
        #     return None

        # if self.data == data:
        #     return self.data

        # if self.data < data and self.right is not None:
        #     return self.right.get_data_recursive(data)

        # if self.data > data and self.left is not None:
        #     return self.left.get_data_recursive(data)

        # return None

    def rotate_left(self):
        self.left  = AvlTree(self.data, self.left, self.right.left)
        self.data  = self.right.data
        self.right = self.right.right

    def rotate_right(self):
        self.right = AvlTree(self.data, self.left.right, self.right)
        self.data  = self.left.data
        self.left  = self.left.left

    def balance(self, neg_bf_desired, pos_bf_desired, parent = None):
        if self.left is not None:
            self.left.balance(neg_bf_desired, pos_bf_desired, self)
        if self.right is not None:
            self.right.balance(neg_bf_desired, pos_bf_desired, self)

        if parent is None:
            parent = self

        if parent.bf < neg_bf_desired or parent.bf > pos_bf_desired:
            while self.bf < neg_bf_desired or self.bf > pos_bf_desired:
                if self.bf > pos_bf_desired:
                    if self.left.bf < 0:
                        self.left.rotate_left()
                    self.rotate_right()
                elif self.bf < neg_bf_desired:
                    if self.right.bf > 0:
                        self.right.rotate_right()
                    self.rotate_left()

                self.update_height()
                self.update_bf()
                parent.update_height()
                parent.update_bf()

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.data
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.data
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.data
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.data
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
