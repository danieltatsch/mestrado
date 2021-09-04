from avl_tree import *
import random

from pprint import pprint

def main():
    tree = AvlTree()
    # tree.insert(10)

    for _ in range(10):
        tree.insert(random.randint(0, 100))

    tree.display()

if __name__ == "__main__":
    main()