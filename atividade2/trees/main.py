import random
from trees import *

def avl_test():
    myTree = AVLTree()
    root   = None

    nums = [33, 13, 52, 9, 21, 61, 8, 11, 11, 11, 13, 52]
    for num in nums:
        root = myTree.insert_node(root, num)

    # N = 10
    # for _ in range(N):
    #     root = myTree.insert_node(root, random.randint(0, 100))

    root.display()

    try:
        key  = int(input("\nSelect a key to delete: "))
        root = myTree.delete_node(root, key)

        print("\nAfter Deletion: ")
        root.display()
    except:
        pass

def main():
    avl_test()

if __name__ == "__main__":
    main()