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

def rb_test():
    bst = RedBlackTree()

    bst.insert(55)
    bst.insert(40)
    bst.insert(65)
    bst.insert(60)
    bst.insert(75)
    bst.insert(56)

    # bst.print_tree()
    bst.display()

    print("\nAfter inserting one more element 57")
    bst.insert(57)
    bst.display()

    print("\nAfter inserting one more element 58")
    bst.insert(58)
    bst.display()

    print("\nAfter inserting one more element 59")
    bst.insert(59)
    bst.display()

    print("\nAfter inserting one more element 57")
    bst.insert(55)
    bst.display()

    print("\nAfter inserting one more element 57")
    bst.insert(41)
    bst.display()

    print("\nAfter inserting one more element 58")
    bst.insert(42)
    bst.display()

    print("\nAfter inserting one more element 59")
    bst.insert(43)
    bst.display()

    print("\nAfter inserting one more element 57")
    bst.insert(44)
    bst.display()

    print("\nAfter inserting one more element 57")
    bst.insert(45)
    bst.display()


    # print("\nAfter deleting an element")
    # bst.delete_node(40)
    # bst.display()

    # print("\nAfter deleting an element")
    # bst.delete_node(55)
    # bst.display()

    # print("\nAfter deleting an element")
    # bst.delete_node(65)
    # bst.display()

def main():
    # avl_test()
    rb_test()

if __name__ == "__main__":
    main()