import random
from trees import *
import sys

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

def rb_avl_test():
    bst = RedBlackTree()

    bst.insert(55)
    bst.insert(40)
    bst.insert(65)
    bst.insert(60)
    bst.insert(75)
    bst.insert(56)
    bst.insert(57)
    bst.insert(58)
    bst.insert(59)
    bst.insert(55)
    bst.insert(41)
    bst.insert(42)
    bst.insert(43)
    bst.insert(44)
    bst.insert(45)
    bst.delete_node(75)
    bst.delete_node(60)
    bst.delete_node(40)
    bst.insert(70)
    bst.insert(71)

    print("ATUAL")
    bst.display()

    print("\nAfter deleting an element 41")
    bst.delete_node(41)
    bst.display()

    print("\nAfter deleting an element 57")
    bst.delete_node(57)
    bst.display()

    print("\nAfter deleting an element 65")
    bst.delete_node(65)
    bst.display()

    print("\nAfter deleting an element 59")
    bst.delete_node(59)
    bst.display()

def insert_menu(rb_avl):
    print("================================================================================")
    while True:
        key = input("Insert (or type BACK to return to main menu): ")

        try:
            key = int(key)
        except:
            if key.lower() == 'back':
                print("================================================================================")
                return

            print("Invalid data, enter a number!")
            continue

        print("\nAfter insert element {}".format(key))

        rb_avl.insert(key)
        rb_avl.display()

def delete_menu(rb_avl):
    print("================================================================================")
    while True:
        key = input("Delete (or type BACK to return to main menu): ")

        try:
            key = int(key)
        except:
            if key.lower() == 'back':
                print("================================================================================")
                return

            print("Invalid data, enter a number!")
            continue

        print("\nAfter delete element {}".format(key))

        rb_avl.delete_node(key)
        rb_avl.display()

def menu():
    option = -1
    while option not in range(0,3):
        print("=============================== AVL + RED BLACK ================================")
        print("Choose an option:")
        print("1. Insert")
        print("2. Delete")
        print("0. Exit")
        print("================================================================================")
      
        try:
            option = int(input())
        except:
            option = -1

    return option

def main():
    # avl_test()
    # rb_avl_test()
    rb_avl = RedBlackTree()

    rb_avl.insert(60)
    rb_avl.insert(50)
    rb_avl.insert(40)
    rb_avl.insert(30)
    rb_avl.insert(20)
    rb_avl.insert(10)
    rb_avl.insert(5)

    rb_avl.display()

    while True:
        option = menu()

        if option == 1:
            insert_menu(rb_avl)
        elif option == 2:
            delete_menu(rb_avl)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()