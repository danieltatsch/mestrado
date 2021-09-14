from avl_tree import *
import random

def main():
    tree = AvlTree(50, 'a')
    tree.insert(60, 'b')
    tree.insert(70, 'c')
    tree.insert(25, 'd')
    tree.insert(80, 'e')
    tree.insert(55, 'f')
    tree.insert(25, 'g')
    tree.insert(65, 'h')

    # for _ in range(10):
    #     tree.insert(random.randint(0, 100), random.randint(0, 100))

    tree.update_height()
    tree.update_bf()

    tree.display()
    print('--------------------------------------')
    tree.balance(-1, 1)
    tree.display()
    print('--------------------------------------')
    key = 80
    # print("key ({}): {}".format(key, value))
    print("key ({}): {}".format(key, tree[key]))
    print('--------------------------------------')

if __name__ == "__main__":
    main()