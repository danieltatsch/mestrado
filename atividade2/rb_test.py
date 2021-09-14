from red_black_tree import *
import random

def main():
    tree = RBTree()
    tree.insert(50, 'a')
    tree.insert(60, 'b')
    tree.insert(25, 'd')
    tree.insert(70, 'c')
    tree.insert(80, 'e')
    tree.insert(55, 'f')
    tree.insert(65, 'h')

    tree.display()

    # Incoerencia ao inserir o 63
    tree.insert(63, 'i')

    # for _ in range(10):
    #     tree.insert(random.randint(0, 100), random.randint(0, 100))

    tree.display()
    print('--------------------------------------')
    # print("key ({}): {}".format(key, tree[key]))
    # print('--------------------------------------')

if __name__ == "__main__":
    main()