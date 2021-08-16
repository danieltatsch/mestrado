import numpy as np
import sortingAlgorithms

input_size = 40

a = np.random.randint(100, size=input_size).tolist()
# a = [2,1,34,4,7,6,5,3]

print("INPUT:  {}".format(a))

# sortingAlgorithms.insertion_sort(a)
# sortingAlgorithms.merge_sort(a)
sortingAlgorithms.merge_insertion_sort(a, len(a), 12)

print("OUTPUT: {}".format(a))