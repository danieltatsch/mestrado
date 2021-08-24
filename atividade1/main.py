import numpy             as np
import matplotlib.pyplot as plt
from   time import time
import sortingAlgorithms

def test_insertion_sort(data):
    interval = []

    for i in data:
        start = time()

        sortingAlgorithms.insertion_sort(i)

        interval.append(time() - start)

        print(i)
    
    print('\n\n')
    print(interval)

    return interval

def test_merge_sort(data):
    interval = []

    for i in data:
        start = time()

        sortingAlgorithms.merge_sort(i)

        interval.append(time() - start)

        print(i)
    
    print('\n\n')
    print(interval)

    return interval

def test_merge_insert_sort(data, insertion_sort_level):
    interval_by_k = {}

    i = 1
    while i <= insertion_sort_level:
    # for i in range(insertion_sort_level + 1):
        interval = []

        for j in data:
            start = time()

            sortingAlgorithms.merge_insertion_sort(j, len(j), i)

            interval.append(time() - start)

            print(j)
    
        interval_by_k[i] = np.array(interval).mean()
        i += 1

    print('\n\n')
    print(interval_by_k)

    return interval_by_k

M     = 50    # rows
N     = 10000 # columns
range = 1000  # 0 - 999
k     = np.floor(np.log2(N))

data = np.random.randint(range, size=(M, N)).tolist()

interval = np.array(test_insertion_sort(data))
# interval = np.array(test_merge_sort(data))
# interval = test_merge_insert_sort(data, int(k))

filename = 'merge_insert_sort_{}_{}_{}_{}.txt'.format(M, N, range, int(time()))

print("Saving data to file...")
f = open(filename, "w")
for i in interval:
    f.write(str(i) + '\n')
# for key, value in interval.items():
#     f.write(str(key) + ': ' + str(value) + '\n')
f.close()

# print('\n------------------------------------')
# print('Mean: {}'.format(interval.mean()))
# print('Std:  {}'.format(np.std(interval)))
# print('Var:  {}'.format(np.var(interval)))
# print('------------------------------------')

a = plt.hist(np.array(list(interval.values())), bins=len(interval))
plt.title("Merge Sort Histogram")
plt.show()

# sortingAlgorithms.insertion_sort(a)
# sortingAlgorithms.merge_sort(a)
# sortingAlgorithms.merge_insertion_sort(a, len(a), 12)

# print("OUTPUT: {}".format(a))

# def generate_rand_data(N, M, range):
    # return 