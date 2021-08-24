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

def loop_sort(N, range, M=10):
    interval = []

    vector_size = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, N]
    for i in vector_size:
        print('Vector size: {}'. format(i))
        data = np.random.randint(range, size=(M, i)).tolist()

        interval.append(np.array(test_merge_sort(data)).mean())

    return interval

M     = 50    # lines
N     = 10000 # columns
range = 1000  # 0 - 999
k     = np.floor(np.log2(N))

interval = loop_sort(N, range)
dale = plt.plot(np.array(interval))
plt.show()

filename = 'loop_merge{}_{}_{}.txt'.format(M, range, int(time()))

print("Saving data to file...")
f = open(filename, "w")
for i in interval:
    f.write(str(i) + '\n')
f.close()
