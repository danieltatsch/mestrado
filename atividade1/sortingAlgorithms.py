def insertion_sort(data):
    for j in range(1, len(data)):
        key = data[j]
        i = j - 1

        while i >= 0 and data[i] > key:
            data[i + 1] = data[i]
            i -= 1

        data[i + 1] = key

def merge_sort(data):
    if len(data) > 1:
        middle     = len(data)//2
        left_list  = data[:middle]
        right_list = data[middle:]

        merge_sort(left_list)
        merge_sort(right_list)

        i = 0
        j = 0
        k = 0

        while i < len(left_list) and j < len(right_list):
            if left_list[i] < right_list[j]:
                data[k] = left_list[i]
                i += 1
            else:
                data[k] = right_list[j]
                j += 1
            k += 1

        while i < len(left_list):
            data[k] = left_list[i]
            i += 1
            k += 1

        while j < len(right_list):
            data[k] = right_list[j]
            j += 1
            k += 1

def merge_insertion_sort(data, data_len, insertion_sort_level):
    if len(data) > 1:
        middle = data_len//2

        if insertion_sort_level == 0:
            insertion_sort(data)
        else:
            left_list  = data[:middle]
            right_list = data[middle:]

            merge_insertion_sort(left_list, len(left_list), insertion_sort_level - 1)
            merge_insertion_sort(right_list, len(right_list), insertion_sort_level - 1)

            i = 0
            j = 0
            k = 0

            while i < len(left_list) and j < len(right_list):
                if left_list[i] < right_list[j]:
                    data[k] = left_list[i]
                    i += 1
                else:
                    data[k] = right_list[j]
                    j += 1
                k += 1

            while i < len(left_list):
                data[k] = left_list[i]
                i += 1
                k += 1

            while j < len(right_list):
                data[k] = right_list[j]
                j += 1
                k += 1