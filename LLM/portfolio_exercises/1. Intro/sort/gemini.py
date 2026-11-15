import random


# 1. Merge Sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


# 2. 3-Way Randomized Quicksort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# 3. Heap Sort
def heap_sort(arr):
    a = arr.copy()
    n = len(a)

    def sift_down(start, end):
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end and a[child] < a[child + 1]:
                child += 1
            if a[root] < a[child]:
                a[root], a[child] = a[child], a[root]
                root = child
            else:
                break

    for start in range((n - 2) // 2, -1, -1):
        sift_down(start, n - 1)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(0, end - 1)
    return a


# 4. 3-Way Radix Quicksort (Bentley-Sedgewick Multikey Quicksort)
def multikey_quicksort(arr):
    def _sort(a, depth):
        if len(a) <= 1:
            return a
        pivot_str = a[len(a) // 2]
        pivot_val = ord(pivot_str[depth]) if depth < len(pivot_str) else -1

        less, equal, greater = [], [], []
        for s in a:
            ch = ord(s[depth]) if depth < len(s) else -1
            if ch < pivot_val:
                less.append(s)
            elif ch > pivot_val:
                greater.append(s)
            else:
                equal.append(s)

        sorted_equal = _sort(equal, depth + 1) if pivot_val != -1 else equal
        return _sort(less, depth) + sorted_equal + _sort(greater, depth)

    return _sort(arr, 0)


# 5. MSD Radix Sort (Most Significant Digit)
def msd_radix_sort(arr):
    def _msd_sort(strings, depth):
        if len(strings) <= 1:
            return strings
        buckets = [[] for _ in range(257)]
        for s in strings:
            if depth < len(s):
                buckets[ord(s[depth]) + 1].append(s)
            else:
                buckets[0].append(s)

        result = list(buckets[0])
        for b in range(1, 257):
            if buckets[b]:
                result.extend(_msd_sort(buckets[b], depth + 1))
        return result

    return _msd_sort(arr, 0)


# 6. LSD Radix Sort (Least Significant Digit)
def lsd_radix_sort(arr):
    if not arr:
        return arr
    max_len = max(len(s) for s in arr)
    result = arr.copy()

    for pos in range(max_len - 1, -1, -1):
        buckets = [[] for _ in range(257)]
        for s in result:
            code = ord(s[pos]) + 1 if pos < len(s) else 0
            buckets[code].append(s)
        result = [item for b in buckets for item in b]
    return result


# 7. Comb Sort
def comb_sort(arr):
    a = arr.copy()
    gap = len(a)
    shrink = 1.3
    sorted_flag = False

    while not sorted_flag:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = True
        i = 0
        while i + gap < len(a):
            if a[i] > a[i + gap]:
                a[i], a[i + gap] = a[i + gap], a[i]
                sorted_flag = False
            i += 1
    return a


# 8. Shell Sort (Knuth Sequence)
def shell_sort(arr):
    a = arr.copy()
    n = len(a)
    gap = 1
    while gap < n // 3:
        gap = 3 * gap + 1

    while gap > 0:
        for i in range(gap, n):
            temp = a[i]
            j = i
            while j >= gap and a[j - gap] > temp:
                a[j] = a[j - gap]
                j -= gap
            a[j] = temp
        gap //= 3
    return a


# 9. Prefix Bucket Sort
def bucket_sort(arr):
    if len(arr) <= 1:
        return arr
    buckets = {}
    for s in arr:
        key = s[:2]
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(s)

    sorted_arr = []
    for key in sorted(buckets.keys()):
        sorted_arr.extend(quick_sort(buckets[key]))
    return sorted_arr


# 10. Cocktail Sort
def cocktail_sort(arr):
    a = arr.copy()
    n = len(a)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False
        for i in range(start, end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True

        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True

        start += 1

    return a


def get_algorithms():
    return {
        "3-Way Radix Quicksort": multikey_quicksort,
        "MSD Radix Sort": msd_radix_sort,
        "LSD Radix Sort": lsd_radix_sort,
        "Prefix Bucket Sort": bucket_sort,
        "Quick Sort (3-Way)": quick_sort,
        "Merge Sort": merge_sort,
        "Heap Sort": heap_sort,
        "Comb Sort": comb_sort,
        "Shell Sort (Knuth)": shell_sort,
        "Cocktail Sort": cocktail_sort,
    }
