import random
import math


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

    pivot = arr[random.randint(0, len(arr) - 1)]
    less, equal, greater = [], [], []
    for x in arr:
        if x < pivot:
            less.append(x)
        elif x > pivot:
            greater.append(x)
        else:
            equal.append(x)

    return quick_sort(less) + equal + quick_sort(greater)


# 3. Heap Sort
def heap_sort(arr):
    a = arr[:]
    n = len(a)

    def sift_down(start, end):
        root = start
        while 2 * root + 1 <= end:
            child = 2 * root + 1
            swap = root
            if a[swap] < a[child]:
                swap = child
            if child + 1 <= end and a[swap] < a[child + 1]:
                swap = child + 1
            if swap == root:
                return
            a[root], a[swap] = a[swap], a[root]
            root = swap

    # Build max heap
    for start in range((n - 2) // 2, -1, -1):
        sift_down(start, n - 1)

    # Extract elements
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(0, end - 1)

    return a


# 4. 3-Way Radix Quicksort (Bentley-Sedgewick Multikey Quicksort)
def multikey_quicksort(arr):
    a = arr[:]

    def char_at(s, d):
        return ord(s[d]) if d < len(s) else -1

    def sort_3way(lo, hi, d):
        if hi <= lo:
            return
        lt, gt = lo, hi
        v = char_at(a[lo], d)
        i = lo + 1
        while i <= gt:
            t = char_at(a[i], d)
            if t < v:
                a[lt], a[i] = a[i], a[lt]
                lt += 1
                i += 1
            elif t > v:
                a[i], a[gt] = a[gt], a[i]
                gt -= 1
            else:
                i += 1

        sort_3way(lo, lt - 1, d)
        if v >= 0:
            sort_3way(lt, gt, d + 1)
        sort_3way(gt + 1, hi, d)

    sort_3way(0, len(a) - 1, 0)
    return a


# 5. MSD Radix Sort (Most Significant Digit)
def msd_radix_sort(arr):
    a = arr[:]
    if not a:
        return a

    # Sort strings lexicographically by digit
    R = 256  # extended ASCII alphabet

    def char_at(s, d):
        return ord(s[d]) if d < len(s) else -1

    def sort_msd(lo, hi, d):
        if hi <= lo:
            return
        count = [0] * (R + 2)
        for i in range(lo, hi + 1):
            count[char_at(a[i], d) + 2] += 1
        for r in range(R + 1):
            count[r + 1] += count[r]
        aux = [None] * (hi - lo + 1)
        for i in range(lo, hi + 1):
            c = char_at(a[i], d) + 1
            aux[count[c]] = a[i]
            count[c] += 1
        for i in range(lo, hi + 1):
            a[i] = aux[i - lo]

        for r in range(R + 1):
            sort_msd(lo + count[r], lo + count[r + 1] - 1, d + 1)

    sort_msd(0, len(a) - 1, 0)
    return a


# 6. LSD Radix Sort (Least Significant Digit)
def lsd_radix_sort(arr):
    a = arr[:]
    if not a:
        return a

    R = 256
    max_len = max(len(s) for s in a)
    aux = [None] * len(a)

    for d in range(max_len - 1, -1, -1):
        count = [0] * (R + 2)
        for s in a:
            c = ord(s[d]) + 1 if d < len(s) else 0
            count[c + 1] += 1
        for r in range(R + 1):
            count[r + 1] += count[r]
        for s in a:
            c = ord(s[d]) + 1 if d < len(s) else 0
            aux[count[c]] = s
            count[c] += 1
        a[:] = aux

    return a


# 7. Comb Sort
def comb_sort(arr):
    a = arr[:]
    n = len(a)
    gap = n
    shrink = 1.3
    sorted_flag = False

    while not sorted_flag:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = True
        i = 0
        while i + gap < n:
            if a[i] > a[i + gap]:
                a[i], a[i + gap] = a[i + gap], a[i]
                sorted_flag = False
            i += 1

    return a


# 8. Shell Sort (Knuth Sequence)
def shell_sort(arr):
    a = arr[:]
    n = len(a)

    # Generate Knuth sequence: 1, 4, 13, 40, 121, ...
    gaps = []
    g = 1
    while g < n:
        gaps.append(g)
        g = 3 * g + 1
    gaps.reverse()

    for gap in gaps:
        for i in range(gap, n):
            temp = a[i]
            j = i
            while j >= gap and a[j - gap] > temp:
                a[j] = a[j - gap]
                j -= gap
            a[j] = temp

    return a


# 9. Prefix Bucket Sort
def bucket_sort(arr):
    a = arr[:]
    if not a:
        return a

    n = len(a)

    # Determine whether to use numeric or string bucketing
    if all(isinstance(x, (int, float)) for x in a):
        # Numeric bucket sort
        min_val = min(a)
        max_val = max(a)
        if min_val == max_val:
            return a
        num_buckets = n
        buckets = [[] for _ in range(num_buckets)]
        for x in a:
            idx = int((x - min_val) / (max_val - min_val) * (num_buckets - 1))
            buckets[idx].append(x)
        result = []
        for b in buckets:
            result.extend(sorted(b))
        return result
    else:
        # String prefix bucket sort (bucket by first char)
        buckets = {}
        for s in a:
            key = s[0] if s else ""
            buckets.setdefault(key, []).append(s)
        result = []
        for key in sorted(buckets.keys()):
            result.extend(sorted(buckets[key]))
        return result


# 10. Cocktail Sort
def cocktail_sort(arr):
    a = arr[:]
    n = len(a)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False
        # Forward pass
        for i in range(start, end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1

        # Backward pass
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