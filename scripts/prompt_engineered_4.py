import time

import numpy as np
from numba import jit


# Generate hash table at module load time using bit manipulation
@jit(nopython=True, cache=True)
def init_hash_table():
    HASH_TABLE = np.zeros(100001, dtype=np.uint8)
    min_val = np.iinfo(np.uint32).max
    max_val = 0

    # Optimal digit sum using parallel bit counting
    for i in range(1, 100001):
        n = i
        sum = 0
        while n and sum <= 30:
            sum += n & 0xF
            n >>= 4
        if sum == 30:
            HASH_TABLE[i] = 1
            min_val = min(min_val, i)
            max_val = max(max_val, i)

    return min_val, max_val, HASH_TABLE


@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def pe_4_find_min_max(numbers):
    MIN_VALID, MAX_VALID, HASH_TABLE = init_hash_table()
    min_val = MAX_VALID  # Start with known bounds
    max_val = MIN_VALID
    found = False

    # Single vectorized operation
    mask = HASH_TABLE[numbers] == 1
    if np.any(mask):
        valid_nums = numbers[mask]
        min_val = np.min(valid_nums)
        max_val = np.max(valid_nums)
        found = True

    return min_val, max_val, found


def main():
    # Generate numbers with optimal memory layout
    numbers = np.random.randint(1, 100001, size=1_000_000, dtype=np.uint32)

    # Single vectorized lookup
    min_val, max_val, found = pe_4_find_min_max(numbers)

    return max_val - min_val if found else 0


if __name__ == "__main__":
    start_time = time.perf_counter()
    for _ in range(100):
        _ = main()
    end_time = time.perf_counter() - start_time

    avg_time = end_time / 100

    print(avg_time)
