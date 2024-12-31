import multiprocessing
import time
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from numba import jit, prange

ArrayInt: TypeAlias = npt.NDArray[np.int_]


@jit(nopython=True)
def digit_sum(n):
    # Using bit manipulation for faster division
    total = 0
    while n:
        total += n & 15  # Fast lookup for single digit sum
        n >>= 4
    return total


@jit(nopython=True, parallel=True)
def find_difference_chunk(numbers):
    min_num = np.iinfo(np.int64).max
    max_num = np.iinfo(np.int64).min

    # Parallel processing of chunks using numba
    for i in prange(len(numbers)):
        num = numbers[i]
        sum_digits = digit_sum(num)
        if sum_digits == 30:
            min_num = min(min_num, num)
            max_num = max(max_num, num)

    return min_num, max_num


def process_chunk(chunk):
    return find_difference_chunk(chunk)


def main():
    # Calculate optimal chunk size based on CPU cores
    chunk_size = 1_000_000 // multiprocessing.cpu_count()

    # Pre-allocate memory for all random numbers
    numbers = np.random.randint(1, 100001, size=1_000_000, dtype=np.int32)

    # Split data into chunks for parallel processing
    chunks = np.array_split(numbers, multiprocessing.cpu_count())

    # Use process pool for true parallelism (bypassing GIL)
    with multiprocessing.Pool() as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results from all chunks
    global_min = min(
        result[0] for result in results if result[0] != np.iinfo(np.int64).max
    )
    global_max = max(
        result[1] for result in results if result[1] != np.iinfo(np.int64).min
    )

    return global_max - global_min if global_max >= global_min else 0


if __name__ == "__main__":
    # Warm up numba compilation
    small_test = np.array([1, 2, 3], dtype=np.int32)
    find_difference_chunk(small_test)

    start_time = time.perf_counter()
    for _ in range(10):
        print(main())
    end_time = time.perf_counter() - start_time

    avg_time = end_time / 10

    print(avg_time)
