from __future__ import annotations

import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class NumberRange:
    """Represents a range of numbers to process."""

    start: int
    end: int
    count: int
    target_sum: int

    def __post_init__(self) -> None:
        """Validate the number range parameters."""
        if not all(
            isinstance(x, int)
            for x in (self.start, self.end, self.count, self.target_sum)
        ):
            raise TypeError("All parameters must be integers")
        if self.start >= self.end:
            raise ValueError("Start must be less than end")
        if self.count <= 0:
            raise ValueError("Count must be positive")
        if self.target_sum <= 0:
            raise ValueError("Target sum must be positive")


@dataclass(frozen=True)
class Result:
    """Stores the result of the number analysis."""

    min_num: Optional[int]
    max_num: Optional[int]
    difference: int
    execution_time: float
    numbers_found: int

    @property
    def found_valid_numbers(self) -> bool:
        """Check if any valid numbers were found."""
        return self.min_num is not None and self.max_num is not None


class DigitSumAnalyzer:
    """Analyzes numbers based on their digit sums."""

    def __init__(self, number_range: NumberRange):
        self.number_range = number_range
        self._digit_sums = self._precompute_digit_sums()

    def _precompute_digit_sums(self) -> array:
        """Precompute digit sums for all possible numbers using vectorized operations."""
        digits = np.arange(self.number_range.end + 1, dtype=np.int32)
        digit_sums = np.zeros(self.number_range.end + 1, dtype=np.int32)

        while digits.any():
            digit_sums += digits % 10
            digits //= 10

        return array("B", digit_sums)

    def _process_chunk(self, chunk_size: int) -> Iterator[int]:
        """Process a chunk of random numbers."""
        numbers = np.random.randint(
            self.number_range.start,
            self.number_range.end + 1,
            chunk_size,
            dtype=np.int32,
        )
        mask = (
            np.frombuffer(self._digit_sums, dtype=np.uint8)[numbers]
            == self.number_range.target_sum
        )

        return numbers[mask]

    def analyze(self, chunk_size: int = 100_000, num_processes: int = None) -> Result:
        """
        Analyze numbers to find min/max with target digit sum.

        Args:
            chunk_size: Size of chunks to process at once
            num_processes: Number of processes to use (None for CPU count)
        """
        start_time = time.perf_counter()
        min_num = float("inf")
        max_num = float("-inf")
        numbers_found = 0

        num_chunks = (self.number_range.count + chunk_size - 1) // chunk_size

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    self._process_chunk,
                    min(chunk_size, self.number_range.count - i * chunk_size),
                )
                for i in range(num_chunks)
            ]

            for future in tqdm(futures, desc="Processing chunks"):
                for num in future.result():
                    numbers_found += 1
                    min_num = min(min_num, num)
                    max_num = max(max_num, num)

        execution_time = time.perf_counter() - start_time

        if numbers_found == 0:
            return Result(None, None, 0, execution_time, 0)

        return Result(
            min_num, max_num, max_num - min_num, execution_time, numbers_found
        )


def format_result(result: Result) -> str:
    """Format the result for display."""
    if not result.found_valid_numbers:
        return "No numbers found with the target digit sum."

    return (
        f"\nResults:\n"
        f"{'=' * 40}\n"
        f"Smallest number: {result.min_num:,}\n"
        f"Largest number:  {result.max_num:,}\n"
        f"Difference:      {result.difference:,}\n"
        f"Numbers found:   {result.numbers_found:,}\n"
        f"Execution time:  {result.execution_time:.3f} seconds"
    )


def main():
    """Main execution function."""
    try:
        # Configuration
        number_range = NumberRange(start=1, end=100_000, count=1_000_000, target_sum=30)

        # Set random seed for reproducibility
        # np.random.seed(42)

        # Initialize analyzer
        analyzer = DigitSumAnalyzer(number_range)

        # Run analysis
        result = analyzer.analyze(
            chunk_size=100_000,
            num_processes=None,  # Uses CPU count
        )

        # Print results
        print(format_result(result))

    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
