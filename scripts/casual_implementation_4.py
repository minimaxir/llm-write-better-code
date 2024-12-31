from __future__ import annotations

import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final, Optional, TypeAlias

import numpy as np
import numpy.typing as npt
import psutil
from tqdm.auto import tqdm

# Type aliases
IntArray: TypeAlias = npt.NDArray[np.int_]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# Constants
DEFAULT_CHUNK_SIZE: Final = 100_000
MIN_CHUNK_SIZE: Final = 1_000
MAX_MEMORY_PERCENT: Final = 75


@dataclass(frozen=True, slots=True)
class SearchParameters:
    """Parameters for the number search."""

    min_value: int
    max_value: int
    sample_size: int
    target_sum: int
    chunk_size: int = field(default=DEFAULT_CHUNK_SIZE)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate parameters."""
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.target_sum <= 0:
            raise ValueError("target_sum must be positive")
        if self.chunk_size < MIN_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be at least {MIN_CHUNK_SIZE}")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of the number search."""

    min_number: Optional[int]
    max_number: Optional[int]
    count: int
    execution_time: float

    @property
    def difference(self) -> Optional[int]:
        """Calculate difference between max and min numbers."""
        if self.min_number is None or self.max_number is None:
            return None
        return self.max_number - self.min_number


class MemoryManager:
    """Manages memory usage for large computations."""

    @staticmethod
    def get_optimal_chunk_size(array_size: int, dtype_size: int) -> int:
        """Calculate optimal chunk size based on available memory."""
        available_memory = psutil.virtual_memory().available
        max_allowed_memory = available_memory * MAX_MEMORY_PERCENT // 100
        max_elements = max_allowed_memory // dtype_size
        return min(array_size, max(MIN_CHUNK_SIZE, max_elements))

    @staticmethod
    @contextmanager
    def monitor_memory():
        """Context manager to monitor memory usage."""
        try:
            yield
        finally:
            if psutil.virtual_memory().percent > MAX_MEMORY_PERCENT:
                warnings.warn("High memory usage detected", ResourceWarning)


class DigitSumCalculator:
    """Handles digit sum calculations with caching."""

    def __init__(self, max_value: int):
        self.max_value = max_value
        self._setup_logging()
        self._digit_sums = self._precompute_digit_sums()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("digit_sum.log"),
            ],
        )

    @staticmethod
    @lru_cache(maxsize=1024)
    def _calculate_digit_sum(num: int) -> int:
        """Calculate sum of digits with caching."""
        return sum(int(d) for d in str(num))

    def _precompute_digit_sums(self) -> IntArray:
        """Precompute digit sums using vectorized operations."""
        logging.info("Precomputing digit sums...")

        with MemoryManager.monitor_memory():
            numbers = np.arange(self.max_value + 1, dtype=np.int32)
            digit_sums = np.zeros_like(numbers)

            while np.any(numbers):
                digit_sums += numbers % 10
                numbers //= 10

        logging.info("Digit sums precomputation completed")
        return digit_sums

    def get_digit_sums(self, numbers: IntArray) -> IntArray:
        """Get digit sums for an array of numbers."""
        return self._digit_sums[numbers]


class NumberAnalyzer:
    """Analyzes numbers based on their digit sums."""

    def __init__(self, params: SearchParameters):
        self.params = params
        self.calculator = DigitSumCalculator(params.max_value)
        self._optimal_chunk_size = self._calculate_optimal_chunk_size()

    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on system resources."""
        return MemoryManager.get_optimal_chunk_size(
            self.params.sample_size, np.dtype(np.int32).itemsize
        )

    def _process_chunk(self, size: int) -> tuple[Optional[int], Optional[int], int]:
        """Process a chunk of random numbers."""
        numbers = np.random.randint(
            self.params.min_value, self.params.max_value + 1, size, dtype=np.int32
        )

        mask = self.calculator.get_digit_sums(numbers) == self.params.target_sum
        matching_numbers = numbers[mask]

        if len(matching_numbers) == 0:
            return None, None, 0

        return matching_numbers.min(), matching_numbers.max(), len(matching_numbers)

    def analyze(self) -> SearchResult:
        """Perform the number analysis using parallel processing."""
        start_time = time.perf_counter()
        chunk_size = min(self._optimal_chunk_size, self.params.chunk_size)
        num_chunks = (self.params.sample_size + chunk_size - 1) // chunk_size

        min_number = float("inf")
        max_number = float("-inf")
        total_count = 0

        logging.info(f"Starting analysis with {num_chunks} chunks")

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_chunk,
                    min(chunk_size, self.params.sample_size - i * chunk_size),
                )
                for i in range(num_chunks)
            ]

            with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
                for future in as_completed(futures):
                    chunk_min, chunk_max, count = future.result()
                    if chunk_min is not None:
                        min_number = min(min_number, chunk_min)
                        max_number = max(max_number, chunk_max)
                        total_count += count
                    pbar.update(1)

        execution_time = time.perf_counter() - start_time

        if total_count == 0:
            return SearchResult(None, None, 0, execution_time)

        return SearchResult(min_number, max_number, total_count, execution_time)


def format_result(result: SearchResult) -> str:
    """Format search result for display."""
    if result.min_number is None:
        return "No numbers found with the target digit sum."

    return (
        f"\nSearch Results\n"
        f"{'=' * 50}\n"
        f"Minimum number: {result.min_number:,}\n"
        f"Maximum number: {result.max_number:,}\n"
        f"Difference: {result.difference:,}\n"
        f"Numbers found: {result.count:,}\n"
        f"Execution time: {result.execution_time:.2f} seconds"
    )


def main() -> None:
    """Main execution function."""
    try:
        params = SearchParameters(
            min_value=1, max_value=100_000, sample_size=1_000_000, target_sum=30
        )

        # Set random seed for reproducibility
        np.random.seed(42)

        analyzer = NumberAnalyzer(params)
        result = analyzer.analyze()

        print(format_result(result))

    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
