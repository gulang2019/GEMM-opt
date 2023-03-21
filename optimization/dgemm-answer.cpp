// dgemm.cpp
// DGEMM (Double GEneral Matrix Multiplication)

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <new>
#include <tuple>
#include <string>
#include <vector>

typedef double
	ElementType; // You may change this to see the influence of the data type

static constexpr int kAlignInBytes = 512 / 8;
static constexpr int kAlignInElements =
	kAlignInBytes / sizeof(ElementType); // Keep compatible with AVX-512
static constexpr int kDim = 512;        // kDim may not be pow of 2!
static constexpr int RoundUpToMultiple(int num, int base) {
	return (num + base - 1) / base * base;
}
static constexpr int kDimRoundUp = RoundUpToMultiple(kDim, kAlignInElements);

static constexpr int kRepetitions =
	20; // Due to the flutuation of actual running time, repetitions are
		// necessary

static void DgemmBaseline(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						  const ElementType src_b[kDimRoundUp][kDimRoundUp],
						  ElementType       dst[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			ElementType sum {};
			for (auto k = 0; k < kDim; k++) {
				sum += src_a[i][k] * src_b[k][j];
			}
			dst[i][j] = sum;
		}
	}
}

constexpr int kStrideYmm = 256 / 8 / sizeof(ElementType);
constexpr int kStrideXmm = 128 / 8 / sizeof(ElementType);
constexpr int kBlockSize = kAlignInElements;

static void DgemmParalleled(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						  const ElementType src_b[kDimRoundUp][kDimRoundUp],
						  ElementType       dst[kDimRoundUp][kDimRoundUp]) {
	#pragma omp parallel for 
	for (auto outer_i = 0; outer_i < kDimRoundUp; outer_i += kBlockSize) {
		for (auto i = outer_i; i < outer_i + kBlockSize; i++) {
			for (auto j = 0; j < kDim; j++) {
				ElementType sum {};
				for (auto k = 0; k < kDim; k++) {
					sum += src_a[i][k] * src_b[k][j];
				}
				dst[i][j] = sum;
			}
		}
	}
}

static void DgemmBlocked (const ElementType src_a[kDimRoundUp][kDimRoundUp],
						  const ElementType src_b[kDimRoundUp][kDimRoundUp],
						  ElementType       dst[kDimRoundUp][kDimRoundUp]) {
#pragma omp parallel for 
	for (auto outer_i = 0; outer_i < kDimRoundUp; outer_i += kBlockSize) {
		for (auto i = outer_i; i < outer_i + kBlockSize; i++) {
			for (auto j = 0; j < kDimRoundUp; j++) 
				dst[i][j] = 0;
		}
		for (auto outer_k = 0; outer_k < kDimRoundUp; outer_k += kBlockSize) {
			for (auto outer_j = 0; outer_j < kDimRoundUp;
				 outer_j += kBlockSize) {
				for (auto i = outer_i; i < outer_i + kBlockSize; i++) {
					for (auto j = outer_j; j < outer_j + kBlockSize; j ++) {
						ElementType sum {};
						for (auto k = outer_k; k < outer_k + kBlockSize; k++) {
							sum += src_a[i][k] * src_b[k][j];
						}
						dst[i][j] += sum;
					}
				}
			}
		}
	}
}

static void DgemmOptimized(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						   const ElementType src_b[kDimRoundUp][kDimRoundUp],
						   ElementType       dst[kDimRoundUp][kDimRoundUp]) {
// Here is what your code should appear. You may modify other parts when
// necessary
#pragma omp parallel for
	for (auto outer_i = 0; outer_i < kDimRoundUp; outer_i += kBlockSize) {
		for (auto outer_k = 0; outer_k < kDimRoundUp; outer_k += kBlockSize) {
			for (auto outer_j = 0; outer_j < kDimRoundUp;
				 outer_j += kBlockSize) {
				for (auto i = outer_i; i < outer_i + kBlockSize; i++) {
					for (auto k = outer_k; k < outer_k + kBlockSize; k++) {
						auto multiplier_a = src_a[i][k];
						for (auto j = outer_j; j < outer_j + kBlockSize;
							 j += kStrideYmm) {
							auto multiplier_b = _mm256_load_pd(&src_b[k][j]);
							auto addend_c     = _mm256_load_pd(&dst[i][j]);
							auto result =
								multiplier_a * multiplier_b + addend_c;
							_mm256_store_pd(&dst[i][j], result);
						}
					}
				}
			}
		}
	}
}

static constexpr int kBlocks = kDimRoundUp / kBlockSize;

static void DgemmDeepBlockingYmm(
	const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
	const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
	ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
#pragma omp parallel for
	for (auto outer_i = 0; outer_i < kBlocks; outer_i++) {
		for (auto outer_k = 0; outer_k < kBlocks; outer_k++) {
			for (auto outer_j = 0; outer_j < kBlocks; outer_j++) {
				for (auto i = 0; i < kBlockSize; i++) {
					for (auto k = 0; k < kBlockSize; k++) {
						auto multiplier_a = src_a[outer_i][outer_k][i][k];
						for (auto j = 0; j < kBlockSize; j += kStrideYmm) {
							auto multiplier_b =
								_mm256_load_pd(&src_b[outer_k][outer_j][k][j]);
							auto addend_c =
								_mm256_load_pd(&dst[outer_i][outer_j][i][j]);
							auto result =
								multiplier_a * multiplier_b + addend_c;
							_mm256_store_pd(&dst[outer_i][outer_j][i][j],
											result);
						}
					}
				}
			}
		}
	}
}

static void DgemmDeepBlockingXmm(
	const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
	const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
	ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
#pragma omp parallel for
	for (auto outer_i = 0; outer_i < kBlocks; outer_i++) {
		for (auto outer_k = 0; outer_k < kBlocks; outer_k++) {
			for (auto outer_j = 0; outer_j < kBlocks; outer_j++) {
				for (auto i = 0; i < kBlockSize; i++) {
					for (auto k = 0; k < kBlockSize; k++) {
						auto multiplier_a = src_a[outer_i][outer_k][i][k];
						for (auto j = 0; j < kBlockSize; j += kStrideXmm) {
							auto multiplier_b =
								_mm_load_pd(&src_b[outer_k][outer_j][k][j]);
							auto addend_c =
								_mm_load_pd(&dst[outer_i][outer_j][i][j]);
							auto result =
								multiplier_a * multiplier_b + addend_c;
							_mm_store_pd(&dst[outer_i][outer_j][i][j], result);
						}
					}
				}
			}
		}
	}
}

static void CopyMatrix(
	const ElementType src[kDimRoundUp][kDimRoundUp],
	ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
	for (auto i = 0; i < kDimRoundUp; i++) {
		for (auto j = 0; j < kDimRoundUp; j++) {
			dst[i / kBlockSize][j / kBlockSize][i % kBlockSize]
			   [j % kBlockSize] = src[i][j];
		}
	}
}

static void RandomlyInitMatrix(ElementType matrix[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (int j = 0; j < kDim; j++) {
			matrix[i][j] = std::rand();
		}
	}
}

static void RandomlyInitMatrix(
	ElementType matrix[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
	for (auto outer_i = 0; outer_i < kBlocks; outer_i++) {
		for (auto outer_j = 0; outer_j < kBlocks; outer_j++) {
			for (auto i = 0; i < kBlockSize; i++) {
				for (auto j = 0; j < kBlockSize; j++) {
					matrix[outer_i][outer_j][i][j] = std::rand();
				}
			}
		}
	}
}

constexpr ElementType kEpsilon = static_cast<ElementType>(1e-8);

static bool MatrixEqual(const ElementType result_a[kDimRoundUp][kDimRoundUp],
						const ElementType result_b[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			auto a = result_a[i][j];
			auto b = result_b[i][j];
			if (std::abs(a - b) > kEpsilon * (std::abs(a) + std::abs(b))) {
				return false;
			}
		}
	}
	return true;
}

static bool MatrixEqual(
	const ElementType result_a[kDimRoundUp][kDimRoundUp],
	const ElementType result_b[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			auto a = result_a[i][j];
			auto b = result_b[i / kBlockSize][j / kBlockSize][i % kBlockSize]
							 [j % kBlockSize];
			if (std::abs(a - b) > kEpsilon * (std::abs(a) + std::abs(b))) {
				return false;
			}
		}
	}
	return true;
}

static bool MatrixEqual(
	const ElementType result_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
	const ElementType result_b[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			auto a = result_a[i / kBlockSize][j / kBlockSize][i % kBlockSize]
							 [j % kBlockSize];
			auto b = result_b[i / kBlockSize][j / kBlockSize][i % kBlockSize]
							 [j % kBlockSize];
			if (std::abs(a - b) > kEpsilon * (std::abs(a) + std::abs(b))) {
				return false;
			}
		}
	}
	return true;
}

void PrintMatrix(const ElementType matrix[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			std::cout << matrix[i][j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void PrintMatrix(
	const ElementType matrix[kBlocks][kBlocks][kBlockSize][kBlockSize]) {
	for (auto outer_i = 0; outer_i < kBlocks; outer_i++) {
		for (auto i = 0; i < kBlockSize; i++) {
			for (auto outer_j = 0; outer_j < kBlocks; outer_j++) {
				for (auto j = 0; j < kBlockSize; j++) {
					if ((outer_i * kBlockSize + i >= kDim)
						|| (outer_j * kBlockSize + j >= kDim)) {
						continue;
					}
					std::cout << matrix[outer_i][outer_j][i][j] << ' ';
				}
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

// Helper function to verify that the optimized version is functionally
// correct.
static bool Verify(void (*dgemm_kernel_baseline)(
					   const ElementType src_a[kDimRoundUp][kDimRoundUp],
					   const ElementType src_b[kDimRoundUp][kDimRoundUp],
					   ElementType       dst[kDimRoundUp][kDimRoundUp]),
				   void (*dgemm_kernel_optimized)(
					   const ElementType src_a[kDimRoundUp][kDimRoundUp],
					   const ElementType src_b[kDimRoundUp][kDimRoundUp],
					   ElementType       dst[kDimRoundUp][kDimRoundUp]),
				   int num_trials) {
	for (auto i = 0; i < num_trials; i++) {
		auto src_a = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto src_b = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization

		auto dst_baseline = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto dst_optimized = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization

		RandomlyInitMatrix(src_a);
		RandomlyInitMatrix(src_b);

		dgemm_kernel_baseline(src_a, src_b, dst_baseline);
		dgemm_kernel_optimized(src_a, src_b, dst_optimized);

#ifdef PRINT
		PrintMatrix(dst_baseline);
		PrintMatrix(dst_optimized);
#endif

		if (MatrixEqual(dst_baseline, dst_optimized) == false) {
			// Almost forget these
			operator delete[](src_a, std::align_val_t(kAlignInBytes));
			operator delete[](src_b, std::align_val_t(kAlignInBytes));
			operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
			operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
			return false;
		}

		// Almost forget these
		operator delete[](src_a, std::align_val_t(kAlignInBytes));
		operator delete[](src_b, std::align_val_t(kAlignInBytes));
		operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
		operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
	}
	return true;
}

// Helper function to verify that the optimized version is functionally
// correct. Compare the baseline with the "cache blocking" version.
static bool Verify(
	void (*dgemm_kernel_baseline)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp]),
	void (*dgemm_kernel_optimized)(
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]),
	int num_trials) {
	for (auto i = 0; i < num_trials; i++) {
		auto src_a = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto src_b = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization

		auto src_a_blocking = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Store the same elements as src_a, but
										// with different order to assist cache
										// blocking
		auto src_b_blocking = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Store the same elements as src_b, but
										// with different order to assist cache
										// blocking

		auto dst_baseline = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto dst_optimized = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Zero initialization

		RandomlyInitMatrix(src_a);
		RandomlyInitMatrix(src_b);

		CopyMatrix(src_a, src_a_blocking);
		CopyMatrix(src_b, src_b_blocking);

		dgemm_kernel_baseline(src_a, src_b, dst_baseline);
		dgemm_kernel_optimized(src_a_blocking, src_b_blocking, dst_optimized);

#ifdef PRINT
		PrintMatrix(dst_baseline);
		PrintMatrix(dst_optimized);
#endif

		if (MatrixEqual(dst_baseline, dst_optimized) == false) {
			// Almost forget these
			operator delete[](src_a, std::align_val_t(kAlignInBytes));
			operator delete[](src_b, std::align_val_t(kAlignInBytes));
			operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
			operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
			return false;
		}

		// Almost forget these
		operator delete[](src_a, std::align_val_t(kAlignInBytes));
		operator delete[](src_b, std::align_val_t(kAlignInBytes));
		operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
		operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
	}
	return true;
}

// Helper function to verify that the optimized version is functionally
// correct. Compare two "cache blocking" version.
static bool Verify(
	void (*dgemm_kernel_baseline)(
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]),
	void (*dgemm_kernel_optimized)(
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]),
	int num_trials) {
	for (auto i = 0; i < num_trials; i++) {
		auto src_a_blocking = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Zero initialization
		auto src_b_blocking = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Zero initialization

		auto dst_baseline = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Zero initialization
		auto dst_optimized = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Zero initialization

		RandomlyInitMatrix(src_a_blocking);
		RandomlyInitMatrix(src_b_blocking);

		dgemm_kernel_baseline(src_a_blocking, src_b_blocking, dst_baseline);
		dgemm_kernel_optimized(src_a_blocking, src_b_blocking, dst_optimized);

#ifdef PRINT
		PrintMatrix(dst_baseline);
		PrintMatrix(dst_optimized);
#endif

		if (MatrixEqual(dst_baseline, dst_optimized) == false) {
			// Almost forget these
			operator delete[](src_a_blocking, std::align_val_t(kAlignInBytes));
			operator delete[](src_b_blocking, std::align_val_t(kAlignInBytes));
			operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
			operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
			return false;
		}

		// Almost forget these
		operator delete[](src_a_blocking, std::align_val_t(kAlignInBytes));
		operator delete[](src_b_blocking, std::align_val_t(kAlignInBytes));
		operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
		operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
	}
	return true;
}

typedef decltype(std::chrono::high_resolution_clock::now()
				 - std::chrono::high_resolution_clock::now())
	UnitOfTime; // The unit of high_resolution_clock may vary with systems,
				// so this is necessary to keep the program cross-platform

// Return value: running time of one trial (only the running time of the kernel
// is included). Time is measured by milliseconds.
static double MeasureTime(
	void (*dgemm_kernel)(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						 const ElementType src_b[kDimRoundUp][kDimRoundUp],
						 ElementType       dst[kDimRoundUp][kDimRoundUp])) {
	UnitOfTime total_time_elapsed {};
	for (auto i = 0; i < kRepetitions; i++) {
		auto src_a = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp]; // Not initialized yet
		auto src_b = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp]; // Not initialized yet
		auto dst = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization

		RandomlyInitMatrix(src_a);
		RandomlyInitMatrix(src_b);

		auto start_time_point = std::chrono::high_resolution_clock::now();
		dgemm_kernel(src_a, src_b, dst);
		auto end_time_point = std::chrono::high_resolution_clock::now();

		total_time_elapsed += end_time_point - start_time_point;

		// Almost forget these
		delete[] src_a;
		delete[] src_b;
		delete[] dst;
	}

	using namespace std::chrono_literals;
	return (double)total_time_elapsed.count()/ kRepetitions / 1e6;
}

// Return value: running time of one trial (only the running time of the kernel
// is included). Time is measured by milliseconds. This is for the "cache
// blocking" version.
static double MeasureTime(void (*dgemm_kernel)(
	const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
	const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
	ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize])) {
	UnitOfTime total_time_elapsed {};
	for (auto i = 0; i < kRepetitions; i++) {
		auto src_a = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize]; // Not initialized yet
		auto src_b = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize]; // Not initialized yet
		auto dst = new (std::align_val_t(kAlignInBytes))
			ElementType[kBlocks][kBlocks][kBlockSize]
					   [kBlockSize] {}; // Zero initialization

		RandomlyInitMatrix(src_a);
		RandomlyInitMatrix(src_b);

		auto start_time_point = std::chrono::high_resolution_clock::now();
		dgemm_kernel(src_a, src_b, dst);
		auto end_time_point = std::chrono::high_resolution_clock::now();

		total_time_elapsed += end_time_point - start_time_point;

		// Almost forget these
		operator delete[](src_a, std::align_val_t(kAlignInBytes));
		operator delete[](src_b, std::align_val_t(kAlignInBytes));
		operator delete[](dst, std::align_val_t(kAlignInBytes));
	}

	using namespace std::chrono_literals;
	return (double)total_time_elapsed.count()/ kRepetitions / 1e6;
}

static auto CalculateSpeedup(
	void (*dgemm_kernel_baseline)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp]),
	void (*dgemm_kernel_optimized)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp])) {
	auto running_time_baseline  = MeasureTime(dgemm_kernel_baseline);
	auto running_time_optimized = MeasureTime(dgemm_kernel_optimized);

	return std::tuple {running_time_baseline, running_time_optimized,
					   static_cast<double>(running_time_baseline)
						   / static_cast<double>(running_time_optimized)};
}

static auto CalculateSpeedup(
	void (*dgemm_kernel_baseline)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp]),
	void (*dgemm_kernel_optimized)(
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize])) {
	auto running_time_baseline  = MeasureTime(dgemm_kernel_baseline);
	auto running_time_optimized = MeasureTime(dgemm_kernel_optimized);

	return std::tuple {running_time_baseline, running_time_optimized,
					   static_cast<double>(running_time_baseline)
						   / static_cast<double>(running_time_optimized)};
}

static auto CalculateSpeedup(
	void (*dgemm_kernel_baseline)(
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]),
	void (*dgemm_kernel_optimized)(
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize])) {
	auto running_time_baseline  = MeasureTime(dgemm_kernel_baseline);
	auto running_time_optimized = MeasureTime(dgemm_kernel_optimized);

	return std::tuple {running_time_baseline, running_time_optimized,
					   static_cast<double>(running_time_baseline)
						   / static_cast<double>(running_time_optimized)};
}

#define DEBUG

struct Config {
	void (*baseline) (
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp]) = nullptr;
	void (*optimized)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp]) = nullptr;
	void (*optimized_relayout) (
		const ElementType src_a[kBlocks][kBlocks][kBlockSize][kBlockSize],
		const ElementType src_b[kBlocks][kBlocks][kBlockSize][kBlockSize],
		ElementType       dst[kBlocks][kBlocks][kBlockSize][kBlockSize]) = nullptr;
	bool relayout = false;
	std::string name;
};

int main() {
	std::srand(std::time(nullptr));
	Config configs[] = {
		{DgemmBaseline, DgemmParalleled, nullptr, false, "Paralleled"},
		{DgemmBaseline, DgemmBlocked, nullptr, false, "Blocked"},
		// {DgemmBaseline, nullptr, DgemmDeepBlockingXmm, true, "SIMDizedXmm"},
		{DgemmBaseline, nullptr, DgemmDeepBlockingYmm, true, "SIMDized-AVX2"},
	};

	for (auto & config: configs) {
		std::cout << "===========" << config.name << "===============" << std::endl;
		if (!config.relayout) {
#ifdef DEBUG
			if(Verify(config.baseline, config.optimized, 1) == false) {
				std::cerr << "Functionally incorrect" << std::endl;
				continue;
			}
			else {
				std::cerr << "Functionally correct" << std::endl;
			}
#endif

			auto [running_time_baseline, runnning_time_optimzed, speedup] =
				CalculateSpeedup(config.baseline, config.optimized);
			std::cout << "Average running time of the baseline version is "
					<< running_time_baseline << "ms" << std::endl;
			std::cout << "Average running time of the optimized version is "
					<< runnning_time_optimzed << "ms" << std::endl;
			std::cout << "The speedup is " << speedup << "x" << std::endl;
		}
		else{ // config.relayout = true
#ifdef DEBUG
			if(Verify(config.baseline, config.optimized_relayout, 1) == false) {
				std::cerr << "Functionally incorrect" << std::endl;
				continue;
			}
			else {
				std::cerr << "Functionally correct" << std::endl;
			}
#endif

			auto [running_time_baseline, runnning_time_optimzed, speedup] =
				CalculateSpeedup(config.baseline, config.optimized_relayout);
			std::cout << "Average running time of the baseline version is "
					<< running_time_baseline << "ms" << std::endl;
			std::cout << "Average running time of the optimized version is "
					<< runnning_time_optimzed << "ms" << std::endl;
			std::cout << "The speedup is " << speedup << "x" << std::endl;
		}
		
	}

}