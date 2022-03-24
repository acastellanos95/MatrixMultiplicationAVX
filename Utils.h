//
// Created by andre on 3/18/22.
//

#ifndef MATRIXMULTIPLICATIONSIMD__UTILS_H_
#define MATRIXMULTIPLICATIONSIMD__UTILS_H_

#include <vector>
#include <random>
#include <iostream>
#include <immintrin.h>

template<typename T>
std::vector<std::vector<T>> normalMultiplication(const std::vector<std::vector<T>> &A,
                                                 const std::vector<std::vector<T>> &B) {
  auto C = std::vector<std::vector<T>>(A.size(), std::vector<T>(B[0].size(), 0));
  if (A[0].size() == B.size()) {
    for (size_t i = 0; i < A.size(); i++) {
      for (size_t j = 0; j < B[0].size(); j++) {
        for (size_t k = 0; k < B.size(); k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  } else {
    throw std::length_error("El número de columnas de A no es igual al número de filas de B");
  }
  return C;
}

//template<typename T>
std::vector<std::vector<float>> instrinsicMultiplication(const std::vector<std::vector<float>> &A,
                                                         const std::vector<std::vector<float>> &B) {
  auto C = std::vector<std::vector<float>>(A.size(), std::vector<float>(B.size(), 0.0));

  const auto col = B[0].size();
  const auto col_reduced = col - col % 64;
  const auto col_reduced_32 = col - col % 32;

  auto scratchpad = std::vector<float>(8, 0.0);
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
      ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

  if (A[0].size() == B[0].size()) {
    for (auto rowIndex = 0; rowIndex < A.size(); rowIndex++) {
      for (auto colIndex = 0; colIndex < B.size(); colIndex++) {
        // K sum optimized
        float res = 0.0;
        for (auto j = 0; j < col_reduced; j += 64) {
          // Load 8 row elements
          ymm0 = __builtin_ia32_loadups256(&A[rowIndex][j]);
          ymm1 = __builtin_ia32_loadups256(&A[rowIndex][j + 8]);
          ymm2 = __builtin_ia32_loadups256(&A[rowIndex][j + 16]);
          ymm3 = __builtin_ia32_loadups256(&A[rowIndex][j + 24]);
          ymm4 = __builtin_ia32_loadups256(&A[rowIndex][j + 32]);
          ymm5 = __builtin_ia32_loadups256(&A[rowIndex][j + 40]);
          ymm6 = __builtin_ia32_loadups256(&A[rowIndex][j + 48]);
          ymm7 = __builtin_ia32_loadups256(&A[rowIndex][j + 56]);

          // Load 8 col elements
          ymm8 = __builtin_ia32_loadups256(&B[colIndex][j]);
          ymm9 = __builtin_ia32_loadups256(&B[colIndex][j + 8]);
          ymm10 = __builtin_ia32_loadups256(&B[colIndex][j + 16]);
          ymm11 = __builtin_ia32_loadups256(&B[colIndex][j + 24]);
          ymm12 = __builtin_ia32_loadups256(&B[colIndex][j + 32]);
          ymm13 = __builtin_ia32_loadups256(&B[colIndex][j + 40]);
          ymm14 = __builtin_ia32_loadups256(&B[colIndex][j + 48]);
          ymm15 = __builtin_ia32_loadups256(&B[colIndex][j + 56]);

          ymm0 = _mm256_mul_ps(ymm0, ymm8);
          ymm1 = _mm256_mul_ps(ymm1, ymm9);
          ymm2 = _mm256_mul_ps(ymm2, ymm10);
          ymm3 = _mm256_mul_ps(ymm3, ymm11);
          ymm4 = _mm256_mul_ps(ymm4, ymm12);
          ymm5 = _mm256_mul_ps(ymm5, ymm13);
          ymm6 = _mm256_mul_ps(ymm6, ymm14);
          ymm7 = _mm256_mul_ps(ymm7, ymm15);

          ymm0 = _mm256_add_ps(ymm0, ymm1);
          ymm2 = _mm256_add_ps(ymm2, ymm3);
          ymm4 = _mm256_add_ps(ymm4, ymm5);
          ymm6 = _mm256_add_ps(ymm6, ymm7);
          ymm0 = _mm256_add_ps(ymm0, ymm2);
          ymm4 = _mm256_add_ps(ymm4, ymm6);
          ymm0 = _mm256_add_ps(ymm0, ymm4);

          __builtin_ia32_storeups256(&scratchpad[0], ymm0);

          for (int k = 0; k < 8; k++) {
            res += scratchpad[k];
            scratchpad[k] = 0;
          }
        }
        for (auto j = col_reduced; j < col_reduced_32; j += 32) {
          ymm0 = __builtin_ia32_loadups256(&A[rowIndex][j]);
          ymm1 = __builtin_ia32_loadups256(&A[rowIndex][j + 8]);
          ymm2 = __builtin_ia32_loadups256(&A[rowIndex][j + 16]);
          ymm3 = __builtin_ia32_loadups256(&A[rowIndex][j + 24]);

          ymm8 = __builtin_ia32_loadups256(&B[colIndex][j]);
          ymm9 = __builtin_ia32_loadups256(&B[colIndex][j + 8]);
          ymm10 = __builtin_ia32_loadups256(&B[colIndex][j + 16]);
          ymm11 = __builtin_ia32_loadups256(&B[colIndex][j + 24]);

          ymm0 = _mm256_mul_ps(ymm0, ymm8);
          ymm1 = _mm256_mul_ps(ymm1, ymm9);
          ymm2 = _mm256_mul_ps(ymm2, ymm10);
          ymm3 = _mm256_mul_ps(ymm3, ymm11);

          ymm0 = _mm256_add_ps(ymm0, ymm1);
          ymm2 = _mm256_add_ps(ymm2, ymm3);
          ymm0 = _mm256_add_ps(ymm0, ymm2);

          __builtin_ia32_storeups256(&scratchpad[0], ymm0);

          for (int k = 0; k < 8; k++) {
            res += scratchpad[k];
            scratchpad[k] = 0;
          }

        }
        for (auto l = col_reduced_32; l < col; l++) {
          res += A[rowIndex][l] * B[colIndex][l];
        }
        C[rowIndex][colIndex] = res;
      }
    }
    return C;
  } else {
    throw std::length_error("El número de columnas de A no es igual al número de filas de B");
  }
}

void maxError(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B) {
  float maxError = 0.0;
  if (A.size() == B.size() && A[0].size() == B[0].size()) {
    for (size_t rowIndex = 0; rowIndex < A.size(); ++rowIndex) {
      for (size_t colIndex = 0; colIndex < A[0].size(); ++colIndex) {
        maxError = std::max(std::abs(A[rowIndex][colIndex] - B[rowIndex][colIndex]), maxError);
      }
    }
    std::cout << "Mayor error: " << std::to_string(maxError) << '\n';
  } else {
    throw std::length_error("Tus matrices no tienen el mismo tamaño");
  }
}

template<typename T>
void initMatrix(std::vector<std::vector<T>> &A) {
  if constexpr (std::is_integral_v<T>) {
    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<T> uniform_dist(0, 5);
    for (size_t indexRow = 0; indexRow < A.size(); ++indexRow) {
      for (size_t indexCol = 0; indexCol < A[indexRow].size(); ++indexCol) {
        A[indexRow][indexCol] = uniform_dist(e);
      }
    }
  } else if (std::is_floating_point_v<T>) {
    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<T> uniform_dist(0, 5);
    for (size_t indexRow = 0; indexRow < A.size(); ++indexRow) {
      for (size_t indexCol = 0; indexCol < A[indexRow].size(); ++indexCol) {
        A[indexRow][indexCol] = uniform_dist(e);
      }
    }
  }
}

template<typename T>
void transpose(std::vector<std::vector<T>> &data) {
  // this assumes that all inner vectors have the same size and
  // allocates space for the complete result in advance
  std::vector<std::vector<T>> result(data[0].size(), std::vector<T>(data.size(), 0.0));
  for (size_t i = 0; i < data[0].size(); i++)
    for (size_t j = 0; j < data.size(); j++)
      result[i][j] = data[j][i];
  data.clear();
  data = std::move(result);
}

template<typename T>
void printMatrix(std::vector<std::vector<T>> &A) {
  for (size_t indexRow = 0; indexRow < A.size(); ++indexRow) {
    for (size_t indexCol = 0; indexCol < A[indexRow].size(); ++indexCol) {
      std::cout << " " << std::to_string(A[indexRow][indexCol]) << " ";
    }
    std::cout << '\n';
  }
}

#endif //MATRIXMULTIPLICATIONSIMD__UTILS_H_
