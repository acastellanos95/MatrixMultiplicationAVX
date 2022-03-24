  #include <iostream>
  #include "Utils.h"

  int main() {
    // Integer multiplication
//    auto A = std::vector<std::vector<int>>(1500, std::vector<int>(3000,0));
//    auto B = std::vector<std::vector<int>>(3000, std::vector<int>(1500,0));
//    initMatrix(A);
//    initMatrix(B);
//    // Normal multiplication
//    auto ti = clock();
//    auto normalRes = normalMultiplication(A,B);
//    auto tf = clock();
//    auto time = (((float)tf - (float)ti) / CLOCKS_PER_SEC );
//    std::cout << "Tiempo normal de multiplicación: " << std::to_string(time) << '\n';
//    // AVX optimized multiplication
//    ti = clock();
//    auto AVXRes = instrinsicMultiplication(A,B);
//    tf = clock();
//    time = (((float)tf - (float)ti) / CLOCKS_PER_SEC );
//    std::cout << "Tiempo AVX de multiplicación: " << std::to_string(time) << '\n';
//    // Compare
//    if(normalRes != AVXRes)
//      std::cout << "No dan el mismo resultado\n";

    // Float multiplication
    auto AFloat = std::vector<std::vector<float>>(4500, std::vector<float>(5000,0.0));
    auto BFloat = std::vector<std::vector<float>>(5000, std::vector<float>(6000,0.0));
    initMatrix(AFloat);
    initMatrix(BFloat);
    // Normal multiplication
    auto ti = clock();
    auto normalFloatRes = normalMultiplication(AFloat,BFloat);
    auto tf = clock();
    auto time = (((float)tf - (float)ti) / CLOCKS_PER_SEC );
    std::cout << "Tiempo normal matriz flotante de multiplicación: " << std::to_string(time) << '\n';
    // AVX optimized multiplication using transpose
    transpose(BFloat);
    ti = clock();
    auto AVXFloatRes = instrinsicMultiplication(AFloat,BFloat);
    tf = clock();
    time = (((float)tf - (float)ti) / CLOCKS_PER_SEC );
    std::cout << "Tiempo AVX matriz flotante de multiplicación: " << std::to_string(time) << '\n';
    maxError(normalFloatRes, AVXFloatRes);
    return 0;
  }

