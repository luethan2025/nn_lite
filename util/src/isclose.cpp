/**
 * @file isclose.cpp
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief C++ implementation of NumPy's isclose function
 * @date 2024-03-25
 * 
 * @copyright Copyright (c) 2024
 */
#include "isclose.h"
#include <stdlib.h>

/**
 * @brief C++ implementation of NumPy's isclose function
 * 
 * @note
 * This function is not symmetric. Let x1 be reference value and let x2 be the
 * comparing value:
 *   isclose(x1, x2) != isclose(x2, x1)
 *
 * @param a reference value
 * @param b comparing value
 * @param rtol relative tolerance
 * @param atol absolute tolerance
 * @return true if abs(a - b) <= (atol + rtol * abs(b)) is true, false
 *         otherwise
 */
bool isclose(double a, double b, double rtol, double atol) {
  return abs(a - b) <= (atol + rtol * abs(b));
}
