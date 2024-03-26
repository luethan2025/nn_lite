/**
 * @file linear_test_util.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Linear testing function definitions
 * @date 2024-03-25
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef _LINEAR_TEST_UTIL_H_
#define _LINEAR_TEST_UTIL_H_

#include "linear.h"
#include "print_util.h"
#include <vector>
#include <stdexcept>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

void test_forward_pass(Linear layer, const vector<double> &x);

#endif  /* _LINEAR_TEST_UTIL_H_ */

