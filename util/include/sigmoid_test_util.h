/**
 * @file sigmoid_test_util.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Sigmoid testing function definitions
 * @date 2024-03-26
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef _SIGMOID_TEST_UTIL_H_
#define _SIGMOID_TEST_UTIL_H_

#include "sigmoid.h"
#include "print_util.h"
#include <vector>
#include <stdexcept>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

void test_forward_pass(Sigmoid layer, const vector<double> &x);

#endif  /* _SIGMOID_TEST_UTIL_H_ */
