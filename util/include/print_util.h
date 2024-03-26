/**
 * @file print_util.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Forward pass and backpropagation message printing functions
 *        definitions
 * @date 2024-03-25
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef _PRINT_UTIL_H_
#define _PRINT_UTIL_H_

#include "linear.h"
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

void print_vector(const vector<double> &v);
void print_forward_pass_message(const vector<double> &v);
void print_backpropagation_message(const vector<double> &v);

#endif  /* _PRINT_UTIL_ */
