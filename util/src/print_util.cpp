/**
 * @file print_util.cpp
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Forward pass and backpropagation message printing functions
 * @date 2024-03-25
 * 
 * @copyright Copyright (c) 2024
 */
#include "print_util.h"

/**
 * @brief Prints vector to cout
 * 
 * @param v Input vector
 */
void print_vector(const vector<double> &v) {
  cout << "{";
  for (const auto& i: v) {
    cout << i;
    if (&i != &v.back()) {
      cout << ", ";
    }
  }
  cout << "}";
}

/**
 * @brief Prints the message "Testing forward pass with vector..." to cout
 * 
 * @param v Input vector
 */
void print_forward_pass_message(const vector<double> &v) {
  cout << "Testing forward pass with vector ";
  print_vector(v);
  cout << endl;
}

/**
 * @brief Prints the message "Testing backpropagation with vector..." to cout
 * 
 * @param v Input vector
 */
void print_backpropagation_message(const vector<double> &v) {
  cout << "Testing backpropagation with vector ";
  print_vector(v);
  cout << endl;
}
