/**
 * @file sigmoid.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Sigmoid object definition
 * @date 2024-03-26
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include <vector>
#include <stdexcept>

using std::vector;
using std::runtime_error;

class Sigmoid {
  public:
    vector<double> forward(const vector<double> &x);
    vector<double> backward(vector<double> grad);

    vector<double> out_vals; /* output vector from forward pass */
};

#endif /* _SIGMOID_H_ */
