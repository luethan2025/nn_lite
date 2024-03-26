/**
 * @file linear.h
 * @author Ethan Lu (https://github.com/luethan2025)
 * @brief Linear object definition
 * @date 2024-03-24
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <vector>
#include <string>
#include <stdexcept>
#include <random>

using std::vector;
using std::string;
using std::runtime_error;
using std::normal_distribution;
using std::default_random_engine;

class Linear {
  public:
    int in_dim;                     /* input dimension of the layer   */
    int out_dim;                    /* output dimension of the layer  */
    vector<double> in_vals;         /* input vector from forward pass */
    double lr;                      /* learning rate                  */
    vector<vector<double>> weights; /* weight matrix                  */
    vector<double> bias;            /* bias vector                    */

    Linear(int input_size, int output_size, double lr, string initialization);
    vector<double> forward(const vector<double> &x);
    vector<double> backward(const vector<double> &grad);    
};

#endif /* _LINEAR_H_ */
