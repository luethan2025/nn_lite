#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
using namespace std;

class Linear {
  private:
    int in_dim;
    int out_dim;
    vector<vector<double>> weights;
    vector<double> bias;
  public:
    Linear(int input_size, int output_size, string initialization);
    vector<vector<double>> get_weights();
    vector<double> get_bias();
    vector<double> forward(vector<double> x);
};

#endif
