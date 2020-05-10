//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

using namespace Eigen;

class InputLayer{
public:
    void convert(vector<vector<double>> &in, vector<vector<Tensor>> &out);
    void convert_targets(vector<vector<double>> &in, vector<Tensor> &out);

    void convert(const string& filename, vector<vector<Tensor>> &out, int amount);
    void convert_targets(const string& filename, vector<Tensor> &out, int amount, int classes);
};




