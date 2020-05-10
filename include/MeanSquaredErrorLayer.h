//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

class MeanSquaredErrorLayer : public Layer {

public:

    /*L = sum(1/2(X_i - t_i)^2*/
    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    /*dL/dX_i = X_i - t_i*/
    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

};
