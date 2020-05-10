//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

using namespace Eigen;

class CrossEntropyLayer : public Layer {

public:

    CrossEntropyLayer(vector<Tensor> targets);

    /*L = -sum(t_i * log(x_i)*/
    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    /*dL/dX_i = -t_i/X_i*/
    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    void update(SGDTrainer &trainer) override;

private:
    vector<Tensor> targets;
    int targetIndex = 0;

};
