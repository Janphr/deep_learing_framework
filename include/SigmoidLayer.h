//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

using namespace Eigen;

class SigmoidLayer : public Layer {
public:
    SigmoidLayer();

    /*Y = sigma(X) = 1/(1+e^-X) | element wise*/
    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    /*dX = [sima(X) * (1 - sigma(X)] (multiplied elementwise(hadama product?)) dY*/
    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;
};


