//
// Created by janphr on 29.04.20.
//

#pragma once

#include "Layer.h"

using namespace Eigen;

class SoftmaxLayer : public Layer  {

public:

    /*Y = softmax(X) = e^x_i/sum(e^x_i) | cancel e^MAX */
    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    /*dX = dY * jacobi(dX/dY)*/
    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

};