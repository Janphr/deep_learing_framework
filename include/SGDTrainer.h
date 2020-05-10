//
// Created by janphr on 29.04.20.
//

#pragma once


#include "Tensor.h"

class SGDTrainer {

private:
//    int batchSize;
    int amountEpochs;
    float learningRate;
//    bool shuffle;

public:
    SGDTrainer(int amountEpochs, float learningRate);

    void optimize(Tensor &data);

    int getAmountEpochs() const;

};
