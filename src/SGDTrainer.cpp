//
// Created by janphr on 29.04.20.
//

#include "SGDTrainer.h"


void SGDTrainer::optimize(Tensor &data) {

        data.getElements().array() -= this->learningRate * data.getDeltas().array();
}

SGDTrainer::SGDTrainer(int amountEpochs, float learningRate) : amountEpochs(amountEpochs), learningRate(learningRate) {}

int SGDTrainer::getAmountEpochs() const {
    return amountEpochs;
}
