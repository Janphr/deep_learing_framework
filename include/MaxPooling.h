//
// Created by janphr on 17.05.20.
//

#ifndef FRAMEWORK_MAXPOOLING_H
#define FRAMEWORK_MAXPOOLING_H


#include "Layer.h"

class MaxPooling  : public Layer {
public:
    MaxPooling(const int poolingSize, const int filterCount, const Shape &inShape, const Shape &outShape);

    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

private:
    const int poolingSize;
    const int filter_count;
    Shape inShape;
    Shape outShape;
};


#endif //FRAMEWORK_MAXPOOLING_H
