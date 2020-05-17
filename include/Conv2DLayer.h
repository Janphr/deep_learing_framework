//
// Created by janphr on 10.05.20.
//

#ifndef FRAMEWORK_CONV2DLAYER_H
#define FRAMEWORK_CONV2DLAYER_H


#include "Layer.h"

class Conv2DLayer : public Layer {
public:

    Conv2DLayer(Tensor kernel, const Shape &inShape, const Shape &outShape, int kernelSize,
                int filterCount);



    void forward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    void backward(vector<Tensor> &in, vector<Tensor> &out, int idx) override;

    void update(SGDTrainer &trainer) override;

private:
    Tensor kernel;
    Tensor bias;
    Shape inShape;
    Shape outShape;
    const int kernel_size;
    const int filter_count;


    void get_filter(Eigen::MatrixXd &f, int n, int size, bool rotate);
    int getOutIndex(int out, int h_, int w_, Shape &shape);
    void getInforKernel(Eigen::MatrixXd &in, Eigen::MatrixXd &out, const Shape &shape, int f, int ch, int h_, int w_);
    void getW_(Eigen::MatrixXd &r, int in, int out);
    double conv(Eigen::MatrixXd &a, Eigen::MatrixXd &b, bool rotate);
};


#endif //FRAMEWORK_CONV2DLAYER_H
