//
// Created by janphr on 17.05.20.
//

#include "MaxPooling.h"

void MaxPooling::forward(vector<Tensor> &in, vector<Tensor> &out, int idx) {

    int in_i, out_i;
    double curr;
    for (int i = 0; i < in.size(); i++) {
        auto e_ = &out[i].getElements();
        auto _e = &in[i].getElements();
        auto out_dim = outShape.r * outShape.c, in_dim = inShape.r * inShape.c;
        for (int f = 0; f < filter_count; f++) {
            for (int r_ = 0; r_ < inShape.r; r_++) {
                for (int c_ = 0; c_ < inShape.c; c_++) {

                    out_i = f*out_dim + (c_/poolingSize) * outShape.r + (r_/poolingSize);
                    in_i = f * in_dim + c_ * inShape.r + r_;

                    curr = _e->coeffRef(in_i);
                    if(get<1>(out[i].getPoolingVec().at(out_i)) < curr) {
                        out[i].getPoolingVec().at(out_i) = tuple<int, double>(in_i, curr);
                        e_->coeffRef(out_i) = curr;
                    }
                }
            }
        }
//        cout << in[0].getElements().size() << endl;
//        cout << out[0].getElements().size() << endl;
//        cout << in[0].getElements() << endl;
//        cout << out[i].getElements() << endl;
    }
}

void MaxPooling::backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    for (int i = 0; i < in.size(); i++) {
        auto _e = &out[i].getElements();
        auto _d = &in[i].getDeltas();
        auto out_dim = outShape.r * outShape.c, in_dim = inShape.r * inShape.c;
        int in_i, out_i;

        for (int f = 0; f < filter_count; f++) {
            for (int c_ = 0; c_ < inShape.c; c_++) {
                for (int r_ = 0; r_ < inShape.r; r_++) {

                    in_i = f * out_dim + (c_/poolingSize) * outShape.r + (r_/poolingSize);
                    out_i = f * in_dim + c_ * inShape.r + r_;


                    if(get<0>(in[i].getPoolingVec().at(in_i)) == f * out_dim + outShape.r * c_ + r_){
                        _e->coeffRef(out_i) = _d->coeffRef(in_i);
                    } else {
                        _e->coeffRef(out_i) = 0;
                    }

                }
            }
        }
    }
}

MaxPooling::MaxPooling(const int poolingSize, const int filterCount, const Shape &inShape, const Shape &outShape)
        : poolingSize(poolingSize), filter_count(filterCount), inShape(inShape), outShape(outShape) {}
