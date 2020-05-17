//
// Created by janphr on 10.05.20.
//

#include "Conv2DLayer.h"

#include <utility>
void Conv2DLayer::forward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*Y = X * F + bias*/
    int filter_size_3d = kernel_size * kernel_size * inShape.d;
    int in_size_2d = inShape.r * inShape.c;
    int f_pos, in2d_f_d, in_c_f_c, in_r_f_r, in_, out_i;
    double curr;
    for (int i = 0; i < in.size(); i++) {
        auto e_ = &out[i].getElements();
        auto _e = &in[i].getElements();
        auto _b = &bias.getElements();
        auto _k = &kernel.getElements();
        for (int f = 0; f < filter_count; f++) {
            for (int r_ = 0; r_ < outShape.r; r_++) {
                for (int c_ = 0; c_ < outShape.c; c_++) {
                    curr = 0;
                    f_pos = f * filter_size_3d;
                    for (int f_d = 0; f_d < inShape.d; f_d++) {
                        in2d_f_d = in_size_2d * f_d;
                        for (int f_c = 0; f_c < kernel_size; f_c++) {
                            in_c_f_c = c_ + f_c;
                            in_ = in_c_f_c * inShape.r + in2d_f_d;
                            for (int f_r = 0; f_r < kernel_size; f_r++) {
                                in_r_f_r = r_ + f_r;
                                if (in_r_f_r >= 0 && in_c_f_c >= 0 && in_r_f_r < inShape.r && in_c_f_c < inShape.c)
                                    curr += _k->coeffRef(f_pos) * _e->coeffRef(in_ + in_r_f_r);
                                f_pos++;
                            }
                        }
                    }
                    out_i = f * outShape.r * outShape.c + c_ * outShape.r + r_;
                    e_->coeffRef(out_i) = curr + _b->coeffRef(f);
                }
            }
        }
//        cout << in[0].getElements().size() << endl;
//        cout << out[0].getElements().size() << endl;
//        cout << in[0].getElements() << endl;
//        cout << out[i].getElements() << endl;
    }
}

void Conv2DLayer::get_filter(Eigen::MatrixXd &f, int n, int size, bool rotate) {
    if (rotate) {
        n += size;
        for (int i = 0; i < size; i++) {
            f.coeffRef(i) = kernel.getElements().coeffRef(n - i - 1);
        }
    } else {
        for (int i = 0; i < size; i++) {
            f.coeffRef(i) = kernel.getElements().coeffRef(n + i);
        }
    }
}


void Conv2DLayer::backward(vector<Tensor> &in, vector<Tensor> &out, int idx) {
    /*dX = dY *_{F} rot_{180}(trans_{0,1,3,2}(F))*/
    /*dL/df = X *_{ch} dY*/

    int full_pad = kernel_size - 1;
//    Eigen::MatrixXd e(1, kernel_size * kernel_size);
//    Eigen::MatrixXd k(1, kernel_size * kernel_size);
    for (int i = 0; i < in.size(); i++) {
        auto _e = &out[i].getElements();
        auto _d = &in[i].getDeltas();
        auto f_dim = kernel_size * kernel_size, fs_dim = inShape.d * kernel_size * kernel_size,
                out_dim = outShape.r * outShape.c, in_dim = inShape.r * inShape.c;
        int pad_idx_row, pad_idx_col, o_col, o_row, delta_idx;
        double delta_y;
        Eigen::MatrixXd f(1, f_dim);


        for (int fs = 0; fs < filter_count; fs++) {
            pad_idx_col = -full_pad;
            for (int col = 0; col < outShape.c; pad_idx_col++, col++) {
                pad_idx_row = -full_pad;
                for (int row = 0; row < outShape.r; pad_idx_row++, row++) {
                    delta_y = _d->coeffRef(fs * out_dim + outShape.r * col + row);
                    delta_idx = 0;
                    for (int f_ch = 0; f_ch < inShape.d; f_ch++) {
                        for (int f_col = 0; f_col < kernel_size; f_col++) {
                            for (int f_row = 0; f_row < kernel_size; f_row++) {
                                kernel.getDeltas().coeffRef(fs * fs_dim + delta_idx) +=
                                        delta_y *
                                        _e->coeffRef(f_ch * in_dim + (col + f_col) * inShape.r + f_row + row);
                                delta_idx++;
                            }
                        }
                    }
                    bias.getDeltas().coeffRef(fs) += delta_y;
                }
            }
        }

//        int in_full_rows = outShape.r + 2 * (kernel_size - 1);
//        int in_full_cols = outShape.c + 2 * (kernel_size - 1);
//        Shape in_full_shape = Shape(in_full_rows, in_full_cols, filter_count);
//        Eigen::MatrixXd in_full = Eigen::MatrixXd::Zero(1, in_full_shape.r*in_full_shape.c*in_full_shape.d);
//        int in_idx = 0;
//        for (int f = 0; f < filter_count; f++) {
//            for (int out_ch = 0; out_ch < inShape.d; out_ch++) {
//                for (int in_full_idx_col = 0; in_full_idx_col < in_full_cols; in_full_idx_col++) {
//                    for (int in_full_idx_row = 0; in_full_idx_row < in_full_rows; in_full_idx_row++) {
//                        if (in_full_idx_row >= 0 && in_full_idx_row < outShape.r && in_full_idx_col >= 0 && in_full_idx_col < outShape.c) {
//                            in_full.coeffRef(f * inShape.d * in_full_rows * in_full_cols + out_ch * in_full_rows * in_full_cols + in_full_idx_col * in_full_rows +
//                                             in_full_idx_row) = in[i].getDeltas().coeffRef(in_idx);
//                            in_idx++;
//                        }
//                    }
//                }
//            }
//        }
////        cout << in_full << endl;
//        for (int f = 0; f < filter_count; f++) {
//            for (int in_ch = 0; in_ch < filter_count; in_ch++) {
//                for (int out_ch = 0; out_ch < outShape.d; out_ch++) {
//                    for (int w_ = 0; w_ < inShape.c; w_++) {
//                        for (int h_ = 0; h_ < inShape.r; h_++) {
//                            getInforKernel(in_full, e, in_full_shape, out_ch, h_, w_);
//                            getW_(k, out_ch, in_ch);
//                            out[i].getDeltas().coeffRef(getOutIndex(out_ch, h_, w_, inShape)) += conv(e, k, true);
//                        }
//                    }
//                }
//            }
//        }
    }
//    cout << bias.getDeltas() << endl;
//    cout << kernel.getDeltas() << endl;
}



void Conv2DLayer::update(SGDTrainer &trainer) {
    trainer.optimize(this->kernel);
    trainer.optimize(this->bias);
}

int Conv2DLayer::getOutIndex(int out, int h_, int w_, Shape &shape) {
    return out * shape.r * shape.c + w_ * shape.r + h_;
}

void
Conv2DLayer::getInforKernel(Eigen::MatrixXd &in, Eigen::MatrixXd &out, const Shape &shape, int f, int ch, int h_,
                            int w_) {
    for (int w = 0; w < kernel_size; w++) {
        for (int h = 0; h < kernel_size; h++) {
            out.coeffRef(w * kernel_size + h) = in.coeffRef(
                    f * inShape.d * shape.c * shape.r + ch * shape.r * shape.c + (w_ + w) * shape.r + h + h_);
        }
    }
}


void Conv2DLayer::getW_(Eigen::MatrixXd &r, int in, int out) {
    for (size_t i = 0; i < r.size(); i++)
        r.coeffRef(i) = kernel.getElements().coeffRef(out * outShape.d * kernel_size * kernel_size
                                                      + in * kernel_size * kernel_size + i);
}

double Conv2DLayer::conv(Eigen::MatrixXd &a, Eigen::MatrixXd &b, bool rotate) {
    double sum = 0, size = a.size();
    for (int i = 0; i < size; i++) {
        sum += a.coeffRef(i) * b.coeffRef(rotate ? size - i - 1 : i);
    }
    return sum;
}

Conv2DLayer::Conv2DLayer(Tensor kernel, const Shape &inShape, const Shape &outShape, const int kernelSize,
                         const int filterCount) : kernel(std::move(kernel)), inShape(inShape), outShape(outShape),
                                                  kernel_size(kernelSize), filter_count(filterCount) {
    Eigen::MatrixXd bias_mat = Eigen::MatrixXd::Zero(1, filterCount);
    Eigen::MatrixXd bias_mat_delta = Eigen::MatrixXd::Zero(1, filterCount);
    Tensor kernel_bias = *new Tensor(bias_mat, bias_mat_delta);
//    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(inShape.r - kernelSize + 1, inShape.c - kernelSize + 1);
    this->bias = kernel_bias;
}




