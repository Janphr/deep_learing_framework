//
// Created by janphr on 29.04.20.
//

#include <fstream>
#include "InputLayer.h"

void InputLayer::convert(vector<vector<double>> &in, vector<vector<Tensor>> &out) {

    for(auto i : in){
        MatrixXd mat;

        mat = Map<MatrixXd>(i.data(), 1, i.size());

        auto *t = new Tensor(mat);
        vector<Tensor> tv = {*t};
        t->getShape().r = mat.rows();
        t->getShape().c = mat.cols();
        out.emplace_back(tv);
    }
}

void InputLayer::convert_targets(vector<vector<double>> &in, vector<Tensor> &out) {
    for(auto i : in){
        MatrixXd mat;

        mat = Map<MatrixXd>(i.data(), 1, i.size());

        auto *t = new Tensor(mat);
        t->getShape().r = mat.rows();
        t->getShape().c = mat.cols();
        out.emplace_back(*t);
    }
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void InputLayer::convert(const string& filename, vector<vector<Tensor>> &out, int amount) {

    ifstream file(filename.c_str());

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char *) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char *) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        for (int i = 0; i < amount; ++i) {
            MatrixXd m(1, n_rows * n_cols);
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));
                    m(r*n_cols + c) = static_cast<double>(temp)/255;
                }
            }
            auto *t = new Tensor(m);
            vector<Tensor> tv = {*t};
            t->getShape().r = n_rows;
            t->getShape().c = n_cols;
            out.emplace_back(tv);
        }
    }
}

void InputLayer::convert_targets(const string& filename, vector<Tensor> &out, int amount, int classes) {

    ifstream file(filename.c_str());

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        if(amount != 0)
            number_of_images = amount;
        for (int i = 0; i < number_of_images; ++i) {
            MatrixXd m = MatrixXd::Zero(1, classes);
            unsigned char temp = 0;
            file.read((char *) &temp, sizeof(temp));
            for(int j = 0; j < m.cols(); j++){
                if(temp == j){
                    m(j) = 1;
                }
            }
            auto *t = new Tensor(m);
            t->getShape().r = 1;
            t->getShape().c = classes;
            out.emplace_back(*t);
        }
    }
}


