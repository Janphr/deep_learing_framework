//
// Created by janphr on 29.04.20.
//

#include "Tensor.h"
using namespace Eigen;

MatrixXd &Tensor::getElements() {
    return *elements;
}

void Tensor::setElements(const MatrixXd &elements) {
    Tensor::elements = make_shared<MatrixXd>(elements);
}

MatrixXd &Tensor::getDeltas() {
    return *deltas;
}

void Tensor::setDeltas(const MatrixXd &deltas) {
    Tensor::deltas = make_shared<MatrixXd>(deltas);
}

Tensor::Tensor(MatrixXd &elements) {
    setElements(elements);
    Tensor::deltas = make_shared<MatrixXd>();
}

Tensor::Tensor(MatrixXd &elements, MatrixXd &deltas) {
    setElements(elements);
    setDeltas(deltas);
}

Tensor::Tensor() {
    Tensor::elements = make_shared<MatrixXd>();
    Tensor::deltas = make_shared<MatrixXd>();
}

Shape &Tensor::getShape() {
    return shape;
}

vector< tuple<int, double> > & Tensor::getPoolingVec() {
return pooling_vec;
}

Shape::Shape(int r, int c, int d) {
    this->r = r;
    this->c = c;
    this->d = d;
}
