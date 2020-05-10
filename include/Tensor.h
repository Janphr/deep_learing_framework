//
// Created by janphr on 29.04.20.
//

#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>



using namespace std;

struct Shape {
    Shape(int r, int c, int d);

    int r,  //rows
    c,  //cols
    d;  //depth
};

class Tensor {

public:
    Tensor();

    Tensor(Eigen::MatrixXd &elements);
    Tensor(Eigen::MatrixXd &elements, Eigen::MatrixXd &deltas);

    Eigen::MatrixXd &getElements();

    void setElements(const Eigen::MatrixXd &elements);

    Eigen::MatrixXd &getDeltas();

    void setDeltas(const Eigen::MatrixXd &deltas);

    Shape &getShape();

private:
    shared_ptr<Eigen::MatrixXd> elements;
    shared_ptr<Eigen::MatrixXd> deltas;

    Shape shape = {0,0,0};
};

