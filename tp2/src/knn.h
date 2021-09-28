#pragma once

#include "types.h"
using namespace std;


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
private:
    unsigned int _n_neighbors;
    int nearest_element_index( vector<tuple<double ,int > > arr, int l, int r, double x);
};
