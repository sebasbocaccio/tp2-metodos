#pragma once

#include "types.h"
#include <fstream>
using namespace std;


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);
    std::vector<tuple<Eigen::VectorXd, int>> retrieve_matrix_from_file(string file);

    Vector predict(Matrix X);
private:
    unsigned int _n_neighbors;




};
