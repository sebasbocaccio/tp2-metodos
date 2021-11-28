#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <fstream>

using namespace std;

#include <iostream>
#include <fstream>

KNNClassifier::KNNClassifier(unsigned int n_neighbors) {
    this->_n_neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix &X, Matrix &y, string matrix_saving_name = "knn_matrix.txt") {
    this->image_name = matrix_saving_name;
    std::ofstream outfile(matrix_saving_name);
    outfile << X.rows() << ' ' << X.cols() << std::endl;
    for (unsigned k = 0; k < X.rows(); ++k) {
        auto image_pixels = Eigen::VectorXd(X.cols());
        for (unsigned j = 0; j < X.cols(); ++j) {
            outfile << X(k, j) << ' ';
        }
        outfile << y(k, 0) << std::endl;
    }
    outfile.close();
    this->imagenes.clear();
}

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
        return p1.second < p2.second;
    });
}

Vector KNNClassifier::predict(Matrix &X) {
    if(this->imagenes.empty()) {
        this->retrieve_matrix_from_file(this->image_name);
    }

    auto prediccion_categoria = Vector(X.rows());
    for (unsigned k = 0; k < X.rows(); ++k) {
        vector<tuple<double, int>>* vecinos_ordenados = this->neighbours_sorted_by_distance(X, this->imagenes,k);
        uint categoria_imagen = this->majority_category(*vecinos_ordenados,this->_n_neighbors);
        prediccion_categoria(k) = categoria_imagen;
        delete vecinos_ordenados;
    }
    return prediccion_categoria;
}

void KNNClassifier::retrieve_matrix_from_file(string file) {
    std::ifstream ifs(file);
    std::vector<tuple<Eigen::VectorXd, int>> imagenes;
    double number;
    string line;
    std::getline(ifs, line);
    stringstream iss(line);
    uint cols;
    uint rows;
    iss >> rows;
    iss >> cols;

    for (unsigned i = 0; i < rows; ++i) {
        std::getline(ifs, line);
        stringstream iss(line);
        auto image_pixels = Eigen::VectorXd(cols);
        for (unsigned k = 0; k < cols; ++k) {
            iss >> number;
            image_pixels[k] = number;
        }
        int category;
        iss >> category;
        this->imagenes.push_back(make_tuple(image_pixels, category));
    }
    ifs.close();
}

vector<tuple<double, int>>*
KNNClassifier::neighbours_sorted_by_distance(Matrix &X, std::vector<tuple<Eigen::VectorXd, int>> &imagenes,
                                             int indice_imagen) {
    vector<tuple<double, int>>* images_norm = new vector<tuple<double, int>>;
    for (int i = 0; i < imagenes.size(); i++) {
        auto image_pixels = Eigen::VectorXd(X.cols());
        for (unsigned j = 0; j < X.cols(); ++j) {
            double numero = get<0>(imagenes[i])[j] - X(indice_imagen, j);
            image_pixels[j] = numero;
        }
        double norma = image_pixels.squaredNorm();
        images_norm->push_back(make_tuple(norma, get<1>(imagenes[i])));
    }

    sort(images_norm->begin(), images_norm->end());
    return images_norm;
}


int KNNClassifier::majority_category(vector<tuple<double, int>> &vecinos, uint cant_vecinos) {
    std::vector<int> occurrences(10, 0);
    for (int i = 0; i < cant_vecinos; i++) {
        occurrences[get<1>(vecinos[i])] = occurrences[get<1>(vecinos[i])] + 1;
    }
    int maxElementIndex = 0;
    int maxElementCount = 0;
    for (int i = 0; i < occurrences.size(); i++) {
        if(occurrences[i] > maxElementCount) {
            maxElementIndex = i;
        }
    }
    return maxElementIndex;
}
void  KNNClassifier::change_k(unsigned int n_neighbors){
    this->_n_neighbors = n_neighbors;
}