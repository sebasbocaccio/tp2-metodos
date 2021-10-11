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

void KNNClassifier::fit(Matrix X, Matrix y) {
    this->x_train=X;
    this->y_train=y;
}

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
        return p1.second < p2.second;
    });
}

Vector KNNClassifier::predict(Matrix X) {
    //std::vector<tuple<Eigen::VectorXd, int>> imagenes = this->retrieve_matrix_from_file("imagenes.txt");
    auto prediccion_categoria = Vector(X.rows());
    for (unsigned k = 0; k < X.rows(); ++k) {
        vector<tuple<double, int>> vecinos_ordenados = this->neighbours_sorted_by_distance_2(X,k);
        uint categoria_imagen = this->majority_category(vecinos_ordenados,this->_n_neighbors);
        prediccion_categoria(k) = categoria_imagen;
    }
    return prediccion_categoria;
}

std::vector<tuple<Eigen::VectorXd, int>> KNNClassifier::retrieve_matrix_from_file(string file) {
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
        imagenes.push_back(make_tuple(image_pixels, category));
    }
    ifs.close();
    return imagenes;
}
vector<tuple<double, int>>
KNNClassifier::neighbours_sorted_by_distance_2(Matrix &X,int indice_imagen) {

    auto imagen = X.row(indice_imagen);
    auto imagen_to_matrix = imagen.replicate(this->x_train.rows(),1);
    auto diferencia_entre_imagenes = this->x_train - imagen_to_matrix;

    vector<tuple<double, int>> images_norm;
    for (unsigned i = 0; i < this->x_train.rows(); ++i) {
        double norma = diferencia_entre_imagenes.row(i).norm();
        int categoria = this->y_train(i, 0);
        images_norm.push_back(make_tuple(norma, categoria));
    }
    sort(images_norm.begin(), images_norm.end());
    return images_norm;


}
vector<tuple<double, int>>
KNNClassifier::neighbours_sorted_by_distance(Matrix &X, std::vector<tuple<Eigen::VectorXd, int>> &imagenes,
                                             int indice_imagen) {
    vector<tuple<double, int>> images_norm;
    std::vector<tuple<Eigen::VectorXd, int>> copy_imagenes = imagenes;
    for (int i = 0; i < imagenes.size(); i++) {
        auto image_pixels = Eigen::VectorXd(X.cols());
        for (unsigned j = 0; j < X.cols(); ++j) {
            double numero = get<0>(copy_imagenes[i])[j] - X(indice_imagen, j);
            image_pixels[j] = numero;
        }
        double norma = image_pixels.squaredNorm();
        images_norm.push_back(make_tuple(norma, get<1>(copy_imagenes[i])));
    }
    // Sorteo por errores y devuelvo los primeros k.
    sort(images_norm.begin(), images_norm.end());
    return images_norm;
}


int KNNClassifier::majority_category(vector<tuple<double, int>> vecinos, uint cant_vecinos) {
    map<int, int> occurrences;
    occurrences[0] = 0;
    occurrences[1] = 0;
    occurrences[2] = 0;
    occurrences[3] = 0;
    occurrences[4] = 0;
    occurrences[5] = 0;
    occurrences[6] = 0;
    occurrences[7] = 0;
    occurrences[8] = 0;
    occurrences[9] = 0;

    for (int i = 0; i < cant_vecinos; i++) {
        occurrences[get<1>(vecinos[i])] = occurrences[get<1>(vecinos[i])] + 1;
    }
    auto max = get_max(occurrences);
    return max.first;
}