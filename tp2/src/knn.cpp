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
    vector<tuple<double, int>> images_norm;
    std::ofstream outfile("manolo1.txt");
    outfile << X.rows() << ' ' <<  X.cols() << std::endl;
    for (unsigned k = 0; k < X.rows(); ++k) {
        auto image_pixels = Eigen::VectorXd(X.cols());
        for (unsigned j = 0; j < X.cols(); ++j) {
            image_pixels[j] = X(k, j);
            outfile << image_pixels[j] << ' ';
        }
        outfile << y(k, 0) << std::endl;
        images_norm.push_back(make_tuple(image_pixels.squaredNorm(), y(k,0)));
    }

    outfile.close();

}

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
        return p1.second < p2.second;
    });
}

Vector KNNClassifier::predict(Matrix X) {
    std::ifstream ifs("manolo1.txt");
    std::vector<tuple<Eigen::VectorXd, int>> imagenes;
    int number;
    string line;// Get number of images
    std::getline(ifs, line);
    stringstream iss(line);
    uint cols;
    uint rows;

    iss >> rows;
    iss >> cols;
    // Get images
    for (unsigned i = 0; i < rows; ++i) {
        std::getline(ifs, line);
        stringstream iss(line);
        auto image_pixels = Eigen::VectorXd(X.cols());
        for (unsigned k = 0; k < cols; ++k) {
            iss >> number;
            image_pixels[k] = number;
        }
        int category;
        iss >> category;
        imagenes.push_back(make_tuple(image_pixels, category));
    }

    ifs.close();
    vector<tuple<double, int>> images_norm;
    auto ret = Vector(X.rows());
    for (unsigned k = 0; k < X.rows(); ++k) {

        std::vector<tuple<Eigen::VectorXd, int>> copy_imagenes = imagenes;
        // Hago la diferencia
        for (int i = 0; i < imagenes.size(); i++) {
            auto image_pixels = Eigen::VectorXd(X.cols());
            for (unsigned j = 0; j < X.cols(); ++j) {
                double numero = get<0>(copy_imagenes[i])[j] - X(k, j);
                image_pixels[j] = numero;
            }
            double norma = image_pixels.squaredNorm();
            images_norm.push_back(make_tuple(norma, get<1>(copy_imagenes[i])));
        }
        // Sorteo por errores y devuelvo los primeros k. 
        sort(images_norm.begin(), images_norm.end());

        // Encuentro los k mas cercanos
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

        for (int i = 0; i < this->_n_neighbors; i++) {
            occurrences[get<1>(images_norm[i])] = occurrences[get<1>(images_norm[i])] + 1;
        }
        images_norm.clear();
        auto max = get_max(occurrences);
        ret(k) = max.first;
    }
    return ret;
}

