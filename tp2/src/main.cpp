//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"

int main(int argc, char** argv){

    Matrix x {
            {1.0, 10.0},
            {2.0, 9.0},
            {3.0,8.0}}
    ;
    Vector y {
            {1},
            {1},
            }
    ;
    Matrix x_test {
            {0.0, 2.0},
            {15, 0},
            {0,4}}
    ;
    auto fila = x_test.row(0);
    std::cout <<fila.norm();
    auto result = fila.replicate(3,1);
    std::cout << "Here is the matrix x:\n" << result << std::endl;
    x = x-result;
    std::cout << "Here is the matrix x:\n" << x << std::endl;
    auto b = result.rows();
    auto a = result.cols();
    auto mona = x-result;


    return 0;
}
