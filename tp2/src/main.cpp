//
// Created by pachi on 5/6/19.
//

#include <iostream>
//#include "pca.h"
#include "eigen.h"
#include "knn.h"
int main(int argc, char** argv){

  std::cout << "Hola mundo!" << std::endl;
  KNNClassifier knn = KNNClassifier(3);
  Matrix x {
            {0, 1},
            {0, 10},
            {0,5}}
            ;
    Matrix y {
            {1},
            {2},
            {3}}
    ;
    knn.fit(x,y);
    Matrix x_test {
            {0, 2},
            {15, 0},
            {0,4}}
    ;
    
    knn.predict(x_test);
    return 0;
}
