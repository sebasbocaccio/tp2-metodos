//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"
#include "knn.h"
int main(int argc, char** argv){

  PCA pca  = PCA(3);


  Matrix m (2,2);
    m(0,0) = 3;
    m(1,0) = 1;
    m(0,1) = 1;
    m(1,1) = -2;


  pair<double,Vector> f = power_iteration(m,1000,1e-6);
  cout << f.first << " " << f.second(0) << f.second(1) << '\n';
  //get_first_eigenvalues
  pair<Vector,Matrix> eigvv = get_first_eigenvalues(m,2,5000,1e-16);
  cout << eigvv.second(0,0) << " " << eigvv.second(0,1)<< " " << eigvv.second(1,0) << " " << eigvv.second(1,1) << '\n';
  cout << eigvv.first(0) << " " << eigvv.first(1) << '\n';

  Matrix A(m);
  cout << m(0,0) << " " << m(0,1) << " " << m(1,0) << " " << m(1,1) << '\n';
  pca.fit(m);
  cout << m(0,0) << " " << m(0,1) << " " << m(1,0) << " " << m(1,1) << '\n';
  Eigen::MatrixXd B = pca.transform(A);
  cout << B(0,0) << " " << B(0,1) << " " << B(1,0) << " " << B(1,1);


  return 0;
}
