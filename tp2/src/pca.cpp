#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components)
{
  _n_components = n_components;
}

void PCA::fit(Matrix &X)
{
  Vector mean(X.cols());

  for (int i = 0; i < X.rows(); ++i)
  {
      for (int j = 0; j < X.row(0).size(); ++j)
      {
          mean(j) = mean(j) + X.row(i)(j);
      }
  }

  mean = mean / X.rows();

  Matrix means(X.rows(),X.cols());
  for(int i=0; i < X.rows(); i++)
  {
      for (int j = 0; j < mean.size(); ++j)
      {
          means.row(i)(j) = mean (j);
      }
  }
  Matrix Y = X - means;

  Matrix CovMatrix = (Y.transpose() * Y) / (Y.rows()-1); 

  X = get_first_eigenvalues(CovMatrix, X.rows(), 1000, 1e-16).second;

}


MatrixXd PCA::transform(Matrix X)
{
  Matrix Y(X);
  this->fit(X);
  return X*Y;

  //throw std::runtime_error("Sin implementar");
}
