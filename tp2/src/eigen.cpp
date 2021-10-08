#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


Vector vectorSobreNumero(Vector c, double k)
{
    Vector d(c.size());
    for (unsigned i = 0; i < d.size(); ++i)
    {
        d(i) = c(i) / k;
        
    }
    return d;
}

pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps) {
    Vector b = Vector::Random(X.cols(), 1);
    double eigenvalue;
    b = vectorSobreNumero(b, b.norm());
    for (unsigned i = 0; i < num_iter; i++) {
        Vector c = X * b;
        double c_n = c.norm();
        if ((b - vectorSobreNumero(c, c_n)).norm() < eps) {
            break;
        }
        b = vectorSobreNumero(c, c_n);
    }
    double a = b.transpose() * (X * b);
    double z = b.transpose() * b;

    eigenvalue = a / z;

    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);    

    for (unsigned i = 0; i < num; ++i)
    {
        pair<double, Vector> c = power_iteration(A, num_iter, epsilon);
        eigvalues(i) = c.first;
        for(unsigned j=0; j<eigvectors.rows(); j++)
        {
            eigvectors(j,i) = c.second(j);
        }
        A = A - c.first * c.second * c.second.transpose();
    }
    return make_pair(eigvalues, eigvectors);
}
