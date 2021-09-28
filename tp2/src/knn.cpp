#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;
#include <iostream>
#include <fstream>  

KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
    this->_n_neighbors= n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y)
{
    vector<tuple<double,int>> images_norm;
    std::ofstream outfile ("images_squaredNorm_sorted.txt");

    for (unsigned k = 0; k < X.rows(); ++k){
        auto image_pixels = Eigen::VectorXd(X.cols()); 
       for (unsigned j = 0; j < X.cols(); ++j){
            image_pixels[j] = X(k,j);
       }
        images_norm.push_back(make_tuple(image_pixels.squaredNorm(), y(k,0)));
    }
    sort(images_norm.begin(), images_norm.end());
    for (long unsigned int i = 0; i < images_norm.size(); i++) 
        outfile << get<0>(images_norm[i]) << ' ' <<  get<1>(images_norm[i]) << std::endl;

    outfile.close();

}



Vector KNNClassifier::predict(Matrix X)
{

    double norm;
    int digit;
    vector<tuple<double,int>> images_norm;

    std::ifstream infile("images_squaredNorm_sorted.txt");
    while (infile >> norm >> digit)
    {
        images_norm.push_back(make_tuple(norm,digit));
    }
    infile.close();
    this->nearest_element_index(images_norm,0,images_norm.size()-1,10.0);

    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());
    for (unsigned k = 0; k < X.rows(); ++k)
    {
        // Consigo su norma 
        // Encuentro donde esta parado.
        // Encuentro los k mas cercanos
        // Me fijo cual tiene mayoria
        // Pongo en vector 
    }

    return ret;

    
}




int KNNClassifier::nearest_element_index( vector<tuple<double ,int >> arr, int l, int r, double x)
{
    std::ofstream outfile ("m.txt");

    int m = l + (r - l) / 2;
    int counter = 0 ;
    while (l <= r && counter < 100) {
        counter++;

        cout << m <<endl;
        m = l + (r - l) / 2;
        outfile << m << std::endl;

        // Check if x is present at mid
        if (get<0>(arr[m]) == x)
            return m;
  
        // If x greater, ignore left half
        if (get<0>(arr[m]) < x)
            l = m + 1;
  
        // If x is smaller, ignore right half
        else
            r = m - 1;
    }
  
    if(m == 0){
       double  below_diff =  abs(x-get<0>(arr[0]));
       double  middle_diff = abs(x-get<0>(arr[1]));
       outfile.close();
       return (below_diff < middle_diff) ? 0 : 1 ;
       
        }
    else if(m  == arr.size()){
       double  below_diff =  abs(x-get<0>(arr[m-1]));
       double middle_diff = abs(x-get<0>(arr[m-2]));
       outfile.close();
       return  (below_diff < middle_diff) ? m-1  : m-2;
    }
    
    else {
        double below_diff =  abs(x-get<0>(arr[m-1]));
        double middle_diff = abs(x-get<0>(arr[m]));
        double above_diff = abs(x-get<0>(arr[m+1]));
        outfile.close();
        return (below_diff < middle_diff) ? m-1 :  (above_diff < middle_diff) ? m+1 : m  ;
        }
}

