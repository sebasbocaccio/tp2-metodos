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


    vector<tuple<double,int>> images_norm;
    double norm;
    int digit;
    std::ofstream outfile ("resultados.txt");
    std::ifstream infile("images_squaredNorm_sorted.txt");
    while (infile >> norm >> digit)
    {
        images_norm.push_back(make_tuple(norm,digit));
    }
    infile.close();

    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());
    
    for (unsigned k = 0; k < X.rows(); ++k)
    {
        outfile << 'Vector' << k << std::endl;

        // Consigo su norma 
        auto image_pixels = Eigen::VectorXd(X.cols());
            for (unsigned j = 0; j < X.cols(); ++j){
                image_pixels[j] = X(k,j);
                }
        double norma = image_pixels.squaredNorm();

        // Encuentro donde esta parado.
        outfile << '9' << '9'<< '9' << std::endl;
        outfile << norma << std::endl;
        int nearest_index = this->nearest_element_index(images_norm,0,images_norm.size()-1,norma);

        // Encuentro los k mas cercanos
        map<int,int> occurrences;
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

        int lower_index = nearest_index - 1;
        int above_index = nearest_index + 1;
        int n_neighbors_counted = 1 ;
        occurrences[get<1>(images_norm[nearest_index])] = 1 ;
        outfile << get<1>(images_norm[nearest_index]) << std::endl;
        while(n_neighbors_counted < this->_n_neighbors){

            if(lower_index < 0 && above_index > images_norm.size()){break;}
            else if(lower_index < 0){
                occurrences[get<1>(images_norm[above_index])] = occurrences[get<1>(images_norm[above_index])] + 1;
                outfile << get<1>(images_norm[above_index]) << std::endl;

                above_index++;
            }
            else if(above_index > images_norm.size()){
                occurrences[get<1>(images_norm[lower_index])] = occurrences[get<1>(images_norm[lower_index])] + 1;
                outfile << get<1>(images_norm[lower_index]) << std::endl;
                lower_index--;
            }
            else{
                outfile << ":" << lower_index << ' ' <<get<0>(images_norm[lower_index]) << ' ' << abs(get<0>(images_norm[lower_index])-norma) << ' ' << above_index << ' ' <<get<0>(images_norm[above_index]) << ' ' << abs(get<0>(images_norm[above_index])-norma) << std::endl;
                if(abs(get<0>(images_norm[lower_index])-norma) < abs(get<0>(images_norm[above_index])-norma)){
                    occurrences[get<1>(images_norm[lower_index])] = occurrences[get<1>(images_norm[lower_index])] + 1;
                    outfile << get<1>(images_norm[lower_index]) << std::endl;
                    lower_index--;
                }
                else{
                    occurrences[get<1>(images_norm[above_index])] = occurrences[get<1>(images_norm[above_index])] + 1;
                    outfile << get<1>(images_norm[above_index]) << std::endl;
                    above_index++;

                }
            }
            n_neighbors_counted++;
        }
        outfile << n_neighbors_counted << std::endl;

        // Me fijo cual tiene mayoria
        int max_index = 0;
        int max_occurrences = 0 ;
        for(int i=0;i<=9;i++){
            if(occurrences[i] > max_occurrences){
                max_index = i;
                max_occurrences=occurrences[i];
            }
        }
        for(int i=0;i<=9;i++){
            outfile << i << ':'<< occurrences[i] << std::endl;
        }
        // Pongo en vector
        ret(k) = max_index;
    }
    outfile.close();
    return ret;


}




int KNNClassifier::nearest_element_index( vector<tuple<double ,int >> arr, int l, int r, double x)
{
    std::ofstream outfile ("m.txt");

    int m = l + (r - l) / 2;
    int counter = 0 ;
    while (l <= r && counter < 100) {
        counter++;
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

