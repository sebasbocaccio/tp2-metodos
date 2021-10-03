#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
int main()
{
    std::vector<std::vector<int>> vec;

    std::ifstream file_in("data.txt");
    if (!file_in) {/*error*/}

    std::string line;
    while (std::getline(file_in, line)) // Read next line to `line`, stop if no more lines.
    {
        // Construct so called 'string stream' from `line`, see while loop below for usage.
        std::istringstream ss(line);

        vec.push_back({}); // Add one more empty vector (of vectors) to `vec`.

        int x;
        while (ss >> x) // Read next int from `ss` to `x`, stop if no more ints.
            vec.back().push_back(x); // Add it to the last sub-vector of `vec`.
    }
     for (unsigned k = 0; k < vec.size(); ++k){
       for (unsigned j = 0; j < vec[0].size(); ++j){
            cout << vec[k][j] ;
       }
       cout<<endl;
    }
}