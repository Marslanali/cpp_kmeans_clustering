#include <iostream>
#include "matplotlibcpp.h"
#include "armadillo"

using namespace arma;
namespace plt = matplotlibcpp;

int main()

{
    // kmeans clustering testing

    mat X1 = randu(100,2)*0.75+ones(100,2);
    mat X2 = randu(100,2)*0.5-ones(100,2);

    mat X = join_cols(X1,X2);

    std::cout<<"\n "<<X<<std::endl;
    std::cout<<X.n_rows<<std::endl;
    std::cout<<X.n_cols<<std::endl;

    std::vector<double> random_data_X;
    std::vector<double> random_data_Y;

    for (int i=0; i<X.n_rows; ++i)
    {
        random_data_X.push_back(X(i,0));
        random_data_Y.push_back(X(i,1));
    }

    plt::figure();
    plt::plot(random_data_X,random_data_Y,".");
    plt::title("Random Data");
    plt::show();

    return 0;
}