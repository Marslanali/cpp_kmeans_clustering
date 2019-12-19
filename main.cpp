#include <iostream>
#include "matplotlibcpp.h"
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <vector>

using namespace mlpack::kmeans;
using namespace arma;
namespace plt = matplotlibcpp;


int main()

{

    mat X1 = randu(500,2)*0.75+ones(500,2);
    mat X2 = randu(500,2)*0.5-ones(500,2);

    mat X = join_cols(X1,X2);

   // std::cout<<"\n "<<X<<std::endl;
    //std::cout<<X.n_rows<<std::endl;
   // std::cout<<X.n_cols<<std::endl;

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

    mat  data = X;
    uint nbStates = 2;


    const float diag_reg_fact = 1e-4f;
    // vec timing_sep = linspace<vec>(data(0, 0), data(0, data.n_cols - 1), nbStates + 1);

    std::vector <mat> Mu;
    std::vector <mat> Sigma;
    Mu.clear();
    Sigma.clear();
    // Priors.clear();



    // The dataset we are clustering.
    // mat data;
// The number of clusters we are getting.
    size_t clusters = nbStates;
// The assignments will be stored in this vector.
    Row <size_t> assignments;
// The centroids will be stored in this matrix.
    Mat <double> centroids;
// Initialize with the default arguments.
    KMeans<> k;
    k.Cluster(data, clusters, assignments, centroids);

    //std::cout<<"Assignments : "<<assignments<<std::endl;
    // std::cout<<"Centroids: "<<centroids<<std::endl;

    uvec idtmp;
   // Col <double> priors (nbStates);

    for (int i = 0; i < nbStates; ++i)
        Mu.push_back(centroids(span::all,i));

    //std::cout<<"Means:\n"<<centroids<<std::endl;

    for (int i = 0; i < nbStates; ++i)
    {
        idtmp = find(assignments==i);
        //std::cout<<"\n IDTMP: \n "<<idtmp.n_elem<<std::endl;
       // priors(i) = idtmp.n_elem;
        // std::cout<<"\n KMEANS Priors: \n "<< priors(i)<<std::endl;
        // std::cout<<"\n Data for specific Index: \n "<<data.cols(idtmp).t()<<std::endl;

        //mat sigma = cov(data.cols(idtmp).t());
        //std::cout<<"Covariances Matrices:\n"<<sigma<<std::endl;
        // Optional regularization term to avoid numerical instability
        //sigma = sigma + eye(3, 3) * diag_reg_fact;

        //Sigma.push_back(sigma);
    }

    //Priors = priors / sum(priors);
    // std::cout<<"Overall Priors:\n"<<priors<<std::endl;
    //std::cout<<"Priors Sum: "<< sum(priors)<<std::endl;



    /*
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
*/
    return 0;
}