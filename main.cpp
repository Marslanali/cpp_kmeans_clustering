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
    X = X.t();

    std::cout<<"Data:\n"<<X<<std::endl;
    std::cout<<X.n_rows<<std::endl;
    std::cout<<X.n_cols<<std::endl;

    std::vector<double> random_data_X;
    std::vector<double> random_data_Y;

    for (int i=0; i<X.n_cols; ++i)
    {
        random_data_X.push_back(X(0,i));
        random_data_Y.push_back(X(1,i));
    }

    plt::figure();
    plt::plot(random_data_X,random_data_Y,".");
    plt::title("Random Data");
    plt::save("/home/arslan/CLionProjects/cpp_kmeans_clustering/plots/random_data.jpg");
    plt::show();

    mat  data = X;
    uint nbStates = 2;

    const float diag_reg_fact = 1e-4f;
    // vec timing_sep = linspace<vec>(data(0, 0), data(0, data.n_cols - 1), nbStates + 1);

    std::vector <mat> Mu;
    std::vector <mat> Sigma;
    Col <double> Priors;

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
    Col <double> priors (nbStates);

    std::vector<double> data_X1;
    std::vector<double> data_Y1;

    std::vector<double> data_X2;
    std::vector<double> data_Y2;


    std::vector<double> C1;
    std::vector<double> C2;



    for (int i = 0; i < nbStates; ++i)
        Mu.push_back(centroids(span::all,i));

    //std::cout<<"Means:\n"<<centroids<<std::endl;

    idtmp = find(assignments==0);
    //std::cout<<"SIZE\n"<<idtmp.n_elem<<std::endl;
    for (int i = 0; i < idtmp.n_elem ; ++i) {
        //std::cout<<"DATA is \n"<<data(0,idtmp[i])<<std::endl;
        data_X1.push_back(data(0,idtmp[i]));
        data_Y1.push_back(data(1,idtmp[i]));

    }
    //std::cout<<"assignments\n"<<assignments<<std::endl;
    //std::cout<<"\nIDTMP:\n "<<idtmp<<std::endl;


    idtmp = find(assignments==1);

    for (int i = 0; i < idtmp.n_elem ; ++i) {
        //std::cout<<"DATA is \n"<<data(0,idtmp[i])<<std::endl;
        data_X2.push_back(data(0,idtmp[i]));
        data_Y2.push_back(data(1,idtmp[i]));

    }

    for (int i = 0; i < nbStates ; ++i) {
        C1.push_back(centroids(0,i));
        C2.push_back(centroids(1,i));
    }


    const span index(0,1);

    plt::figure();
    plt::plot(data_X1,data_Y1,"r.");
    //plt::title("Random Data");
    //plt::show();

    plt::plot(data_X2,data_Y2,"b.");
    plt::plot(C1,C2,"kx");
    plt::title("Kmeans Clustering");
    plt::save("/home/arslan/CLionProjects/cpp_kmeans_clustering/plots/kmeans_clustering_lot.jpg");
    plt::show();

    std::cout<<"Centers\n"<<centroids<<std::endl;

/*
    for (int i = 0; i < nbStates; ++i)
    {
        priors(i) = idtmp.n_elem;
        // std::cout<<"\n KMEANS Priors: \n "<< priors(i)<<std::endl;
        // std::cout<<"\n Data for specific Index: \n "<<data.cols(idtmp).t()<<std::endl;

        mat sigma = cov(data.cols(idtmp).t());
        //std::cout<<"Covariances Matrices:\n"<<sigma<<std::endl;
        // Optional regularization term to avoid numerical instability
       //sigma = sigma + eye(2, 2) * diag_reg_fact;

        Sigma.push_back(sigma);
    }

    Priors = priors / sum(priors);
    // std::cout<<"Overall Priors:\n"<<priors<<std::endl;
    //std::cout<<"Priors Sum: "<< sum(priors)<<std::endl;
*/
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