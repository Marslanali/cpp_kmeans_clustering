//
// Created by arslan on 2/12/19.
//

/**
 * @author: Arslan Ali
 *
 * C++ Implementation of K means clustering algorithm using C++ MLPACK library
 *
 */

#include <armadillo>
#include <iostream>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <vector>
#include "matplotlibcpp.h"

using namespace mlpack::kmeans;
namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
  // Generate random vector using random number generators
  arma::Mat<double> x1 = arma::randu(500, 2) * 0.75 + arma::ones(500, 2);
  arma::Mat<double> x2 = arma::randu(500, 2) * 0.5 - arma::ones(500, 2);

  arma::Mat<double> x = join_cols(x1, x2);
  x = x.t();

  std::cout << "Data:\n" << x << std::endl;
  std::cout << x.n_rows << std::endl;
  std::cout << x.n_cols << std::endl;

  std::vector<double> random_data_x;
  std::vector<double> random_data_y;

  for (int i = 0; i < x.n_cols; ++i) {
    random_data_x.push_back(x(0, i));
    random_data_y.push_back(x(1, i));
  }

  plt::figure();
  plt::plot(random_data_x, random_data_y, ".");
  plt::title("Random Data");
  plt::save("../plots/random_data.jpg");
  plt::show();

  arma::Mat<double> data = x;
  uint nb_states = 2;

  const float diag_reg_fact = 1e-4f;

  // vec timing_sep = linspace<vec>(data(0, 0), data(0, data.n_cols - 1), nb_states + 1);

  std::vector<arma::Mat<double>> mu;
  std::vector<arma::Mat<double>> sigma;
  arma::Col<double> priors;

  mu.clear();
  sigma.clear();

  // priors.clear();
  // The dataset we are clustering.
  // arma::Mat<double> data;

  // The number of clusters we are getting.
  size_t clusters = nb_states;

  // The assignments will be stored in this vector.
  arma::Row<size_t> assignments;

  // The centroids will be stored in this matrix.
  arma::Mat<double> centroids;

  // Initialize with the default arguments.
  KMeans<> k;
  k.Cluster(data, clusters, assignments, centroids);

  // std::cout<<"Assignments : "<<assignments<<std::endl;
  // std::cout<<"Centroids: "<<centroids<<std::endl;

  arma::Col<arma::uword> idtmp;
  arma::Col<double> priors_vec(nb_states);

  std::vector<double> data_x1;
  std::vector<double> data_y1;

  std::vector<double> data_x2;
  std::vector<double> data_y2;

  std::vector<double> c1;
  std::vector<double> c2;

  for (int i = 0; i < nb_states; ++i) {
    mu.push_back(centroids(arma::span::all, i));
  }

  // std::cout<<"Means:\n"<<centroids<<std::endl;

  idtmp = find(assignments == 0);
  // std::cout<<"SIZE\n"<<idtmp.n_elem<<std::endl;
  for (int i = 0; i < idtmp.n_elem; ++i) {
    // std::cout<<"DATA is \n"<<data(0,idtmp[i])<<std::endl;
    data_x1.push_back(data(0, idtmp[i]));
    data_y1.push_back(data(1, idtmp[i]));
  }
  // std::cout<<"assignments\n"<<assignments<<std::endl;
  // std::cout<<"\nIDTMP:\n "<<idtmp<<std::endl;

  idtmp = find(assignments == 1);

  for (int i = 0; i < idtmp.n_elem; ++i) {
    // std::cout<<"DATA is \n"<<data(0,idtmp[i])<<std::endl;
    data_x2.push_back(data(0, idtmp[i]));
    data_y2.push_back(data(1, idtmp[i]));
  }

  for (int i = 0; i < nb_states; ++i) {
    c1.push_back(centroids(0, i));
    c2.push_back(centroids(1, i));
  }

  const arma::span index(0, 1);

  plt::figure();
  plt::plot(data_x1, data_y1, "r.");

  // plt::title("Random Data");
  // plt::show();

  plt::plot(data_x2, data_y2, "b.");
  plt::plot(c1, c2, "kx");
  plt::title("Kmeans Clustering");
  plt::save("../plots/kmeans_clustering_lot.jpg");
  plt::show();

  std::cout << "Centers\n" << centroids << std::endl;

  /*
  for (int i = 0; i < nb_states; ++i)
  {
      priors_vec(i) = idtmp.n_elem;
      // std::cout<<"\n KMEANS priors: \n "<< priors_vec(i)<<std::endl;
      // std::cout<<"\n Data for specific Index: \n "<<data.cols(idtmp).t()<<std::endl;
      arma::Mat<double> sigma = cov(data.cols(idtmp).t());
      //std::cout<<"Covariances Matrices:\n"<<sigma<<std::endl;
      // Optional regularization term to avoid numerical instability
     //sigma = sigma + eye(2, 2) * diag_reg_fact;
      sigma.push_back(sigma);
  }
  priors = priors_vec / sum(priors_vec);
  // std::cout<<"Overall priors:\n"<<priors_vec<<std::endl;
  //std::cout<<"priors Sum: "<< sum(priors_vec)<<std::endl;
  */

  /*
  // kmeans clustering testing
  arma::Mat<double> x1 = randu(100,2)*0.75+ones(100,2);
  arma::Mat<double> x2 = randu(100,2)*0.5-ones(100,2);
  arma::Mat<double> x = join_cols(x1,x2);
  std::cout<<"\n "<<x<<std::endl;
  std::cout<<x.n_rows<<std::endl;
  std::cout<<x.n_cols<<std::endl;
  std::vector<double> random_data_x;
  std::vector<double> random_data_y;
  for (int i=0; i<x.n_rows; ++i)
  {
      random_data_x.push_back(x(i,0));
      random_data_y.push_back(x(i,1));
  }
  plt::figure();
  plt::plot(random_data_x,random_data_y,".");
  plt::title("Random Data");
  plt::show();
  */

  return 0;
}