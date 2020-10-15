# CPP Kmeans Clustering

This repositroy is C++ implementation of kmeans clustering algorithm.

## Files

```
.
├── src
├── include
├── test
├── CMakeList.txt
└── README.md
```
## Dependencies

* [mlpack](https://www.mlpack.org/)   Version 3.4.1

* [Armadillo](http://arma.sourceforge.net/download.html)

* [Matplotlib-CPP](https://github.com/lava/matplotlib-cpp)

## Demo

To build run the following command in terminal:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make 
```

```bash
$ ./kmean --mean 1000 --sigma 500 --distance 200
```

<p align="center">
    <img src = "./figures/particle_filter_demo.gif" width="60%">
</p>

## Miscellaneous

* [mlpack Installation](https://www.mlpack.org/)

* [Armadillo Installation](http://arma.sourceforge.net/download.html)

* [matplotlib-cpp Installation](https://github.com/lava/matplotlib-cpp)

## References

[Kmean Clustering](https://www.researchgate.net/publication/308020680_The_k-means_clustering_technique_General_considerations_and_implementation_in_Mathematica/link/584dd9be08aeb989252647ac/download)

## To-Do List

- :ballot_box_with_check: Refactor src/include
- :negative_squared_cross_mark: Add CMake Cross Platform Support
- :negative_squared_cross_mark: Add Docker Image Support 


