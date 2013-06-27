//
//  main.cpp
//  pkmDPGMM
//
//  Created by Mr. Magoo on 5/21/13.
//
//

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "libcluster.h"
#include "util.hpp"

using Eigen::MatrixXd;
using namespace std;
using namespace libcluster;
using namespace distributions;

int main()
{
    int num = 300;
    MatrixXd data(num*2,3);
    MatrixXd data1(num,3);
    REDSVD::Util::sampleGaussianMat(data1);
    MatrixXd data2(num,3);
    REDSVD::Util::sampleGaussianMat(data2);
    data2 += MatrixXd::Constant(num,3,9.0);

    data.block(0,0,num,3) = data1;
    data.block(num,0,num,3) = data2;
    
    MatrixXd qZ;
    distributions::StickBreak weights;
    std::vector<distributions::GaussWish> clusters;
    
    clock_t start = clock();
    learnVDP(data, qZ, weights, clusters);
    double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
    cout << "GMC Elapsed time = " << stop << " sec." << endl;
    
    cout << endl << "Cluster Weights:" << endl;
    cout << weights.Elogweight().exp().transpose() << endl;
    
    cout << endl << "Cluster means:" << endl;
    for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
        cout << k->getmean() << endl;
    
    
    cout << endl << "Cluster covariances:" << endl;
    for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
        cout << k->getcov() << endl << endl;
    
    return 0;
}