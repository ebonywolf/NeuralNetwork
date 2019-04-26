#include <armadillo>
#include <bits/stdc++.h>
#include "BasicNN.h"
#include "Output.h"
using namespace wag;
using namespace std;




 inline double activation(double x){
        return std::log(1.0 + std::exp(x) );
    }


struct Func {
    
     void setT(double t){
         T = t;
     }
    double T=0;
    double operator()(double x){
        return  ((x+5)*(x+3)*(x-3)*(x-5));
    }
};
double fdex(double x){
    static double T =0;
    
    //if(x==-10) T=0;
    
    return ((x+5)*(x+3)*(x-3)*(x-5));

    
}

    
int main(int argc, const char **argv) {
    arma::Mat<double> mat(2,3);
    
    
    mat(1,0) =3;
    std::cout << mat << std::endl;
    
    return 1;
    srand(time(0));
    
    //while(true);
Func f;
NNTrainer nntrainer;
nntrainer.trainBasicFunction(f);

    

}
