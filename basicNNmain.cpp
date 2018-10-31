#include <armadillo>
#include <bits/stdc++.h>
#include "BasicNN.h"
#include "Output.h"
using namespace wag;
using namespace std;




 inline double activation(double x){
        return std::log(1.0 + std::exp(x) );
    }

double fdex(double x){
  return x*x;
   
    
}

    
int main(int argc, const char **argv) {
    
    srand(time(0));
    
    //while(true);
   NNTrainer nntrainer;
   nntrainer.trainBasicFunction(fdex);
   
    

}
