#include <armadillo>
#include <bits/stdc++.h>
#include "BasicNN.h"
#include "Output.h"
using namespace wag;
using namespace std;



template <class T>
ostream& operator<<(ostream& os, const vector<T>& t ){
    for (int i =0; i < t.size() ; i++){
       std::cout << i<<":"<<t[i] << std::endl;
    }
    return os;
   
}

 inline double activation(double x){
        return std::log(1.0 + std::exp(x) );
    }

double fdex(double x){
    return cos(x)*15;
}

double test(std::function< Vetor (Vetor)> v){
    Vetor inputs={3};
    
    Vetor output=v(inputs);
    std::cout << output << std::endl;
}
    
int main(int argc, const char **argv) {
    
    vector<double> inputs={-5};
    ifstream in("bugnn",ios::binary);
    BasicNN nn;
    if(in.is_open()){
       
       in>>nn;
       
    }else{
        std::cout << "foo" << std::endl;
        exit(0);
    }
    std::cout << nn.layers << std::endl;

    auto output= nn(inputs);
    std::cout << output << std::endl;
    return 0;
   
   
    
    srand(time(0));
    
    

    //while(true);
   NNTrainer nntrainer;
   nntrainer.trainBasicFunction(fdex);
   
    

}
