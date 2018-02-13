
#include "Factory.h"
#include "NeuralNetwork.h"
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>
#include "Functions.h"
using namespace std;

void test (vector<float> input, NeuralNetwork n){

    n.setInput(input);
    n.wave();
    vector<float> output = n.getOutput();

     for (int i=0;i<output.size();i++){
        std::cout<< output[i]<<" " <<std::endl;
     }


}
struct  A{
int v;
A(){};
A(int a):v(a){}

bool operator==(A& a)const {
    if(this->v * a.v >0)return true;
    return false;

}
};




int main() {

    srand(time(0));
    NeuralNetwork n = Factory::basic(2,2,2);

//    n.threshold->exits.front().str=.5;
    for(auto& x: n.threshold->exits){
      // std::cout<< x.str <<std::endl;
    //   x.str=-50;
    }

    n.neurons[0]->exits.front().str=-50.7;
    for(auto x: n.neurons[0]->exits){
        std::cout<< x.str <<std::endl;
    }


   // n.neurons[0]->exits.front().str=1;
    std::cout<< "" <<std::endl;
    vector<float> input = {0, 0};
    test(input ,n);
    std::cout<< "" <<std::endl;
    input = {0, 1};
    test(input ,n);
        std::cout<< "" <<std::endl;
     input = {1, 0};
    test(input ,n);
        std::cout<< "" <<std::endl;
     input = {1, 1};
    test(input ,n);

   //while(true);
    return 0;

}
