#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "NeuralNetwork.h"
#include <vector>
#include <list>
#include "Neuron.h"
#include <cmath>
#include <iostream>
#define M_E 2.718281828459045

class Functions
{
    public:
        float result(float a ){
           // std::cout<<"value:"<< a <<std::endl;
            return (1/(1+pow(M_E, -a)));
        }
         void basicTrain(NeuralNetwork &n, std::list<std::vector<float>>){
            std::list<Neuron::Connection*> con;
            for(auto x: n.neurons){
                for(auto y: x->exits ){
                    con.push_back(&y);
                }
            }
            for(auto x: n.input){
                for(auto y: x->exits ){
                    con.push_back(&y);
                }
            }
            for(auto x: n.threshold->exits ){
                    con.push_back(&x);
            }
        }

        static Functions* getFunctions(Functions* f=0){
            static Functions* singleton = 0;
            if( f!=0 ){
                singleton = f;
            }
            return singleton;
        }

        virtual ~Functions();

    protected:
    private:
};

#endif // FUNCTIONS_H
