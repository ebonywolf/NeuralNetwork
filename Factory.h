#ifndef FACTORY_H
#define FACTORY_H
#include "NeuralNetwork.h"
#include "Neuron.h"
#include <string>
#include <vector>
class Factory
{
    public:
       static NeuralNetwork basic(int inputN,int middle , int outputN){

            NeuralNetwork novo;


            std::vector<Neuronptr> input;
            std::vector<Neuronptr> neurons;
            std::vector<Neuronptr> output;

            Neuronptr threshold;

             for (int i=0;i<outputN;i++){
                output.push_back( Neuronptr());
             }
            for (int i=0;i<inputN;i++){
                input.push_back( Neuronptr());
            }

            for (int i=0;i<middle;i++){
                neurons.push_back( Neuronptr());
            }
            for(auto x: input){
                   for(auto y: neurons){
                       x->createConnection(y);
                   }
            }
            for(auto x: neurons){
                   for(auto y: output){
                       x->createConnection(y);
                   }
            }
            novo = NeuralNetwork(input, neurons, output);



            return novo;
        }



        virtual ~Factory() {}
    protected:
    private:
};

#endif // FACTORY_H
