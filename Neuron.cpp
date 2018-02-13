#include "Neuron.h"
#include "Functions.h"
#include <stdlib.h>
#include <iostream>
Neuron::Neuron()
{

}

Neuron::~Neuron()
{
	//dtor
}
void Neuron::reset()
{
	sum = 0;
}
void Neuron::updateOutput(){
     output = Functions::getFunctions()->result ( sum );
}

void Neuron::send()
{
	for ( Connection connection : exits ) {
		Neuronptr neuron = connection.destiny;
        float value = output*connection.str;
		neuron->sum += value ;

		value = 0;
    }
}
void Neuron::createConnection(Neuronptr n){
    Connection novo;
    novo.destiny = n;
    novo.str = (((rand()%10000)/10000.0)*3.9)-1.95;
    exits.push_back(novo);
}
