#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <list>
#include <vector>
#include "Neuron.h"
#include <tuple>
class Neuron;

class NeuralNetwork
{
	public:
		NeuralNetwork();
		NeuralNetwork ( std::vector<Neuronptr> input, std::vector<Neuronptr> neurons, std::vector<Neuronptr> output );


		virtual ~NeuralNetwork();


		void addNeuron ( Neuronptr );
		void addInput ( Neuronptr );
		void addOutput ( Neuronptr );
		void setInputNormalizationValue(float min, float max);
        void setOutputNormalizationValue(float min, float max);


		std::vector<Neuronptr> neurons;
		std::vector<Neuronptr> input;
		std::vector<Neuronptr> output;
		Neuronptr threshold;

		void setInput ( std::vector<float> );
		std::vector<float> getOutput();

		void wave();

	protected:
	    std::tuple<float,float,bool> inputNormal;
        std::tuple<float,float,bool> outputNormal;

	private:
};
#endif // NEURALNETWORK_H
