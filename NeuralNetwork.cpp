#include "NeuralNetwork.h"

#include "Neuron.h"

#include <unordered_map>

#include "Functions.h"
typedef std::unordered_map<Neuronptr, bool>VisitList ;
NeuralNetwork::NeuralNetwork()
{

}
NeuralNetwork:: NeuralNetwork ( std::vector<Neuronptr> input, std::vector<Neuronptr> neurons, std::vector<Neuronptr> output )
	: input ( input ), neurons ( neurons ), output ( output )
{
	threshold = Neuronptr();
	threshold->output = 1;
	for ( auto x : neurons ) {
		threshold->createConnection ( x );
	}
	for ( auto x : output ) {
		threshold->createConnection ( x );
	}


}
NeuralNetwork::~NeuralNetwork()
{

}
void NeuralNetwork::addNeuron ( Neuronptr n )
{
	neurons.push_back ( n );
}
void NeuralNetwork::addInput ( Neuronptr n )
{
	input.push_back ( n );
}
void NeuralNetwork::addOutput ( Neuronptr n )
{
	output.push_back ( n );
}
void NeuralNetwork::setInput ( std::vector<float> v )
{
	int count = 0;
	for ( auto x : input ) {
		x->output = v[count];
		count++;
	}
}
std::vector<float> NeuralNetwork::getOutput()
{
	std::vector<float> out;
	out.resize ( output.size() );
	int count = 0;

	for ( auto x : output ) {
		out[count] = Functions::getFunctions()->result(x->sum);
		count++;
	}
	return out;


}

void NeuralNetwork::wave()
{
	VisitList current, next;

	for ( auto x : neurons ) {
		x->reset();
	}
	for ( auto x : output ) {
		x->reset();
	}
	threshold->send();

	current.clear();
	for ( auto x : input ) {
		current[x] = 1;
	}

	next.clear();
	while ( current.size() > 0 ) {
		for ( auto par : current ) {
            Neuronptr n = par.first;
            n->updateOutput();
			n->send();
			for ( auto x : n->exits ) {
				next[x.destiny]=1;
			}
        }
        current = next;
        next.clear();
	}


}
