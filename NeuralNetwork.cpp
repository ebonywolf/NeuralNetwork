#include "NeuralNetwork.h"

#include "Neuron.h"

#include <unordered_map>

#include "Functions.h"
using namespace std;
typedef std::unordered_map<Neuronptr, bool>VisitList ;
NeuralNetwork::NeuralNetwork()
{

}
NeuralNetwork:: NeuralNetwork ( std::vector<Neuronptr> input, std::vector<Neuronptr> neurons, std::vector<Neuronptr> output )
	: input ( input ), neurons ( neurons ), output ( output )
{
	threshold = Neuronptr ( new Neuron() );
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
void NeuralNetwork::setInputNormalizationValue ( float min, float max )
{
	std::get<0> ( inputNormal ) = min;
	std::get<1> ( inputNormal ) = max;
	std::get<2> ( inputNormal ) = true;
}

void NeuralNetwork::setOutputNormalizationValue ( float min, float max )
{
	std::get<0> ( outputNormal ) = min;
	std::get<1> ( outputNormal ) = max;
	std::get<2> ( outputNormal ) = true;
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
	if ( get<2> ( inputNormal ) ) {
        double var = get<1>(inputNormal) - get<0>(inputNormal);
		for ( auto x : input ) {
			x->output = (v[count]-get<0>(inputNormal))/var;
			count++;
		}

	} else {
		for ( auto x : input ) {
			x->output = v[count];
			count++;
		}
	}

}
std::vector<float> NeuralNetwork::getOutput()
{
	std::vector<float> out;
	out.resize ( output.size() );
	int count = 0;
    if ( get<2> ( outputNormal ) ) {
       for ( auto x : output ) {
		out[count] = Functions::getFunctions()->result ( x->sum );
		count++;
        }
    }else{
         double var = get<1>(outputNormal) - get<0>(outputNormal);
        for ( auto x : output ) {
            double resul = Functions::getFunctions()->result ( x->sum );
            resul  = resul*var + get<0>(outputNormal);
		    out[count] = resul;
		count++;
        }
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
				next[x.destiny] = 1;
			}
		}
		current = next;
		next.clear();
	}


}
