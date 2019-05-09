#pragma once
#include "Runner.h"
#include <iostream>
#include <sstream>
#include "NeuralNetwork.h"
#include "Organism.h"

#define PRECISION 100000.0
using std::vector;
using std::string;
using std::get;
namespace wag
{
struct FFNN : public Organism, public NeuralNetwork {
    FFNN (  ){}

	FFNN ( vector<int> rows )
	{
		std::vector<vector<Neuronptr>> _neurons ( rows.size() );
		Neuronptr threshold;
		for ( int i = 0; i < _neurons.size() ; i++ ) {
			_neurons[i].resize ( rows[i] );
		}
		for ( int i = 0; i < _neurons.size() - 1; i++ ) {
			for ( int j = 0; j < _neurons[i].size(); j++ ) {
				for ( int w = 0 ; w < _neurons[i + 1].size(); w++ ) {
					_neurons[i][j]->createConnection ( _neurons[i + 1][w] );
					threshold->createConnection ( _neurons[i + 1][w] );
				}
			}
		}
		input.swap ( _neurons[0] );
		for ( int i = 1; i < _neurons.size() - 1; i++ ) {
			for ( int j = 0; j < _neurons[i].size(); j++ ) {
				this->neurons.push_back ( _neurons[i][j] );
			}
		}
		output.swap ( _neurons[ _neurons.size() - 1 ] );

		auto setDna = [] ( vector<Neuronptr>& neuron, wag::GeneChain& geneChain ) {
			geneChain = wag::GeneChain();
			for ( int i = 0; i < neuron.size(); i++ ) {
				for ( auto& x : neuron[i]->exits ) {
					geneChain.push_back ( x.str * PRECISION )  ;
				}
			}
		};

		dna.resize ( 3 );
		setDna ( input, dna[0] );
		setDna ( neurons, dna[1] );

		dna[2] =  wag::GeneChain();
		for ( auto& x : threshold->exits ) {
			dna[2].push_back ( x.str * PRECISION )  ;
		}


	}
	void updateFromDna()
	{
		auto getDna = [] ( vector<Neuronptr>& neuron, wag::GeneChain& geneChain ) {
			int j = 0;
			for ( int i = 0; i < neuron.size(); i++ ) {
				for ( auto& x : neuron[i]->exits ) {
					x.str = ( double ) geneChain[j] / PRECISION;
					j++;
				}
			}
		};
		getDna ( input, dna[0] );
		getDna ( neurons, dna[1] );

        int j = 0;
		for ( auto& x : threshold->exits ) {
			x.str = ( double ) dna[1][j] / PRECISION;
			j++;
		}
	}
    virtual Organismptr cross( Organismptr o){
        std::shared_ptr<FFNN> son = std::static_pointer_cast<FFNN>(Organism::cross(o));
        son->updateFromDna();
        return std::static_pointer_cast<Organism>(son);

    };

    Organismptr clone(){
        FFNN* novo = new FFNN();
        *novo = *this;
        std::shared_ptr<FFNN> pp(novo);
        return std::static_pointer_cast<Organism>(pp);
    }


};

struct FileLearning : public Environment {
		vector<vector<double>> inputs;
		vector<vector<double>> outputs;

		vector<vector<double>> testInputs;
		vector<vector<double>> testOutputs;

		std::tuple<double,double> inputNormal;
		std::tuple<double,double> outputNormal;

		vector<NeuralNetwork> nn;

		FileLearning ( std::istream learnData, std::istream testData, int inputs_num, int outputs_num )
		{
			string line;
            get<0>(inputNormal)=10e10; get<0>(outputNormal)=10e10;
             get<1>(inputNormal)=-10e10; get<1>(outputNormal)=-10e10;

			while ( std::getline ( learnData, line ) ) {
				std::stringstream ss ( line );
				vector<double> in, out;

				for ( int i = 0 ; i < inputs_num; i++ ) {
					double val;
					ss >> val;

					if( val < get<0>(inputNormal))
                        get<0>(inputNormal) = val;

                    if( val > get<1>(inputNormal))
                        get<1>(inputNormal) = val;

					in.push_back ( val );
				}
				inputs.push_back ( in );
				for ( int i = 0 ; i < outputs_num; i++ ) {
					double val;
					ss >> val;
					if( val < get<0>(outputNormal))
                        get<0>(outputNormal) = val;

                    if( val > get<1>(outputNormal))
                        get<1>(outputNormal) = val;

					out.push_back ( val );
				}
				outputs.push_back ( out );
			}

			while ( std::getline ( testData, line ) ) {
				std::stringstream ss ( line );
				vector<double> in, out;

				for ( int i = 0 ; i < inputs_num; i++ ) {
					double val;
					ss >> val;
					in.push_back ( val );
				}
				testInputs.push_back ( in );
				for ( int i = 0 ; i < outputs_num; i++ ) {
					double val;
					ss >> val;
					out.push_back ( val );
				}
				testOutputs.push_back ( out );
			}
		}

		virtual Units createUnits ( int number )
		{



		}

		virtual void runUnits ( Units& units )
		{

		}
		virtual UnitMap readUnits ( std::istream& is )
		{

		}

	private:
		friend Runner;
};

}
