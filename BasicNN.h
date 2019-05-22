#ifndef BASIC_NN
#define BASIC_NN

#include <cmath>
#include <armadillo>
#include <Organism.h>
using Matrix = arma::Mat<double>;
using Vetor = std::vector<double>;
#define Precision 10000000.0
#define M_E 2.718281828459045
#define M_PI 3.14159265359
namespace wag {
using Pontos =   std::vector<std::pair<double, double> >;
struct NNException: public std::exception {
	NNException(double val) :
			val(val) {
	}
	double val;

};

class BasicNN: public Organism, public std::function<Vetor(Vetor)> {
public:

	BasicNN() {
	}
	BasicNN(const std::vector<int>& sizes);
	~BasicNN() ;
	friend std::ostream& operator<<(std::ostream& out, const BasicNN& nn) {
		int size = nn.params.size();

		out.write((char *) &size, sizeof(int));
		for (int i = 0; i < size; i++) {
			int alce = nn.params[i];
			out.write((char*) &alce, sizeof(int));
		}
		const Organism* org = &nn;
		out << *org;
		return out;
	}

	friend std::istream& operator>>(std::istream& in, BasicNN& nn) {
		int size;
		in.read((char *) &size, sizeof(int));

		std::vector<int> params(size);

		for (int i = 0; i < size; i++) {
			int alce;
			in.read((char*) &alce, sizeof(int));
			params[i] = alce;
		}
		nn = BasicNN(params);
		Organism* org = &nn;
		in >> *org;
		nn.fromDna();
		return in;
	}

	inline double activation(double x) const ;

	double operator()(double d) const ;

	Vetor operator()(const Vetor& _in) const;

	void fromDna();
	void toDna() ;

	virtual Organismptr cross(Organismptr o) const override;

	Organismptr clone() const override ;

protected:
	std::vector<double> getOutput(const std::vector<double>& _in) const ;

public:
	std::vector<int> params;
	std::vector<Matrix> layers;
	std::vector<Matrix> bias;

};

struct PropagationTrainer{
    int epoch;
    BasicNN* nn;
    std::vector< std::vector<double> >neuronValues;

    PropagationTrainer(BasicNN* nn):nn(nn){
    }
    int getOutputNum()const {
        nn->params.back();
    }
    void train(Pontos p){
    	for (auto& pt: p) {
			train(pt.first, pt.second);
		}
    }

    void train(double input, double output){
    	Vetor vi = {input};
    	Vetor vo = {output};
    	train(vi,vo);
    }

    void updateLayer(Matrix& m, int n){
        neuronValues[n].resize(m.n_rows);
        for (int i = 0; i < m.n_rows; i++) {
           neuronValues[n][i]= m(i,0);
        }
    }

    std::vector<double> updateNeuronValues(const std::vector<double>& _in){
        neuronValues.resize( nn->layers.size()+1);
        using namespace std;
        auto& layers = nn->layers;
        auto& bias = nn->bias;
        if (_in.size() != layers[0].n_cols) {
                std::stringstream ss;
                ss << "Wrong input Amount got:" << _in.size() << " Expected:" << layers[0].n_cols;
                throw runtime_error(ss.str());
            }
            Matrix input = Matrix(_in);
            updateLayer(input,0);

            Matrix output = layers[0] * input;
            output = output + bias[0];
            for (int i = 1; i < layers.size(); i++) {
                for (auto& x : output) {
                    x = nn->activation(x);
                }
                updateLayer(output,i);
                output = layers[i] * output;
                if (i < bias.size()) {
                    output += bias[i];
                }
            }
            updateLayer( output ,layers.size() );
            if (!output.is_vec())
                throw std::runtime_error("non vector output");

            std::vector<double> result;
            for (auto& x : output) {
                result.emplace(result.end(), x);
            }
            return result;
    }

    void train(Vetor input, Vetor output) {
        updateNeuronValues(input);
        auto& layers = nn->layers;
        auto& bias = nn->bias;

        double epsilon=0.03;
        double change_rate = 2.0*epsilon  ;

        Vetor realAnswer = output;


//        std::cout<< "val"<<neuronValues.size()<<" "<< layers.size() << std::endl;

        for (int layer = layers.size()-1; layer >= 0; layer--) {
                int outNum = neuronValues[layer+1].size();
                int neuronNum = neuronValues[layer].size();

                if(outNum!=realAnswer.size()){
                	std::cout<< "Out:"<<outNum<<" "<<realAnswer.size() << std::endl;
                	throw std::runtime_error("Wrong backpropagation");
                }
                Vetor nextAnswer(neuronNum);
               // nextAnswer.resize(neuronNum);
                for (int neuron = 0;  neuron< neuronNum; neuron++) {
                	double neuronVal = neuronValues[layer][neuron];

                	for (int nextNeuron = 0; nextNeuron< outNum; nextNeuron++) {
                		double realAns = realAnswer[nextNeuron];
                        double answerVal = neuronValues[layer+1][nextNeuron];
                        double cost = answerVal - realAnswer[nextNeuron];
                        cost*=cost;

                        double answerCost = 2 * (answerVal- realAns);

                        double derivative  = 0;
                        if(layer == layers.size()-1){
                        	std::cout<< "Last" << std::endl;
                        	derivative=nn->layers[layer](nextNeuron, neuron);
                        }else{
                        	std::cout<< "NotLast" << std::endl;
                        	derivative=(1 - answerVal) * answerVal;
                        }

                        double previousAnswer = neuronVal;

                        double shift = answerCost * derivative * previousAnswer ;
                        std::cout<< "nn:"<<answerVal<<" "<<realAns << " "<<cost<<std::endl;

                        std::cout<< "Shift:"<<shift<<" "<<answerCost<< " "<<derivative<< " "<<previousAnswer << std::endl;

                        double val = nn->layers[layer](nextNeuron, neuron);
                        nn->layers[layer](nextNeuron, neuron) = val + shift*change_rate;

                      //  weightShifts[neuron]+=shift;
                        nextAnswer[neuron]+=cost;
                      //  double error = answerVal  -
                    }
                	nextAnswer[neuron]/=(double)outNum;
                	nextAnswer[neuron] = neuronValues[layer][neuron]+ nextAnswer[neuron];
                }
                realAnswer = std::move(nextAnswer);
           }
    }
private:
    const std::vector<int> layers;

};


}
#endif // BASIC_NN
