#ifndef BASIC_NN
#define BASIC_NN

#include <cmath>
#include <armadillo>
#include <Organism.h>

using Matrix = arma::Mat<double>;
using Vetor = std::vector<double>;

#define IS_NAN(X)if(X!=X || std::isinf(X)) throw "Explosion";

#define Precision 10000000.0
#define M_E 2.718281828459045
#define M_PI 3.14159265359
namespace wag {
using Pontos = std::vector<std::pair<double, double> >;
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
	~BasicNN();
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

	inline double activation(double x) const;

	double operator()(double d) const;

	Vetor operator()(const Vetor& _in) const;

	void fromDna();
	void toDna();

	virtual Organismptr cross(Organismptr o) const override;

	Organismptr clone() const override;

protected:
	std::vector<double> getOutput(const std::vector<double>& _in) const;

public:
	std::vector<int> params;
	std::vector<Matrix> layers;
	std::vector<Matrix> bias;

};

template<class T>
std::vector<T> operator+(std::vector<T> a,const std::vector<T>& b){
	for (int i = 0; i < a.size() && i < b.size(); i++) {
		a[i]+=b[i];
	}
	return a;
}
template<class T>
std::ostream& operator<<(std::ostream& os,const std::vector<T>& b){
	for (auto& x: b) {
		os<< x<<" " ;
	}
	return os;
}

struct PropagationTrainer {
	int epoch;
	BasicNN* nn;
	double minVal = 0.00007;
	double maxVal = 20000;
	std::vector<std::vector<double> > neuronValues;

	Vetor inputValue;
	std::vector<Vetor> allDeltas;
	std::vector<Vetor> biasDelta;
	void train(Pontos p) {
		for (auto& pt : p) {
			train(pt.first, pt.second);
			double epsilon = 0.0012;

			rebalanceWeights(epsilon);

		}


	}

	PropagationTrainer(BasicNN* nn) :
			nn(nn) {
		allDeltas.resize(nn->layers.size());
		biasDelta.resize(nn->bias.size());
	}
protected:
	void rebalanceWeights(double eta) {

		for (int i = 0; i < neuronValues.size()-1; i++) {
			auto& inputs = neuronValues[i];
			Matrix& weights = nn->layers[i];
			for (int n = 0; n < neuronValues[i+1].size(); n++) {
				double delta =allDeltas[i+1][n] * eta;
				IS_NAN(delta);
				for (int j = 0; j < inputs.size(); j++) {
					double var = inputs[j] *delta; ;
					IS_NAN(var);
					weights(n,j)+=var;
				}
				if( i < nn->bias.size())
					nn->bias[i][n]+=delta;
			}
		}

	}
	void train(double input, double output) {
		Vetor vi = { input };
		Vetor vo = { output };
		train(vi, vo);
	}

	void updateLayer(Matrix& m, int n) {
		neuronValues[n].resize(m.n_rows);
		for (int i = 0; i < m.n_rows; i++) {
			IS_NAN(m(i, 0));
			neuronValues[n][i] = m(i, 0);
		}
	}


	std::vector<double> updateNeuronValues(const std::vector<double>& _in) {
		neuronValues.resize(nn->layers.size() + 1);
		using namespace std;
		auto& layers = nn->layers;
		auto& bias = nn->bias;
		if (_in.size() != layers[0].n_cols) {
			std::stringstream ss;
			ss << "Wrong input Amount got:" << _in.size() << " Expected:" << layers[0].n_cols;
			throw runtime_error(ss.str());
		}
		Matrix input = Matrix(_in);
		for (auto x : input) {
			IS_NAN(x);
		}
		for (auto x : layers[0]) {
			IS_NAN(x);
		}
		inputValue.resize(input.n_rows);
		for (int i = 0; i < input.n_rows; i++) {
			inputValue[i] = input(i, 0);
		}
		updateLayer(input, 0 );

		Matrix output = layers[0] * input;

		auto temp = output;
		output = output + bias[0];

		for (int i = 1; i < layers.size(); i++) {
			//if(i!=layers.size()-1){
			for (auto& x : output) {
				x = nn->activation(x);
			}

			updateLayer(output, i );
			output = layers[i] * output;
			if (i < bias.size()) {
				output += bias[i];
			}
		}
		updateLayer(output, layers.size() );
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
		allDeltas.resize(neuronValues.size());

		for (int i = neuronValues.size()-1 ; i >= 1; i--) {
			allDeltas[i].resize( neuronValues[i].size() )  ;
			//auto& layer = nn->layers[i];
			Vetor errors ;
			if( i != neuronValues.size()-1 ){//middle nodes
				for (int j = 0; j < neuronValues[i].size(); j++) {
					double error = 0.0;
					for (int k = 0; k < neuronValues[i+1].size(); k++) {
						double delta = allDeltas[i+1][k];
						IS_NAN(delta);
						double weight = nn->layers[i-1](j,k);
						IS_NAN(weight);
						double err = delta*weight;
						IS_NAN(err);
						error += err;
					}
					errors.push_back(error);
				}
			}else{//last node
				for (int j = 0; j < neuronValues[i].size(); j++) {
					double val = neuronValues[i][j];
					double error = output[j]- val;
					IS_NAN(error);
					errors.push_back(error);
				}
			}
			for (int j = 0; j < neuronValues[i].size(); j++) {
				double val = neuronValues[i][j];
				double derivative = val*(1.0-val);
				double eps=0.0000001 ;

				if( abs(derivative)<0.0000001 ){
					if(derivative>0){
						derivative = eps;
					}else{
						derivative = -eps;
					}
				}

				double result= errors[j] * derivative;
				IS_NAN(derivative);
				IS_NAN(result);

				allDeltas[i][j] = result;
			}
		}
	}
	void calcBiasDelta(int indice) {

		Vetor delta;
		delta = allDeltas[indice];
		double deltas_sum = 0;
		auto& mat = nn->bias[indice];
		biasDelta[indice].resize(nn->bias[indice].size());

		for (int i = 0; i < nn->bias[indice].size(); i++) {
			double alce = delta[i];

			double val = nn->bias[indice](i, 0);
			biasDelta[indice][i] += alce * nn->bias[indice](i, 0);
			IS_NAN(val);
		}
	}

private:
	const std::vector<int> layers;

};

}
#endif // BASIC_NN
