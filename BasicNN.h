#ifndef BASIC_NN
#define BASIC_NN

#include <cmath>
#include <armadillo>
#include <Organism.h>

using Matrix = arma::Mat<double>;
using Vetor = std::vector<double>;

#define IS_NAN(X) ;//throw "Explosion";

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
		allDeltas.resize(nn->layers.size());
		biasDelta.resize(nn->bias.size());
		allDeltas.clear();
		biasDelta.clear();

		for (auto& pt : p) {
			train(pt.first, pt.second);

			for (int indice = nn->layers.size() - 1; indice >= 0; indice--) {
				double epsilon = 0.000008;
				if(rand()%10000<10)
					epsilon*=1000;

				rebalanceWeights(indice, epsilon);
			}
			allDeltas.resize(nn->layers.size());
			biasDelta.resize(nn->bias.size());
			allDeltas.clear();
			biasDelta.clear();

		}
		for (int indice = nn->layers.size() - 1; indice >= 0; indice--) {
			//rebalanceWeights(indice, epsilon);
		}

	}

	PropagationTrainer(BasicNN* nn) :
			nn(nn) {
		allDeltas.resize(nn->layers.size());
		biasDelta.resize(nn->bias.size());
	}
protected:
	void rebalanceWeights(int indice, double eta) {
		using namespace std;
		Vetor prev_outputs;
		Vetor& delta = allDeltas[indice];
		if (indice == 0) {
			prev_outputs = inputValue;
		} else {
			prev_outputs = neuronValues[indice - 1];
		}
		auto& weights = nn->layers[indice];

		for (int i = 0; i < weights.n_rows; i++) {
			for (int j = 0; j < weights.n_cols; j++) {
				double val = eta * delta[j] * prev_outputs[i];

				if(isnan(val)  ){
					weights(i, j)=0;
					continue;
				}
				if( val < -maxVal)val = -maxVal;
				if( val > maxVal)val = maxVal;

				weights(i, j) += val;
			}
		}
		if (indice < nn->bias.size()) {
			auto& bias = nn->bias[indice];
			for (int i = 0; i < nn->bias[indice].size(); i++) {
				double del = biasDelta[indice][i];
				double out = neuronValues[indice][i];
				double val = eta * del * out;

				if(isnan(val)  )val=0;

				if( val < -maxVal)val = -maxVal;
				if( val > maxVal)val = maxVal;

				bias(i, 0) += val;
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

	std::vector<double> calcDelta(int indice, std::vector<double>& targets) { //aka layer and y answer
		std::vector<double> deltas;

		auto get_nb_neurons = [this, indice]() {
			return nn->params[indice+1];
		};
		auto get_output = [this,indice](int i) {
			return this->neuronValues[indice][i];
		};
		double deltas_sum = 0;

		if (indice == nn->layers.size() - 1) {
			for (int i = 0; i < get_nb_neurons(); i++) {
				IS_NAN(targets[i]);
				IS_NAN(neuronValues[indice][i]);
				double alce = (targets[i] - neuronValues[indice][i])*2 ; //

				IS_NAN(alce);
				deltas.push_back(alce);
			}
		} else {
			Matrix& next_weights = nn->layers[indice + 1];
			std::vector<double> next_deltas = allDeltas[indice + 1];
			for (int i = 0; i < get_nb_neurons(); i++) {

				for (int j = 0; j < next_weights.n_rows; j++) {
					IS_NAN(next_weights(j, i));
					IS_NAN(next_deltas[j]);
					auto alce = next_weights(j, i);
					auto alce2=next_deltas[j];
					deltas_sum += alce *alce2;
				}
				IS_NAN(get_output(i));
				double derivative = get_output(i) * (1 - get_output(i));
				double alce = deltas_sum * derivative;
				IS_NAN(alce);
				deltas.push_back(alce);
				deltas_sum = 0;
			}
		}
		return deltas;
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

		Matrix output = layers[0] * input;

		auto temp = output;
		output = output + bias[0];

		inputValue.resize(output.n_rows);
		for (int i = 0; i < output.n_rows; i++) {
			inputValue[i] = output(i, 0);
		}

		for (int i = 1; i < layers.size(); i++) {
			//if(i!=layers.size()-1){
			for (auto& x : output) {
				x = nn->activation(x);
			}

			updateLayer(output, i - 1);
			output = layers[i] * output;
			if (i < bias.size()) {
				output += bias[i];
			}
		}
		updateLayer(output, layers.size() - 1);
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
		for (int indice = nn->layers.size() - 1; indice >= 0; indice--) {
			allDeltas[indice] = calcDelta(indice, output)+allDeltas[indice];
		}
		for (int i = 0; i < nn->bias.size(); i++) {
			calcBiasDelta(i);
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
