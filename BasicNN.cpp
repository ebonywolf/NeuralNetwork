#include "BasicNN.h"


namespace wag {

BasicNN::BasicNN(const std::vector<int>& sizes) :
		params(sizes) {

	if (sizes.size() < 3)
		throw std::runtime_error("invalid params");
	for (double i = 0; i < sizes.size() - 1; i++) {
		layers.emplace(layers.end(), arma::randu(sizes[i + 1], sizes[i]));
		if (i > 0) {
			bias.emplace(bias.end(), arma::randu(sizes[i], 1));
		}
	}
	//bias.emplace(bias.end(), arma::randu(sizes.back(), 1));

	for (auto& layer : layers) {
		for (auto& x : layer) {
			x = (x * 30) - 15;
		}
	}
	for (auto& layer : bias) {
		for (auto& x : layer) {
			x = (x * 30) - 15;
		}
	}


}
BasicNN::~BasicNN() {
}

inline double BasicNN::activation(double x) const {

	double alce = (1.0/(1.0+exp(-x)));
	if( alce < 0.05)alce=0.05;
	if( alce > 0.95)alce = 0.95;
	return alce;



	//x = std::tanh(x) * (PI/2)*.95;
	//return std::tan(x);
	 //return std::tanh(x);

	if (x > 5)
		return x;
	if (x < -5)
		return 0.000001;
	return std::log(1.0 + std::exp(x));       //1.0/(1.0+exp(-x));
}

double BasicNN::operator()(double d) const {
	std::vector<double> novo(params[0]);
	novo[0] = d;
	auto ret = getOutput(novo);
	return ret[0];
}

Vetor BasicNN::operator()(const Vetor& _in) const {
	return getOutput(_in);
}

void BasicNN::fromDna() {
	auto it = dna.begin();
	for (auto& x : layers) {
		auto itg = it->begin();
		for (int i = 0; i < x.n_rows; i++) {
			for (int j = 0; j < x.n_cols; j++) {

				x(i, j) = (*itg) / Precision;
				itg++;
			}
		}
		it++;
	}
	for (auto& x : bias) {
		auto itg = it->begin();
		for (int i = 0; i < x.n_rows; i++) {
			for (int j = 0; j < x.n_cols; j++) {
				x(i, j) = *itg / Precision;
				itg++;
			}
		}
		it++;
	}
}
void BasicNN::toDna() {
	Dna novodna;
	for (auto& x : layers) {
		GeneChain novo;
		for (int i = 0; i < x.n_rows; i++) {
			for (int j = 0; j < x.n_cols; j++) {
				novo.push_back(x(i, j) * Precision);
			}
		}
		novodna.push_back(novo);
	}
	for (auto& x : bias) {
		GeneChain novo;
		for (int i = 0; i < x.n_rows; i++) {
			for (int j = 0; j < x.n_cols; j++) {
				novo.push_back(x(i, j) * Precision);
			}
		}
		novodna.push_back(novo);
	}
	dna = novodna;
}

Organismptr BasicNN::cross(Organismptr o) const  {
	std::shared_ptr<BasicNN> son = std::static_pointer_cast<BasicNN>(Organism::cross(o));
	son->fromDna();
	return std::static_pointer_cast<Organism>(son);
}


Organismptr BasicNN::clone() const  {
	BasicNN* novo = new BasicNN(*this);
	std::shared_ptr<BasicNN> pp(novo);
	return std::static_pointer_cast<Organism>(pp);
}


std::vector<double> BasicNN::getOutput(const std::vector<double>& _in) const {
	using namespace std;

	if (_in.size() != layers[0].n_cols) {
		stringstream ss;
		ss << "Wrong input Amount got:" << _in.size() << " Expected:" << layers[0].n_cols;
		throw runtime_error(ss.str());
	}
	Matrix input = Matrix(_in);
	for (auto& x : input) {
		x = activation(x);
	}

	Matrix output = layers[0] * input;
	output = output + bias[0];
	for (int i = 1; i < layers.size(); i++) {
		for (auto& x : output) {
			x = activation(x);
		}
		output = layers[i] * output;
		if(i<bias.size())
			output += bias[i];
	}
	if (!output.is_vec())
		throw std::runtime_error("non vector output");

	std::vector<double> result;
	for (auto& x : output) {
		result.emplace(result.end(), x);
	}

	return result;
}

}
