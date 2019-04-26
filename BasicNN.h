#ifndef BASIC_NN
#define BASIC_NN

#include <cmath>
#include <armadillo>
#include "Organism.h"
#include "GnuOutput.h"
#include "Runner.h"

using Matrix = arma::Mat<double>;
using Vetor = std::vector<double>;
#define Precision 10000000.0
#define E 2.718281828459045
namespace wag {

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
	BasicNN(const std::vector<int>& sizes) :
			params(sizes) {

		if (sizes.size() < 3)
			throw std::runtime_error("invalid params");
		for (double i = 0; i < sizes.size() - 1; i++) {
			layers.emplace(layers.end(), arma::randu(sizes[i + 1], sizes[i]));
			if (i > 0) {
				bias.emplace(bias.end(), arma::randu(sizes[i], 1));
			}
		}
		for (auto& layer : layers) {
			for (auto& x : layer) {
				x = (x * 5) - 2.5;
			}
		}
		for (auto& layer : bias) {
			for (auto& x : layer) {
				x = (x * 5) - 2.5;
			}
		}

	}
	~BasicNN() {
	}
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

	inline double activation(double x) const {
		//return 1.0/(1.0+exp(-x));
		// return std::tanh(x);

		if (x > 5)
			return x;
		if (x < -5)
			return 0.000001;
		return std::log(1.0 + std::exp(x));       //1.0/(1.0+exp(-x));
	}

	double operator()(double d) const {
		std::vector<double> novo(params[0]);
		novo[0] = d;
		auto ret = getOutput(novo);
		return ret[0];
	}

	Vetor operator()(const Vetor& _in) const {
		return getOutput(_in);
	}

	void fromDna() {
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
	void toDna() {
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

	virtual Organismptr cross(Organismptr o) const override {
		std::shared_ptr<BasicNN> son = std::static_pointer_cast<BasicNN>(Organism::cross(o));
		son->fromDna();
		return std::static_pointer_cast<Organism>(son);

	}
	;

	Organismptr clone() const override {
		BasicNN* novo = new BasicNN(*this);
		std::shared_ptr<BasicNN> pp(novo);
		return std::static_pointer_cast<Organism>(pp);
	}

protected:
	std::vector<double> getOutput(const std::vector<double>& _in) const {
		using namespace std;

		if (_in.size() != layers[0].n_cols) {
			stringstream ss;
			ss << "Wrong input Amount got:" << _in.size() << " Expected:" << layers[0].n_cols;
			throw runtime_error(ss.str());
		}

		Matrix input = Matrix(_in);

		Matrix output = layers[0] * input;
		output = output + bias[0];
		for (int i = 1; i < layers.size(); i++) {
			for (auto& x : output) {

				x = activation(x);
			}
			output = layers[i] * output;
			if (i < bias.size()) {
				output += bias[i];
			}
		}
		if (!output.is_vec())
			throw std::runtime_error("non vector output");

		std::vector<double> result;
		for (auto& x : output) {
			result.emplace(result.end(), x);
		}

		return result;
	}

public:
	std::vector<int> params;
	std::vector<Matrix> layers;
	std::vector<Matrix> bias;

};

struct BasicNNenv: public Environment, public ProgressIO {

	using FitnessFunction = std::function< double (std::function< Vetor (const Vetor&)> ) >;

	BasicNNenv(std::vector<int> nnParams, FitnessFunction func) :
			func(func), nn_params(nnParams) {
	}

	virtual Units createUnits(int number) {
		Units novo;
		for (int i = 0; i < number; i++) {
			std::shared_ptr<Organism> org = std::make_shared<BasicNN>(nn_params);
			novo.push_back(org);
		}
		return novo;
	}

	virtual std::shared_ptr<EvolutionIO> getIO() {
		return io;
	}

	virtual void runUnits(Units& units) {
		using namespace std;

		for (auto& x : units) {       ///run Units
			auto nn = std::dynamic_pointer_cast<BasicNN>(x);
			nn->toDna();
			int fit;
			try {
				fit = func(*nn);
			} catch (NNException& e) {
				std::ofstream file("bugnn", ios::trunc | ios::binary);  //print nn to file to debug it
				if (file.is_open()) {

					file << (*nn);
					file.close();
					exit(0);
				} else {
					std::cout << "fooo" << std::endl;
				}
			}
			x->fitness = fit;
		}
	}
	virtual void printProgress(Progress p) {
		progressOut.printProgress(p);
		progressOut.plot();

		for (auto& x : listeners) {
			try {
				x(p);
			} catch (std::runtime_error& error) {
				std::cout << error.what() << std::endl;
			}
		}

	}
	virtual void saveProgress(Progress p) {
		progressOut.saveProgress(p);
	}
	virtual Progress loadProgress() {
		progressOut.loadProgress();
	}

	virtual void addProgressListener(std::function<void(Progress)> f) {
		listeners.push_back(f);
	}

private:
	FitnessFunction func;
	std::vector<int> nn_params;

	EvolutionIOptr io = std::make_shared<EvolutionMemory>();
	Gnuplot_os progressOut;

	std::vector<std::function<void(Progress)>> listeners;

	friend Runner;
};

template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& t) {
	for (int i = 0; i < t.size(); i++) {
		std::cout << i << ":" << t[i] << std::endl;
	}
	return os;

}

struct NNTrainer {
	std::shared_ptr<BasicNN> top;

	BasicNN trainFromPoints(Pontos pts) {

		auto lamd2 = [&pts]( std::function< Vetor (Vetor)> nn) {
			double precision=20.0;
			double maxFit=5000;
			double sum=0;
			double max = maxFit/pts.size();

			for (const auto& x : pts ) {
				vector<double> val= {x.first};
				auto vec = nn(val);
				double y = vec[0];
				if( y!=y ) {
					std::cout << "Nan error:"<<x.first << std::endl;
					throw NNException(x.first);
				}
				double error = (x.second -y)*(x.second -y);

				error=(-max/precision)*error+max;

				if(error<0)error=0;
				sum+=error;
			}
			return sum;
		};

		auto lamd = [&pts]( std::function< Vetor (Vetor)> nn) {
			double precision=10000.0;
			double maxFit=5000;
			double sum=2;
			for (const auto& x : pts ) {
				vector<double> val= {x.first};
				auto vec = nn(val);
				double y = vec[0];
				if( y!=y ) {
					std::cout << "Nan error:"<<x.first << std::endl;
					throw NNException(x.first);
				}
				double error = (x.second -y)*(x.second -y);
				sum+=error;

			}
			return precision/ log(22026.4657948+sum);
		};

		auto progressFunc = [this](Progress p ) {

			auto unitmap = evoIO->readPopulation(p.generations);

			auto orgptr = unitmap[p.highestId];
			auto nnptr = (BasicNN*)orgptr.get();  //;std::dynamic_pointer_cast<BasicNN>(orgptr);

				std::shared_ptr<BasicNN> alce= std::dynamic_pointer_cast<BasicNN>(orgptr);
				if(!top || top->fitness < alce->fitness) {
					top = alce;
				}
				/*
				 std::cout << "========Fitness======: "<<p.highestFitness << std::endl;
				 for (auto x : nnptr->layers ){
				 std::cout << x << std::endl;
				 }
				 std::cout << "===BIAS===" << std::endl;
				 for (auto x : nnptr->bias ){
				 std::cout << x << std::endl;
				 }
				 */
				Pontos pts = Plotter::generatePoints(-12,12, *nnptr, 100);

				plotter.addPontos("BasicNN",pts);
				plotter.plot();

			};

		plotter.addPontos("Function", pts);

		BasicNNenv env( { 1, 5,5,  1 }, lamd2);
		env.addProgressListener(progressFunc);

		evoIO = env.getIO();

		Permutations perm(0.02, 0.03, Precision *30);
		getPermutation(&perm);

		RunnerConfig config;
		config.fitnessTarget = 4995;
		config.generations = 500;
		config.population = 80;
		auto envPtr =ProgressIOptr(&env);
		{

		Runner runner(config, env,envPtr );
		Progress p = runner.startSimulation();
		}

		return *top.get();
	}

	template<class F>
	BasicNN trainBasicFunction(F func) {
		using namespace std;
		Pontos pts;
		for (int i = 0; i < 20; i++) {
			func.setT(i);
			auto alce = Plotter::generatePoints(-10 + i, -9 + i, func, 12);
			pts.insert(pts.end(), alce.begin(), alce.end());
		}
		return trainFromPoints(pts);
	}

private:
	Plotter plotter;
	EvolutionIOptr evoIO;

};

}

#endif // BASIC_NN
