#ifndef BASIC_NN
#define BASIC_NN

#include <cmath>
#include <armadillo>
#include <Organism.h>
using Matrix = arma::Mat<double>;
using Vetor = std::vector<double>;
#define Precision 10000000.0
#define E 2.718281828459045
#define PI 3.14159265359
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



}
#endif // BASIC_NN
