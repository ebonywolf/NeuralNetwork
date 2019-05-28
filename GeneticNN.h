#pragma once
#include "Runner.h"
#include <iostream>
#include <sstream>
#include "BasicNN.h"

#include "Organism.h"
#include "GnuOutput.h"
#include "Runner.h"
using std::vector;
using std::string;
using std::get;


namespace wag{

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



struct NNTrainer {
    std::shared_ptr<BasicNN> top;

    BasicNN trainFromPoints(Pontos pts) {
    	double smallest=999999999,biggest=-9999999999;
    	for (auto& pt: pts) {
			if(pt.second < smallest)smallest = pt.second;
			if(pt.second > biggest)biggest = pt.second;
		}
    	double diff = biggest - smallest;


        auto lamd2 = [&pts, diff]( std::function< Vetor (Vetor)> nn) {
            double precision=diff/2.0;
            double maxFit=5000;
            double sum=0;
            double max = maxFit/pts.size();

            for (const auto& x : pts ) {
                std::vector<double> val= {x.first};
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
                std::vector<double> val= {x.first};
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

                Pontos pts = Plotter::generatePoints(-12,12, *nnptr, 100);

                plotter.addPontos("BasicNN",pts);
                plotter.plot();

            };

        plotter.addPontos("Function", pts);

        BasicNNenv env( { 1, 8, 8, 1 }, lamd2);
        env.addProgressListener(progressFunc);

        evoIO = env.getIO();

        Permutations perm(0.02, 0.03, Precision * 2);
        getPermutation(&perm);

        RunnerConfig config;
        config.fitnessTarget = 4995;
        config.generations = 1500;
        config.population = 80;
        auto envPtr = ProgressIOptr(&env);
        {

            Runner runner(config, env, envPtr);
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
