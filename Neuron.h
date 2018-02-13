#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <unordered_map>
#include <memory>
class Neuron;

using Neuronptr = std::shared_ptr<Neuron>;

class Neuron
{
    public:
        struct Connection{
        std::shared_ptr<Neuron> destiny;
        float str;

    };

    Neuron();

    virtual ~Neuron();
    std::vector<Connection> exits;
    void reset();
    void send();
    void updateOutput();

    void createConnection(Neuronptr);
    float sum;
    float output;

    protected:



    private:
};
#endif // NEURON_H
