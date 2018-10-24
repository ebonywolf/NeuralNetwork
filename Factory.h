#ifndef FACTORY_H
#define FACTORY_H
#include "NeuralNetwork.h"
#include "Neuron.h"
#include <string>
#include <vector>
#include <ProjGaia/SFML/MultiSprite.h>
#include <ProjGaia/SFML/DrawableObject.h>
#include <ProjGaia/SFML/DrawableSprite.h>
#include <ProjGaia/SFML/Renderer.h>
#include <ProjGaia/SFML/ColoredShape.h>
#include <SFML/Graphics/CircleShape.hpp>
using namespace pg;

pg::DrawableType* createNeuron(Neuronptr m, Coord c){

    ColoredShape* circle = new ColoredShape((new sf::CircleShape(20,30) ));
    std::list<pg::DrawableSprite*> spriteList;
    spriteList.push_back(circle);
    DrawableSprite* dro = new MultiSprite(spriteList);
    circle->getHitBox()->position = c;
    circle->setColor(255,0,0,255);
    pg::DrawableType* finalsprite= new DrawableObject<Neuronptr>(m,dro);


    return finalsprite;
}


class Factory
{
    public:


        static NeuralNetwork basic(int inputN,int middle , int outputN,Renderer& ren ){

            NeuralNetwork novo;


            std::vector<Neuronptr> input;
            std::vector<Neuronptr> neurons;
            std::vector<Neuronptr> output;

            Neuronptr threshold;

             for (int i=0;i<outputN;i++){
                Coord c = Coord(250,i*100+50);
                Neuronptr nr = Neuronptr(new Neuron());
                auto spri = createNeuron(nr,c);
                ren.addDrawable(spri);
                output.push_back(nr);


             }
            for (int i=0;i<inputN;i++){
                Coord c = Coord(50,i*100+50);
                 Neuronptr nr = Neuronptr(new Neuron());
                auto spri = createNeuron(nr,c);
                ren.addDrawable(spri);
                input.push_back( nr);
            }

            for (int i=0;i<middle;i++){
                 Coord c = Coord(150,i*100+50);
                  Neuronptr nr = Neuronptr(new Neuron());
                auto spri = createNeuron(nr,c);
                ren.addDrawable(spri);
                neurons.push_back( nr);
            }
            for(auto x: input){
                   for(auto y: neurons){
                       x->createConnection(y);
                   }
            }
            for(auto x: neurons){
                   for(auto y: output){
                       x->createConnection(y);
                   }
            }
            novo = NeuralNetwork(input, neurons, output);



            return novo;
        }



        virtual ~Factory() {}
    protected:
    private:
};

#endif // FACTORY_H
