#include "AlgoLin.h"
#include "Mlp.h"
#include "RBFN.h"
#include <iostream>
//#include <GL/glut.h>
#include <GL/freeglut.h> 


int main(int argc, char** argv) {
    int choiceAlgo;

    // Choisir l’algo
    std::cout << "1 - Algo Lineaire\n";
    std::cout << "2 - MLP\n";
     std::cout << "3 - RBFN\n";
    std::cout << "Votre choix : ";
    std::cin >> choiceAlgo;

    if (choiceAlgo == 1) {
        launchAlgoLin(argc, argv);
    }
    else if (choiceAlgo == 2) {
        // puis on lance
        launchMLP(argc, argv);
    }else if(choiceAlgo == 3){
        launchRBFN(argc, argv);
    }
    else {
        std::cout << "Erreur : Choix invalide.\n";
        return 1;
    }

    return 0;
}

