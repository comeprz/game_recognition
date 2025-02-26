#ifndef RBFN_H
#define RBFN_H

#include <vector>

struct PointR {
    float x, y;
    float r, g, b;
    float label;
};

extern std::vector<float> WR;
extern std::vector<std::vector<float>> centers;


extern std::vector<PointR> pointRs;

void setRBFNPoints(const std::vector<PointR>& newPoints) ;
void initializeWeightsR();
void initializeCenters();
float gaussian(const std::vector<float>& center, float x, float y);
float predictR(const std::vector<float>& W, float x, float y) ;
void drawAxesR();
void displaySeparationR() ;
void trainR(int iteration_count, float alpha) ;
void displayR();
void setDataset1R();
void setDataset2R();
void setDataset3R();
void setDataset4R();
void setDataset5R();
void keyboardR(unsigned char key, int x, int y);
void initR();
void launchRBFN(int argc, char** argv) ;





#endif