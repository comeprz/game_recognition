#ifndef ALGOLIN_H
#define ALGOLIN_H

#include <vector>

struct Point {
    float x, y;
    float r, g, b;
    float label;
};

extern std::vector<Point> points;
extern std::vector<float> W;
extern float alpha;
extern int iteration_count;

void initializeWeights();
void train(int iteration_count, float alpha);
void displaySeparation();
void setDataset1();
void setDataset2();
void setDataset3();
void setDataset4();
void setDataset5();
void display();
void keyboard(unsigned char key, int x, int y);
void init();
void setAlgoLinPoints(const std::vector<Point>& newPoints);
void launchAlgoLin(int argc, char** argv);

#endif
