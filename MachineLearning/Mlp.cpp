#include "Mlp.h"
#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <GL/freeglut.h>


static std::vector<int> d;           
static int L = 0;                                 
static std::vector<std::vector<std::vector<double>>> W; // Poids
static std::vector<std::vector<double>> X;       
static std::vector<std::vector<double>> deltas; 

// Données du dataset
std::vector<std::vector<double>> mlp_points;       // Coordonnées 2D
std::vector<std::vector<double>> labels;           // Label pour chaque point


std::vector<std::vector<float>> colors;


/**
 * \brief Initialise la structure (d, L, W, X, deltas)
 */
void initMLP(const std::vector<int>& npl)
{
    d = npl;
    L = static_cast<int>(d.size()) - 1;
    
    W.clear();
    W.resize(d.size());
    
    srand(static_cast<unsigned>(time(nullptr)));
    
    for (int l = 0; l < (int)d.size(); ++l) {
        if (l == 0) {
            continue;
        }
        W[l].resize(d[l - 1] + 1);
        for (int i = 0; i <= d[l - 1]; ++i) {
            W[l][i].resize(d[l] + 1);
            for (int j = 0; j <= d[l]; ++j) {
                if (j == 0) {
                    W[l][i][j] = 0.0;
                } else {
                    double r = double(rand()) / RAND_MAX; 
                    W[l][i][j] = 2.0 * r - 1.0;           
                }
            }
        }
    }

    X.clear();
    X.resize(d.size());
    deltas.clear();
    deltas.resize(d.size());
    for (int l = 0; l < (int)d.size(); ++l) {
        X[l].resize(d[l] + 1);
        deltas[l].resize(d[l] + 1);
        for (int j = 0; j <= d[l]; ++j) {
            if (j == 0) {
                X[l][j] = 1.0;   
            } else {
                X[l][j] = 0.0;
            }
            deltas[l][j] = 0.0;
        }
    }
}


void propagate(const std::vector<double>& inputs, bool is_classification)
{
    for (int j = 1; j <= d[0]; ++j) {
        X[0][j] = inputs[j - 1];
    }
    for (int l_index = 1; l_index <= L; ++l_index) {
        for (int j = 1; j <= d[l_index]; ++j) {
            double total = 0.0;
            for (int i = 0; i <= d[l_index - 1]; ++i) {
                total += W[l_index][i][j] * X[l_index - 1][i];
            }
            X[l_index][j] = total;
            
            if (is_classification || (l_index != L)) {
                X[l_index][j] = std::tanh(total);
            }
        }
    }
}

std::vector<double> predict(const std::vector<double>& inputs, bool is_classification)
{
    propagate(inputs, is_classification);
    std::vector<double> out;
    out.reserve(d[L]);
    for (int j = 1; j <= d[L]; ++j) {
        out.push_back(X[L][j]);
    }
    return out;
}


std::vector<double> trainMLP(const std::vector<std::vector<double>>& all_inputs,
                             const std::vector<std::vector<double>>& all_outputs,
                             double alpha,
                             int iteration_count,
                             bool is_classification)
{
    std::vector<double> losses;
    losses.reserve(iteration_count);
    
    for (int iter = 0; iter < iteration_count; ++iter) {
        int k = rand() % all_inputs.size();
        
        const std::vector<double>& Xk = all_inputs[k];
        const std::vector<double>& Yk = all_outputs[k];

        propagate(Xk, is_classification);
  
        double loss = 0.0;
        for (int j = 1; j <= d[L]; ++j) {
            double diff = X[L][j] - Yk[j - 1];
            loss += diff * diff;
            
            deltas[L][j] = diff;
            if (is_classification) {
                double th = std::tanh(X[L][j]);
                deltas[L][j] *= (1.0 - th * th);
            }
        }
        loss /= (double)d[L];  
        losses.push_back(loss);

        for (int l_index = L; l_index >= 2; --l_index) {
            for (int i = 1; i <= d[l_index - 1]; ++i) {
                double total = 0.0;
                for (int j = 1; j <= d[l_index]; ++j) {
                    total += W[l_index][i][j] * deltas[l_index][j];
                }
                double th = std::tanh(X[l_index - 1][i]);
                double deriv = 1.0 - th * th;
                deltas[l_index - 1][i] = deriv * total;
            }
        }
        
        for (int l_index = 1; l_index <= L; ++l_index) {
            for (int i = 0; i <= d[l_index - 1]; ++i) {
                for (int j = 1; j <= d[l_index]; ++j) {
                    W[l_index][i][j] -= alpha * X[l_index - 1][i] * deltas[l_index][j];
                }
            }
        }
    }
    
    return losses;
}



void trainMLP()
{
    trainMLP(mlp_points, labels, 0.1, 1000, true);
}


std::vector<std::vector<float>> getColors(const std::vector<std::vector<double>>& labs)
{
    std::vector<std::vector<float>> c;
    c.reserve(labs.size());
    for (auto &lab : labs) {
        // lab[0] >= 0 => bleu, sinon rouge
        if (lab[0] >= 0.0) {
            c.push_back({0.0f, 0.0f, 1.0f}); // Bleu
        } else {
            c.push_back({1.0f, 0.0f, 0.0f}); // Rouge
        }
    }
    return c;
}


void displayMLP()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(5.0f);

    glBegin(GL_POINTS);
    int nx = 100, ny = 100;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double x1 = i / double(nx);
            double x2 = j / double(ny);
            std::vector<double> input = { x1, x2 };

            std::vector<double> output = predict(input, true);

            if (output[0] >= 0.0) {
                glColor3f(0.68f, 0.85f, 0.90f);
            } else {
                glColor3f(1.0f, 0.75f, 0.80f);
            }

            glVertex2f(float(x1 * 2.0 - 1.0), float(x2 * 2.0 - 1.0));
        }
    }
    glEnd();
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < mlp_points.size(); ++i) {
        glColor3f(colors[i][0], colors[i][1], colors[i][2]);
        float px = float(mlp_points[i][0] * 2.0 - 1.0);
        float py = float(mlp_points[i][1] * 2.0 - 1.0);
        glVertex2f(px, py);
    }
    glEnd();

    glFlush();
}


void setMlpData(const std::vector<std::vector<double>>& newPoints,
                const std::vector<std::vector<double>>& newLabels)
{
    mlp_points = newPoints;
    labels     = newLabels;
    colors     = getColors(labels);
}

void setMlpDataset1()
{
    mlp_points = {
        {0.1, 0.1},
        {0.1, 0.8},
        {0.8, 0.1},
        {0.8, 0.8}
    };
    labels = {
        {-1.0},
        { 1.0},
        { 1.0},
        {-1.0}
    };
    colors = getColors(labels);
    trainMLP();
}

void setMlpDataset2()
{
    mlp_points = {
        {0.1, 0.1},
        {0.2, 0.3},
        {0.3, 0.3}
    };
    labels = {
        { 1.0},
        {-1.0},
        {-1.0}
    };
    colors = getColors(labels);
    trainMLP();
}

void setMlpDataset3()
{
    mlp_points.clear();
    labels.clear();

    for (int i = 0; i < 50; ++i) {
        double x = (rand() / double(RAND_MAX)) * 0.4; 
        double y = (rand() / double(RAND_MAX)) * 0.4;   
        mlp_points.push_back({ x, y });
        labels.push_back({ 1.0 });
    }

    for (int i = 0; i < 50; ++i) {
        double x = 0.6 + (rand() / double(RAND_MAX)) * 0.4;
        double y = 0.6 + (rand() / double(RAND_MAX)) * 0.4; 
        mlp_points.push_back({ x, y });
        labels.push_back({ -1.0 });
    }

    colors = getColors(labels);
    trainMLP();
}

void setMlpDataset4()
{
    mlp_points.clear();
    labels.clear();

    for (int i = 0; i < 500; ++i) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;  

        double label = (std::fabs(x - 0.5) <= 0.3 || std::fabs(y - 0.5) <= 0.3)
                       ? 1.0
                       : -1.0;

        mlp_points.push_back({x, y});
        labels.push_back({label});
    }

    colors = getColors(labels);
    trainMLP();
}

void mlpKeyboard(unsigned char key, int x, int y)
{
    switch (key) {
    case '1':
        setMlpDataset1();
        break;
    case '2':
        setMlpDataset2();
        break;
    case '3':
        setMlpDataset3();
        break;
    case '4':
        setMlpDataset4();
        break;
    case 'q':
        exit(0);
        break;
    default:
        std::cout << "Touche inconnue : " << key << std::endl;
    }
    glutPostRedisplay();
}

void launchMLP(int argc, char** argv)
{
    std::vector<int> architecture = {2, 4, 1};
    initMLP(architecture);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("MLP Visualization");

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);

    glutDisplayFunc(displayMLP);
    glutKeyboardFunc(mlpKeyboard);

    glutMainLoop();
}