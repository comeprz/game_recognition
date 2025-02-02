#include "Mlp.h"
#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <GL/glut.h>


MyMLP mlp({ 2, 3, 1 });


MyMLP::MyMLP(const std::vector<int>& npl) {
    d = npl;
    L = d.size() - 1;

    W.resize(L + 1);
    for (int l = 1; l <= L; ++l) {
        W[l].resize(d[l - 1] + 1);
        for (int i = 0; i <= d[l - 1]; ++i) {
            W[l][i].resize(d[l] + 1);
            for (int j = 1; j <= d[l]; ++j) {
                W[l][i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }

    X.resize(L + 1);
    deltas.resize(L + 1);
    for (int l = 0; l <= L; ++l) {
        X[l].resize(d[l] + 1, 0.0);
        deltas[l].resize(d[l] + 1, 0.0);
    }
}


std::vector<std::vector<double>> mlp_points = {
    {0.0, 0.0},
    {1.0, 1.0},
    {1.0, 0.0},
    {0.0, 1.0}
};

std::vector<std::vector<double>> labels = {
    {-1.0},
    {-1.0},
    {1.0},
    {1.0}
};

std::vector<std::vector<float>> getColors(const std::vector<std::vector<double>>& labels) {
    std::vector<std::vector<float>> colors;
    for (const auto& y : labels) {
        if (y[0] >= 0) {
            colors.push_back({ 0.0f, 0.0f, 1.0f }); // Bleu
        }
        else {
            colors.push_back({ 1.0f, 0.0f, 0.0f }); // Rouge
        }
    }
    return colors;
}

std::vector<std::vector<float>> colors = getColors(labels);

std::vector<double> MyMLP::predict(const std::vector<double>& inputs, bool is_classification) {
    _propagate(inputs, is_classification);
    return { X[L].begin() + 1, X[L].end() };
}

void MyMLP::train(const std::vector<std::vector<double>>& all_inputs,
    const std::vector<std::vector<double>>& all_outputs,
    double alpha, int iteration_count, bool is_classification) {
    for (int iteration = 0; iteration < iteration_count; ++iteration) {
        int k = rand() % all_inputs.size();
        const std::vector<double>& Xk = all_inputs[k];
        const std::vector<double>& Yk = all_outputs[k];

        _propagate(Xk, is_classification);
        for (int j = 1; j <= d[L]; ++j) {
            deltas[L][j] = X[L][j] - Yk[j - 1];
        }

        for (int l = L - 1; l >= 1; --l) {
            for (int i = 1; i <= d[l - 1]; ++i) {
                double total = 0.0;
                for (int j = 1; j <= d[l]; ++j) {
                    total += W[l + 1][i][j] * deltas[l + 1][j];
                }
                deltas[l][i] = (1.0 - tanh(X[l][i]) * tanh(X[l][i])) * total;
            }
        }

        for (int l = 1; l <= L; ++l) {
            for (int i = 0; i <= d[l - 1]; ++i) {
                for (int j = 1; j <= d[l]; ++j) {
                    W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];
                }
            }
        }
    }
}

void MyMLP::_propagate(const std::vector<double>& inputs, bool is_classification) {
    for (int j = 1; j <= d[0]; ++j) {
        X[0][j] = inputs[j - 1];
    }
    for (int l = 1; l <= L; ++l) {
        for (int j = 1; j <= d[l]; ++j) {
            double total = 0.0;
            for (int i = 0; i <= d[l - 1]; ++i) {
                total += W[l][i][j] * X[l - 1][i];
            }
            X[l][j] = tanh(total);
        }
    }
}

void initMLP() {
    if (mlp_points.empty()) {
        std::cerr << "Erreur: Le dataset mlp_points est vide, impossible d'initialiser MLP." << std::endl;
        return;
    }

    mlp = MyMLP({ 2, 3, 1 });

    std::cout << "Initialisation du MLP terminée." << std::endl;
}

void displayMLP() 
{
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(5.0);
    glBegin(GL_POINTS);

    int nx = 100, ny = 100;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double x1 = i / (double)nx;
            double x2 = j / (double)ny;
            std::vector<double> input = { x1, x2 };
            std::vector<double> output = mlp.predict(input, true);

            if (output[0] >= 0) {
                glColor3f(0.68f, 0.85f, 0.90f);
            }
            else {
                glColor3f(1.0f, 0.75f, 0.80f);
            }

            glVertex2f(x1 * 2 - 1, x2 * 2 - 1);
        }
    }
    glEnd();

    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < mlp_points.size(); ++i) {
        glColor3f(colors[i][0], colors[i][1], colors[i][2]);
        glVertex2f(mlp_points[i][0] * 2 - 1, mlp_points[i][1] * 2 - 1);
    }
    glEnd();

    glFlush();
}

void trainMLP() {
    mlp.train(mlp_points, labels, 0.05, 10000, true);
}


void launchMLP(int argc, char** argv) {
    initMLP(); 
    trainMLP(); 

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("MLP Visualization");

    glClearColor(0.0, 0.0, 0.0, 1.0);
    gluOrtho2D(-1, 1, -1, 1);

    glutDisplayFunc(displayMLP);
    glutMainLoop();
}

