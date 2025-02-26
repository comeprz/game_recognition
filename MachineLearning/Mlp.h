#ifndef MYMLP_H
#define MYMLP_H

#include <vector>

class MyMLP {
public:
    explicit MyMLP(const std::vector<int>& npl);

    std::vector<double> predict(const std::vector<double>& inputs, bool is_classification);
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& expected_outputs,
               double alpha,
               int iteration_count,
               bool is_classification);

private:
    void propagate(std::vector<std::vector<double>>& X,
                   const std::vector<std::vector<std::vector<double>>>& W,
                   const std::vector<int>& d,
                   int L,
                   const std::vector<double>& inputs,
                   bool is_classification);

    std::vector<int> d;
    int L;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> deltas;
};

extern MyMLP mlp;

extern std::vector<std::vector<double>> mlp_points;
extern std::vector<std::vector<double>> labels;
extern std::vector<std::vector<float>> colors;

void initMLP();
void trainMLP();

void setMlpData(const std::vector<std::vector<double>>& newPoints,
                const std::vector<std::vector<double>>& newLabels);

void launchMLP(int argc, char** argv);

#endif 
