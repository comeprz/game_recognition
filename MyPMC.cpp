#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <filesystem>

class MyMLP {
public:
    explicit MyMLP(const std::vector<int>& npl) {
        d = npl;
        L = d.size() - 1;

        W.resize(L + 1);
        for (int l = 1; l <= L; ++l) {
            W[l].resize(d[l - 1] + 1);
            for (int i = 0; i <= d[l - 1]; ++i) {
                W[l][i].resize(d[l] + 1);
                for (int j = 1; j <= d[l]; ++j) {
                    if (j == 0) {
                        W[l][i][j] = 0.0;
                    } else {
                        W[l][i][j] = randomFloat(-1.0, 1.0);
                    }
                }
            }
        }

        X.resize(L + 1);
        deltas.resize(L + 1);
        for (int l = 0; l <= L; ++l) {
            X[l].resize(d[l] + 1, 0.0);
            deltas[l].resize(d[l] + 1, 0.0);
            if (l == 0)
                X[l][0] = 1.0;
            else
                X[l][0] = 0.0;
        }
    }

    void _propagate(const std::vector<double>& inputs, bool is_classification) {
        for (int j = 1; j <= d[0]; ++j) {
            X[0][j] = inputs[j - 1];
        }

        for (int l = 1; l <= L; ++l) {
            for (int j = 1; j <= d[l]; ++j) {
                double total = 0.0;
                for (int i = 0; i <= d[l - 1]; ++i) {
                    total += W[l][i][j] * X[l - 1][i];
                }

                X[l][j] = total;
                if (is_classification || l != L) {
                    X[l][j] = tanh(total);
                }
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& inputs, bool is_classification) {
        _propagate(inputs, is_classification);
        return {X[L].begin() + 1, X[L].end()};
    }

    std::vector<double> train(const std::vector<std::vector<double>>& all_dataset_inputs,
                              const std::vector<std::vector<double>>& all_dataset_expected_outputs,
                              double alpha, int iteration_count, bool is_classification) {
        std::vector<double> losses;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            int k = rand() % all_dataset_inputs.size();
            const std::vector<double>& Xk = all_dataset_inputs[k];
            const std::vector<double>& Yk = all_dataset_expected_outputs[k];

            double loss = 0.0;
            _propagate(Xk, is_classification);

            for (int j = 1; j <= d[L]; ++j) {
                loss += (X[L][j] - Yk[j - 1]) * (X[L][j] - Yk[j - 1]);
                deltas[L][j] = X[L][j] - Yk[j - 1];
                if (is_classification) {
                    deltas[L][j] *= (1.0 - tanh(X[L][j]) * tanh(X[L][j]));
                }
            }
            loss /= d[L];
            losses.push_back(loss);

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
        return losses;
    }

    void print() const {
        std::cout << "W: ";
        for (int l = 1; l <= L; ++l) {
            for (int i = 0; i <= d[l - 1]; ++i) {
                for (int j = 1; j <= d[l]; ++j) {
                    std::cout << W[l][i][j] << " ";
                }
            }
        }
        std::cout << std::endl;
        std::cout << "X: ";
        for (int l = 0; l <= L; ++l) {
            for (int j = 0; j <= d[l]; ++j) {
                std::cout << X[l][j] << " ";
            }
        }
        std::cout << std::endl;
        std::cout << "deltas: ";
        for (int l = 0; l <= L; ++l) {
            for (int j = 0; j <= d[l]; ++j) {
                std::cout << deltas[l][j] << " ";
            }
        }
    }

private:
    std::vector<int> d;
    int L;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> deltas;

    static double randomFloat(double min, double max) {
        static std::default_random_engine engine(std::time(nullptr));
        std::uniform_real_distribution dist(min, max);
        return dist(engine);
    }
};

int main() {
    std::vector<int> npl = {2, 3, 1};
    MyMLP mlp(npl);

    mlp.print();

    std::vector<std::vector<double>> inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> expected_outputs = {{-1.0}, {1.0}, {1.0}, {-1.0}};
    std::vector<double> losses = mlp.train(inputs, expected_outputs, 0.05, 10000, true);
    std::vector<double> prediction = mlp.predict({0.0, 1.0}, true);

    std::ofstream loss_file("losses.txt");
    if (loss_file.is_open()) {
        for (double loss : losses) {
            loss_file << loss << "," << "\n";
        }
        loss_file.close();
        std::cout << "Losses saved to losses.txt" << std::endl;
    } else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }
    std::cout << "Losses saved to: " << std::filesystem::current_path() / "losses.txt" << std::endl;
    return 0;
}
