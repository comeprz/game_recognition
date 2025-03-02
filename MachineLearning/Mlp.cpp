#include "Mlp.h"
#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <GLUT/glut.h>
#include <fstream>
#include <string>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static std::vector<int> d;           
static int L = 0;                                 
static std::vector<std::vector<std::vector<double> > > W; // Poids
static std::vector<std::vector<double> > X;       
static std::vector<std::vector<double> > deltas; 

// Données du dataset
std::vector<std::vector<double> > mlp_points;       // Coordonnées 2D
std::vector<std::vector<double> > labels;           // Label pour chaque point


std::vector<std::vector<float> > colors;



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
    std::vector<double> output(d[L]);
    for (int j = 1; j <= d[L]; ++j) {
        output[j - 1] = X[L][j];
    }
    return output;
}


std::vector<double> trainMLP(const std::vector<std::vector<double> >& all_inputs,
                             const std::vector<std::vector<double> >& all_outputs,
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



std::vector<double> preprocessImage(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return {};
    }
    cv::resize(img, img, cv::Size(28, 28));
    img.convertTo(img, CV_64F, 1.0 / 255.0);

    std::vector<double> input;
    input.assign(img.begin<double>(), img.end<double>());
    return input;
}

std::vector<double> extractLabel(const std::string& filename) {    
    if (filename.find("lol") != std::string::npos) {
        return {1.0, 0.0, 0.0};
    } else if (filename.find("cod") != std::string::npos) {
        return {0.0, 1.0, 0.0};
    } else if (filename.find("starcraft") != std::string::npos) {
        return {0.0, 0.0, 1.0};
    }
    
    std::cerr << "Warning: Label not found for " << filename << std::endl;
    return {0.0, 0.0, 0.0};
}

void saveModel(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to save model" << std::endl;
        return;
    }
    file.write(reinterpret_cast<char*>(&L), sizeof(L));
    for (const auto& layer : W) {
        for (const auto& neuron : layer) {
            file.write(reinterpret_cast<const char*>(neuron.data()), neuron.size() * sizeof(double));
        }
    }
    file.close();
}

void loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to load model" << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(&L), sizeof(L));
    W.resize(L + 1);
    for (auto& layer : W) {
        for (auto& neuron : layer) {
            file.read(reinterpret_cast<char*>(neuron.data()), neuron.size() * sizeof(double));
        }
    }
    file.close();
}

void trainMLPWithImages(const std::vector<std::string>& imagePaths) {
    mlp_points.clear();
    labels.clear();

    for (const auto& path : imagePaths) {
        std::vector<double> input = preprocessImage(path);
        if (input.empty()) continue;
        
        std::vector<double> label = extractLabel(path);
        
        std::cout << "Image: " << path << " -> Label: ";
        for (double val : label) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        mlp_points.push_back(input);
        labels.push_back(label);
    }
    
    trainMLP(mlp_points, labels, 0.001, 1000000, true);
    saveModel("mlp_model.dat");
}

int testImage(const std::string& imagePath) {    
    std::vector<double> input = preprocessImage(imagePath);
    if (input.empty()) return 0;
    
    std::vector<double> output = predict(input, true);

    std::cout << "Raw prediction: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    int predicted_class = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::vector<std::string> categories = {"lol", "cod", "starcraft"};

    std::cout << "Predicted Class: " << categories[predicted_class] << std::endl;
    std::string actual_label = imagePath.substr(imagePath.find_last_of("/") + 1);
    int actual_class = -1;
    for (int i = 0; i < categories.size(); ++i) {
        if (actual_label.find(categories[i]) != std::string::npos) {
            actual_class = i;
            break;
        }
    }

    if (actual_class == predicted_class) {
        std::cout << "Prediction correct!" << std::endl;
        return 1;
    } else {
        std::cout << "Prediction incorrect!" << std::endl;
        return -1;
    }
}

std::vector<std::vector<float> > getColors(const std::vector<std::vector<double> >& labs)
{
    std::vector<std::vector<float> > c;
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

std::vector<std::string> getImagePaths(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    DIR* dir = opendir(folderPath.c_str());
    if (!dir) {
        std::cerr << "Error: Could not open directory " << folderPath << std::endl;
        return imagePaths;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename != "." && filename != "..") {
            imagePaths.push_back(folderPath + "/" + filename);
        }
    }
    closedir(dir);
    return imagePaths;
}

int main(int argc, char** argv) {
    int count = 0;
    int falsed = 0;
    int res = 0;
    std::vector<int> architecture = {2, 2, 3};
    initMLP(architecture);
    std::vector<std::string > imagePaths = getImagePaths("images");
    trainMLPWithImages(imagePaths);
    for (const auto& path : imagePaths) {
        std::cout << "Testing image: " << path << std::endl;
        res = testImage(path);
        if (res == 1) {
            count++;
        } else {
            falsed++;
        }
        res = 0;
        std::cout << "Count: " << count << std::endl;
        std::cout << "False: " << falsed << std::endl;
    }
    return 0;
}