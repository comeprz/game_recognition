#include <GL/freeglut.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

struct PointR {
    float x, y;   
    float r, g, b;
    float label; 
};

std::vector<PointR> PointRs;

int K = 3;


std::vector<std::pair<float,float>> centersRBF;
std::vector<float> sigmaRBF;
std::vector<float> WR;  
float lambdaR = 0.0f;  

float distance2D(float x1, float y1, float x2, float y2)
{
    float dx = x1 - x2;
    float dy = y1 - y2;
    return std::sqrt(dx*dx + dy*dy);
}

void kmeans(int k, int maxIter=100)
{
    centersRBF.clear();
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, (int)PointRs.size()-1);

    for(int i = 0; i < k; i++){
        int idx = dist(rng);
        centersRBF.push_back({PointRs[idx].x, PointRs[idx].y});
    }

    for(int iter = 0; iter < maxIter; iter++){
        std::vector<std::vector<PointR>> clusters(k);

        for(const auto& p : PointRs){
            float bestDist = std::numeric_limits<float>::max();
            int bestCenter = 0;
            for(int c = 0; c < k; c++){
                float d = distance2D(p.x, p.y, centersRBF[c].first, centersRBF[c].second);
                if(d < bestDist){
                    bestDist = d;
                    bestCenter = c;
                }
            }
            clusters[bestCenter].push_back(p);
        }
        for(int c = 0; c < k; c++){
            if(!clusters[c].empty()){
                float meanX = 0.0f, meanY = 0.0f;
                for(const auto& pt : clusters[c]){
                    meanX += pt.x;
                    meanY += pt.y;
                }
                meanX /= clusters[c].size();
                meanY /= clusters[c].size();
                centersRBF[c] = {meanX, meanY};
            }
        }
    }
}

void computeSigmas(int k)
{
    sigmaRBF.resize(k, 0.0f);

    std::vector<std::vector<PointR>> clusters(k);
    for(const auto& p : PointRs){
        float bestDist = std::numeric_limits<float>::max();
        int bestCenter = 0;
        for(int c=0; c < k; c++){
            float d = distance2D(p.x, p.y, centersRBF[c].first, centersRBF[c].second);
            if(d < bestDist){
                bestDist = d;
                bestCenter = c;
            }
        }
        clusters[bestCenter].push_back(p);
    }

    for(int c=0; c < k; c++){
        if(!clusters[c].empty()){
            float sum = 0.0f;
            for(const auto& pt : clusters[c]){
                sum += distance2D(pt.x, pt.y, centersRBF[c].first, centersRBF[c].second);
            }
            sigmaRBF[c] = sum / clusters[c].size();
            if(sigmaRBF[c] < 1e-6f) sigmaRBF[c] = 0.1f; 
        }
        else {
            sigmaRBF[c] = 0.1f; 
        }
    }
}

float gaussianRBF(int c, float x, float y)
{
    float cx = centersRBF[c].first;
    float cy = centersRBF[c].second;
    float dist = distance2D(x, y, cx, cy);
    float sigma = sigmaRBF[c];
    if(sigma < 1e-7f) sigma = 0.1f;

    return std::exp( - (dist*dist) / (2.0f * sigma*sigma) );
}

void trainRBFN_leastSquares(int k, float lambda)
{
    int N = (int)PointRs.size();
    if(N == 0) return;

    std::vector<float> Phi(N*k, 0.0f);

    for(int i=0; i < N; i++){
        for(int j=0; j < k; j++){
            Phi[i*k + j] = gaussianRBF(j, PointRs[i].x, PointRs[i].y);
        }
    }

    std::vector<float> Y(N);
    for(int i=0; i < N; i++){
        Y[i] = PointRs[i].label; 
    }

    std::vector<float> A(k*k, 0.0f);
    for(int i=0; i<N; i++){
        for(int c1=0; c1<k; c1++){
            float v1 = Phi[i*k + c1];
            for(int c2=0; c2<k; c2++){
                float v2 = Phi[i*k + c2];
                A[c1*k + c2] += v1 * v2;
            }
        }
    }

    for(int diag=0; diag<k; diag++){
        A[diag*k + diag] += lambda;
    }

    std::vector<float> b(k, 0.0f);
    for(int i=0; i<N; i++){
        for(int c1=0; c1<k; c1++){
            b[c1] += Phi[i*k + c1] * Y[i];
        }
    }

    WR.resize(k);
    
    std::vector<float> Awork = A;
    std::vector<float> Bwork = b;

    for(int c=0; c<k; c++){
        float pivot = Awork[c*k + c];
        if(std::fabs(pivot) < 1e-12){
            std::cerr << "Pivot nul!\n";
            pivot = 1e-12;
        }
        for(int col=c; col<k; col++){
            Awork[c*k + col] /= pivot;
        }
        Bwork[c] /= pivot;

        for(int l=0; l<k; l++){
            if(l != c){
                float alpha = Awork[l*k + c];
                for(int col=c; col<k; col++){
                    Awork[l*k + col] -= alpha * Awork[c*k + col];
                }
                Bwork[l] -= alpha * Bwork[c];
            }
        }
    }

    for(int c=0; c<k; c++){
        WR[c] = Bwork[c];
    }
}

float predictRBFN(float x, float y)
{
    float sum = 0.0f;
    for(int j=0; j<(int)WR.size(); j++){
        sum += WR[j] * gaussianRBF(j, x, y);
    }
    return (sum >= 0.0f) ? 1.0f : -1.0f;
}


void displaySeparationR()
{
    int nx = 100, ny = 100;
    float step = 4.0f / nx; 

    glBegin(GL_POINTS);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            float x = i * step;
            float y = j * step;

            float pred = predictRBFN(x, y);
            if (pred >= 0) 
                glColor3f(0.68f, 0.85f, 0.90f);
            else 
                glColor3f(1.0f, 0.75f, 0.80f); 

            glVertex2f(x, y);
        }
    }
    glEnd();
}

void displayR()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(4.0f);

    displaySeparationR();

    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for (const auto& p : PointRs) {
        glColor3f(p.r, p.g, p.b);
        glVertex2f(p.x, p.y);
    }
    glEnd();

    glFlush();
}

void trainRBFN(int k, float lambda)
{
    if(PointRs.empty()) return;

    kmeans(k, 50);

    computeSigmas(k);

    trainRBFN_leastSquares(k, lambda);
}


void setDataset1R()
{
    PointRs = {
        {1.0f,1.0f, 0.0f,0.0f,1.0f,  1.0f},  
        {1.0f,0.0f, 1.0f,0.0f,0.0f, -1.0f},  
        {0.0f,1.0f, 1.0f,0.0f,0.0f, -1.0f}   
    };

    trainRBFN(K, lambdaR);
}

void setDataset2R()
{
    PointRs = {
        {1.0f,1.0f, 0.0f,0.0f,1.0f, 1.0f},
        {2.0f,3.0f, 1.0f,0.0f,0.0f,-1.0f},
        {3.0f,3.0f, 1.0f,0.0f,0.0f,-1.0f}
    };

    trainRBFN(K, lambdaR);
}

void setDataset3R()
{
    PointRs.clear();
    srand((unsigned int)time(nullptr));

    for (int i = 0; i < 50; ++i) {
        float x = ((float)rand()/RAND_MAX)*0.9f + 1.0f;
        float y = ((float)rand()/RAND_MAX)*0.9f + 1.0f;
        PointRs.push_back({x,y, 0.0f,0.0f,1.0f, 1.0f});
    }
    for (int i = 0; i < 50; ++i) {
        float x = ((float)rand()/RAND_MAX)*0.9f + 2.0f;
        float y = ((float)rand()/RAND_MAX)*0.9f + 2.0f;
        PointRs.push_back({x,y, 1.0f,0.0f,0.0f,-1.0f});
    }

    trainRBFN(K, lambdaR);
}

void setDataset4R()
{
    PointRs = {
        {1.0f, 0.0f, 0.0f,0.0f,1.0f, 1.0f},
        {0.0f, 1.0f, 0.0f,0.0f,1.0f, 1.0f},
        {0.0f, 0.0f, 1.0f,0.0f,0.0f,-1.0f},
        {1.0f, 1.0f, 1.0f,0.0f,0.0f,-1.0f}
    };

    trainRBFN(K, lambdaR);
}

void setDataset5R()
{
    PointRs.clear();
    srand((unsigned int)time(nullptr));

    for (int i = 0; i < 500; ++i) {
        float x = ((float)rand()/RAND_MAX)*4.0f;
        float y = ((float)rand()/RAND_MAX)*4.0f;
        float label = (std::fabs(x-2.0f) <= 0.6f || std::fabs(y-2.0f) <= 0.6f) ? 1.0f : -1.0f;

        float rr = (label==1.0f) ? 0.0f : 1.0f;
        float gg = 0.0f;
        float bb = (label==1.0f) ? 1.0f : 0.0f;

        PointRs.push_back({x,y, rr,gg,bb, label});
    }

    trainRBFN(K, lambdaR);
}


void keyboardR(unsigned char key, int x, int y)
{
    switch (key) {
    case '1': setDataset1R(); break;
    case '2': setDataset2R(); break;
    case '3': setDataset3R(); break;
    case '4': setDataset4R(); break;
    case '5': setDataset5R(); break;
    case 'q': case 'Q':
        exit(0);
        break;
    default:
        std::cout << "Choix invalide\n";
    }
    glutPostRedisplay();
}


void initR()
{
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-0.2,4.0,-0.2,4.0);
}

int launchRBFN(int argc, char** argv)
{

    K = 4;        
    lambdaR = 0.01f; 

  
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(600,600);
    glutCreateWindow("RBFN");

    initR();


    glutDisplayFunc(displayR);
    glutKeyboardFunc(keyboardR);

    glutMainLoop();
    return 0;
}
