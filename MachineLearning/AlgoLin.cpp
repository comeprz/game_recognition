#include "AlgoLin.h"
#include <GL/glut.h>
#include <iostream>
#include <cstdlib>
#include <ctime>


std::vector<float> W;
float alpha = 0.1f;
int iteration_count = 200;

std::vector<Point> points;

void initializeWeights() {
	srand(time(0));
	W.resize(3);
	for (auto& w : W) {
		w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Valeurs entre -1 et 1
	}
	//W = { 0.5f, 1.0f, -1.0f };
	std::cout << "Initial Weights: " << W[0] << " " << W[1] << " " << W[2] << std::endl;
}


float predict(const std::vector<float>& W, float x1, float x2) {
	float signal = W[1] * x1 + W[2] * x2 + W[0];
	return signal >= 0 ? 1.0f : -1.0f;
}

void drawAxes()
{
	glColor3f(0.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);

	glVertex2f(-0.2f, -0.1f);
	glVertex2f(1.2f, -0.1f);

	glVertex2f(-0.1f, -0.2f);
	glVertex2f(-0.1f, 1.2f);
	glEnd();

	//glRasterPos2f(-0.15f, -0.05f);
	//glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, '0');

	//glRasterPos2f(1.0f, -0.15f);
	//glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, '1');

	//glRasterPos2f(-0.15f, 1.0f);
	//glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, '1');
}

void displaySeparation() {
	int nx = 100, ny = 100;
	float step = 4.0f / nx;

	//std::cout << "Weights: " << W[0] << " " << W[1] << " " << W[2] << std::endl;

	//for (int i = 0; i < nx; ++i) {
	//	for (int j = 0; j < ny; ++j) {
	//		float x1 = i * step;
	//		float x2 = j * step;
	//		float pred = predict(W, x1, x2);
	//		std::cout << "Point: (" << x1 << ", " << x2 << ") -> Prediction: " << pred << std::endl;
	//	}
	//}

	glBegin(GL_POINTS);
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			float x1 = i * step;
			float x2 = j * step;
			float pred = predict(W, x1, x2);

			if (pred >= 0) glColor3f(0.68f, 0.85f, 0.90f);
			else glColor3f(1.0f, 0.75f, 0.80f);

			glVertex2f(x1, x2);
		}
	}
	glEnd();
}

void train(int iteration_count, float alpha) {
	for (int i = 0; i < iteration_count; ++i) {
		int k = rand() % points.size();
		Point Xk = points[k];
		float Yk = Xk.label;// Bleu = 1, Rouge = -1
		float gXk = predict(W, Xk.x, Xk.y);

		W[0] += alpha * (Yk - gXk) * 1.0f;
		W[1] += alpha * (Yk - gXk) * Xk.x;
		W[2] += alpha * (Yk - gXk) * Xk.y;

		displaySeparation();
	}
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glPointSize(10.0f);

	displaySeparation();

	glPointSize(10.0f);
	glBegin(GL_POINTS);
	for (const auto& p : points)
	{
		glColor3f(p.r, p.g, p.b);
		glVertex2d(p.x, p.y);
	}
	glEnd();
	glFlush();

}

void setDataset1() {
	points = {
		{1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f}, 
		{1.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f}, 
		{0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f} 
	};
	train(iteration_count, alpha);


}

void setDataset2() {
	points = {
		{1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f}, 
		{2.0f, 3.0f, 1.0f, 0.0f, 0.0f, -1.0f}, 
		{3.0f, 3.0f, 1.0f, 0.0f, 0.0f, -1.0f}  
	};
	train(iteration_count, alpha);
}

void setDataset3() {
	points.clear();
	for (int i = 0; i < 50; ++i) {
		float x = static_cast<float>(rand()) / RAND_MAX * 0.9f + 1.0f;
		float y = static_cast<float>(rand()) / RAND_MAX * 0.9f + 1.0f;
		points.push_back({ x, y, 0.0f, 0.0f, 1.0f, 1.0f });
	}
	for (int i = 0; i < 50; ++i) {
		float x = static_cast<float>(rand()) / RAND_MAX * 0.9f + 2.0f;
		float y = static_cast<float>(rand()) / RAND_MAX * 0.9f + 2.0f;
		points.push_back({ x, y, 1.0f, 0.0f, 0.0f, -1.0f });
	}
	train(iteration_count, alpha);

}

void setDataset4() {
	points = {
		{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f},   
		{0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},   
		{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f}, 
		{1.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f}   
	};
	train(iteration_count, alpha);
}

void setDataset5() {
	points.clear();
	srand(time(0)); 

	for (int i = 0; i < 500; ++i) {
		float x = static_cast<float>(rand()) / RAND_MAX * 4.0f;
		float y = static_cast<float>(rand()) / RAND_MAX * 4.0f; 
		float label = (abs(x - 2.0f) <= 0.6f || abs(y - 2.0f) <= 0.6f) ? 1.0f : -1.0f;

		float r = (label == 1.0f) ? 0.0f : 1.0f; 
		float g = 0.0f;
		float b = (label == 1.0f) ? 1.0f : 0.0f;

		points.push_back({ x, y, r, g, b, label });
	}

	train(iteration_count, alpha); 
}


void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case '1':
		setDataset1();
		break;
	case '2':
		setDataset2();
		break;
	case '3':
		setDataset3();
		break;
	case '4':
		setDataset4();
		break;
	case '5':
		setDataset5();
		break;
	case 'q':  
		exit(0);
		break;
	default:
		std::cout << "Choix invalide\n";
	}
	glutPostRedisplay();
}

void init() {
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-0.2, 4.0, -0.2, 4.0);
	initializeWeights(); 
}

/*int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutCreateWindow("OpenGL Scatter Plot");

	init();

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard); // Gestion du choix avec le clavier

	glutMainLoop(); // Boucle principale OpenGL (ne retourne jamais)
	return 0;
}*/

void launchAlgoLin(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutCreateWindow("Perceptron Linear");

	init();

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	glutMainLoop(); 
}