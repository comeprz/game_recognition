#include <vector>
#include <GL/glut.h>
#include <iostream>

struct Point
{
	float x, y;
	float r, g, b;
};

std::vector<Point> points = {
	{1.0f, 1.0f, 0.0f, 0.0f, 1.0f}, //point bleu
	{1.0f,0.0f, 1.0f, 0.0f,0.0f}, //point rouge
	{0.0f, 1.0f, 1.0f,0.0f, 0.0f} //point rouge
};

void drawAxes()
{
	glColor3f(0.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);

	glVertex2f(-0.2f, -0.1f);
	glVertex2f(1.2f, -0.1f);

	glVertex2f(-0.1f, -0.2f);
	glVertex2f(-0.1f, 1.2f);
	glEnd();

	glRasterPos2f(-0.15f, -0.05f);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, '0');

	glRasterPos2f(1.0f, -0.15f);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, '1');

	glRasterPos2f(-0.15f, 1.0f);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, '1');
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glPointSize(10.0f);

	drawAxes();

	glBegin(GL_POINTS);
	for (const auto& p : points)
	{
		glColor3f(p.r, p.g, p.b);
		glVertex2d(p.x, p.y);
	}
	glEnd();
	glFlush();
}


void init()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-0.2, 1.2, -0.2, 1.2);
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutCreateWindow("OpenGl Scatter Plot");
	init();
	glutDisplayFunc(display);
	glutMainLoop();
	return 0;

}