#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

extern GLuint texture;

extern GLFWwindow* window;

extern int32_t WindowWidth;
extern int32_t WindowHeight;

bool initGLFW();

GLuint createShaderProgram();

void renderQuad();

void run();

void beginFrame();

void endFrame();