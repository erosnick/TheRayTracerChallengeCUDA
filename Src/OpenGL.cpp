#include "OpenGL.h"

#include <iostream>

constexpr float Aspect = 16.0f / 9.0f;

int32_t WindowWidth = 1280;
int32_t WindowHeight = static_cast<int32_t>(WindowWidth / Aspect);

GLFWwindow* window = nullptr;

int32_t frameCount = 0;

GLuint texture = 0;

bool bRightMouseButtonDown = false;
bool bMiddleMouseButtonDown = false;

float frameTime = 0.016667f;

float lastMousePositionX = 0.0f;
float lastMousePositionY = 0.0f;

void onKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void onFrameBufferResize(GLFWwindow* inWindow, int width, int height);
void onMouseButtonCallback(GLFWwindow* inWindow, int button, int action, int mods);
void onScrollCallback(GLFWwindow* inWindow, double xOffset, double yOffset);
void onMouseMoveCallback(GLFWwindow* inWindow, double x, double y);

void onKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	//ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_RELEASE)
	{

	}

	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_RELEASE)
	{

	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}
}

void onFrameBufferResize(GLFWwindow* inWindow, int width, int height)
{
	if (width > 0 && height > 0)
	{
		WindowWidth = width;
		WindowHeight = height;
		glViewport(0, 0, width, height);
	}
}

void onMouseButtonCallback(GLFWwindow* inWindow, int button, int action, int mods)
{

	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		bRightMouseButtonDown = true;
		double x;
		double y;
		glfwGetCursorPos(inWindow, &x, &y);
		lastMousePositionX = static_cast<float>(x);
		lastMousePositionY = static_cast<float>(y);
	}

	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		bRightMouseButtonDown = false;
	}

	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
		bMiddleMouseButtonDown = true;
	}

	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE) {
		bMiddleMouseButtonDown = false;
	}

	//ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void onScrollCallback(GLFWwindow* inWindow, double xOffset, double yOffset)
{

	//ImGui_ImplGlfw_ScrollCallback(inWindow, xOffset, yOffset);

	//if (ImGui::GetIO().WantCaptureMouse) {
	//    return;
	//}
}

void onMouseMoveCallback(GLFWwindow* inWindow, double x, double y)
{
	float dx = (lastMousePositionX - static_cast<float>(x)) * frameTime;
	float dy = (lastMousePositionY - static_cast<float>(y)) * frameTime;

	if (bRightMouseButtonDown)
	{
	}

	if (bMiddleMouseButtonDown)
	{
	}

	lastMousePositionX = static_cast<float>(x);
	lastMousePositionY = static_cast<float>(y);
}

void bindCallbacks()
{
	glfwSetFramebufferSizeCallback(window, onFrameBufferResize);
	glfwSetKeyCallback(window, onKeyCallback);
	glfwSetMouseButtonCallback(window, onMouseButtonCallback);
	glfwSetScrollCallback(window, onScrollCallback);
	glfwSetCursorPosCallback(window, onMouseMoveCallback);
}

void createOpenGLTexture()
{
	// Generate OpenGL texture from CUDA array
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WindowWidth, WindowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
}

bool initGLFW()
{
	// Initialize GLFW
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return false;
	}

	// Create a GLFW window
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(WindowWidth, WindowHeight, "CUDA-OpenGL Interoperability", nullptr, nullptr);

	if (!window)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return false;
	}

	const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	glfwSetWindowPos(window, (mode->width - WindowHeight) / 2, (mode->height - WindowHeight) / 2);

	bindCallbacks();

	// Make the OpenGL context current
	glfwMakeContextCurrent(window);

	glfwSwapInterval(0);

	// Load OpenGL function pointers with GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize GLAD" << std::endl;
		glfwTerminate();
		return false;
	}

	createOpenGLTexture();

	return true;
}

void beginFrame()
{

}

void endFrame()
{

}

// Shader sources
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main()
    {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D texture1;
    void main()
    {
        FragColor = texture(texture1, TexCoord);
    }
)";

// Compile and link shader program
GLuint createShaderProgram()
{
	// Create vertex shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	// Check vertex shader compile errors
	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
	}

	// Create fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	// Check fragment shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
	}

	// Create shader program
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// Check shader program linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cerr << "Shader program linking failed: " << infoLog << std::endl;
	}

	// Delete the shaders as they're linked into the program now and no longer needed
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

GLuint VAO = 0;
GLuint VBO = 0;
GLuint EBO = 0;
GLuint shaderProgram = 0;

void renderQuad()
{
	if (VAO == 0)
	{
		// Set up vertex data and attribute pointers
		GLfloat vertices[] =
		{
			-1.0f, -1.0f, 0.0f, 0.0f,
			 1.0f, -1.0f, 1.0f, 0.0f,
			 1.0f,  1.0f, 1.0f, 1.0f,
			-1.0f,  1.0f, 0.0f, 1.0f
		};

		GLuint indices[] =
		{
			0, 1, 2,
			2, 3, 0
		};

		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
		glBindVertexArray(0);
		
		shaderProgram = createShaderProgram();
	}

	// Bind the shader program
	glUseProgram(shaderProgram);

	// Set the texture uniform
	glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

	// Bind the texture
	glBindTexture(GL_TEXTURE_2D, texture);

	// Render the quad
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void updateFPSCounter(GLFWwindow* window)
{
	static double previousSeconds = glfwGetTime();
	double currentSeconds = glfwGetTime();
	double elapsedSeconds = currentSeconds - previousSeconds;

	if (elapsedSeconds >= 0.25f)
	{
		previousSeconds = currentSeconds;
		double fps = (double)frameCount / elapsedSeconds;
		char temp[32];
		sprintf_s(temp, "FPS:%.1f", fps);
		glfwSetWindowTitle(window, temp);
		frameCount = 0;
	}

	frameCount++;
}

void run()
{
	// Render loop
	while (!glfwWindowShouldClose(window))
	{
		// Poll events
		glfwPollEvents();

		updateFPSCounter(window);

		// Clear the color buffer
		glClear(GL_COLOR_BUFFER_BIT);

		renderQuad();

		// Swap buffers
		glfwSwapBuffers(window);
	}

	// Clean up resources
	glDeleteTextures(1, &texture);
	glDeleteBuffers(1, &EBO);
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);

	glfwTerminate();
}
