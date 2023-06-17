#include "render.cuh"
#include "OpenGL.h"

#include <cstdint>

int main(int32_t argc, char* argv[])
{
	if (initGLFW())
	{
		createCUDAResources();

		run();
	}

	releaseCUDAResources();

	return 0;
}