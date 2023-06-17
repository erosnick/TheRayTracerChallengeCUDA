#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>

ImageTexture::ImageTexture(const std::string& filename)
{
	auto componentsPerPixel = bytesPerPixel;

	data = stbi_load(filename.c_str(), &width, &height, &componentsPerPixel, componentsPerPixel);

	if (!data)
	{
		std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
		width = height = 0;
	}

	bytesPerScanline = bytesPerPixel * width;
}
