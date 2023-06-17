#pragma once

#include "tuple.h"

#include <string>

class Texture
{
public:
	virtual tuple colorAt(const tuple& position, float u, float v)
	{
		return color(u, v, 0.0f);
	}
};

inline static auto createDefaultTexture()
{
	return std::make_shared<Texture>();
}

class ImageTexture : public Texture
{
public:
	const static int bytesPerPixel = 3;

	ImageTexture()
	: data(nullptr), width(0), height(0), bytesPerScanline(0)
	{}

	ImageTexture(const std::string& filename);

	~ImageTexture()
	{
		delete data;
	}

	tuple colorAt(const tuple& position, float u, float v) override
	{
		// If we have no texture data, then return solid cyan as a debugging aid.
		if (data == nullptr)
			return color(0.0f, 1.0f, 1.0f);

		// Clamp input texture coordinates to [0,1] x [1,0]
		u = Math::clamp(u, 0.0f, 1.0f);
		v = 1.0f - Math::clamp(v, 0.0f, 1.0f);  // Flip V to image coordinates

		auto i = static_cast<int>(u * width);
		auto j = static_cast<int>(v * height);

		// Clamp integer mapping, since actual coordinates should be less than 1.0
		if (i >= width)  i = width - 1;
		if (j >= height) j = height - 1;

		const auto colorScale = 1.0f / 255.0f;
		auto pixel = data + j * bytesPerScanline + i * bytesPerPixel;

		return color(colorScale * pixel[0], colorScale * pixel[1], colorScale * pixel[2]);
	}

private:
	uint8_t* data;
	int32_t width, height;
	int32_t bytesPerScanline;
};

inline static std::shared_ptr<Texture> createImageTexture(const std::string& filename)
{
	return std::make_shared<ImageTexture>(filename);
}