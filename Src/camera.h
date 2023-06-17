#pragma once

#include "matrix.h"
#include "ray.h"

struct Resolution
{
	int32_t width;
	int32_t height;
};

class Camera
{
public:
	Camera() 
	: Camera({ 800, 600 }, Math::radians(60.0f))
	{}

	Camera(int32_t imageWidth, int32_t imageHeight, float inFieldOfView, float inTime0 = 0.0f, float inTime1 = 0.5f)
	: Camera({ imageWidth, imageHeight }, inFieldOfView, inTime0, inTime1)
	{}

	Camera(const Resolution& resolution, float inFieldOfView, float inTime0 = 0.0f, float inTime1 = 0.5f)
	: imageWidth(resolution.width), 
	  imageHeight(resolution.height), 
	  fieldOfView(inFieldOfView),
	  pixelSize(0.0f),
	  time0(inTime0),
	  time1(inTime1)
	{
		auto halfView = std::tan(fieldOfView / 2.0f);
		aspect = static_cast<float>(imageWidth) / imageHeight;

		if (aspect >= 1.0f)
		{
			halfWidth = halfView;
			halfHeight = halfView / aspect;
		}
		else
		{
			halfWidth = halfView * aspect;
			halfHeight = halfView;
		}

		halfWidth *= focusDistance;
		halfHeight *= focusDistance;

		pixelSize = (halfWidth * 2.0f) / imageWidth;
	}

	Ray rayForPixel(float x, float y) const
	{
		// The offset from the edge of the canvas to the pixel's center
		auto xOffset = (x + 0.5f) * pixelSize;
		auto yOffset = (y + 0.5f) * pixelSize;

		// The untransformed coordinates of the pixel in world space
		// (remember that the camera looks toward -z, so +x is to the *left*)
		auto worldX = halfWidth - xOffset;
		auto worldY = halfHeight - yOffset;

		// Using the camera matrix, transform the canvas point and the origin,
		// then compute the ray's direction vector.
		// (remember that the canvas is at z = -1)
		// transform: World space to camera space
		// inverse(transform): Camera space to world space
		// Optimization: Using cached inversed view transform
		// TODO Update inversed view transform when view transform changed
		auto pixel = inversedTransform * point(worldX, worldY, -focusDistance);

		tuple rd = randomInUnitDisk() * lensRadius;

		auto origin = inversedTransform * point(rd.x, rd.y, 0.0f);

		auto direction = normalize(pixel - origin);

		return { origin, direction, Math::randomFloat(time0, time1) };
	}

	int32_t imageWidth;
	int32_t imageHeight;
	float halfWidth = 0.0f;
	float halfHeight = 0.0f;
	float fieldOfView = 0.0f;
	matrix4 transform = matrix4(1.0f);
	matrix4 inversedTransform = matrix4(1.0f);
	float pixelSize = 0.0f;
	float time0 = 0.0f;
	float time1 = 0.5f;
	//float aperture = 0.05f;
	//float focusDistance = 2.0f;
	//float lensRadius = aperture * 0.5f;
	float aperture = 0.0f;
	float focusDistance = 1.0f;
	float lensRadius = aperture * 0.5f;
	float aspect = 0.0f;
};