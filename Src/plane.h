#pragma once

#include "shape.h"

class Plane : public Shape
{

public:
	Plane(float inExtentX = std::numeric_limits<float>::max(),
		  float inExtentZ = std::numeric_limits<float>::max())
		: extentX(inExtentX), extentZ(inExtentZ)
	{}
	
	virtual ~Plane() = default;

	std::vector<Intersection> localIntersect(const Ray& transformedRay) override
	{
		if (std::fabsf(transformedRay.direction.y) < EPSILON)
		{
			return {};
		}

		auto t = -transformedRay.origin.y / transformedRay.direction.y;

		auto position = transformedRay.at(t);

		if ((position.x > extentX || position.x < -extentX) ||
			(position.z > extentZ || position.z < -extentZ))
		{
			return {};
		}

		auto shape = std::dynamic_pointer_cast<Plane>(shared_from_this());
		
		return { { t, shape }};
	}

	tuple localNormalAt(const tuple& localPosition, const Intersection intersection = {}) const override
	{
		return vector(0.0f, 1.0f, 0.0f);
	}

	virtual bool boundingBox(BoundingBox& outputBox) override
	{
		return true;
	}

	float extentX = std::numeric_limits<float>::max();
	float extentZ = std::numeric_limits<float>::max();
};

inline static std::shared_ptr<Shape> createPlane(float extendX = std::numeric_limits<float>::max(),
												 float extendZ = std::numeric_limits<float>::max())
{
	return std::make_shared<Plane>(extendX, extendZ);
}