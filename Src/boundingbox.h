#pragma once

#include "tuple.h"

class BoundingBox
{
public:
	BoundingBox() = default;
	BoundingBox(const tuple& inMin, const tuple& inMax)
	: min(inMin), max(inMax)
	{}

	void addPoint(const tuple& point)
	{
		// Adding "point" to "box"
		if (point.x < min.x) min.x = point.x;
		if (point.y < min.y) min.y = point.y;
		if (point.z < min.z) min.z = point.z;

		if (point.x > max.x) max.x = point.x;
		if (point.y > max.y) max.y = point.y;
		if (point.z > max.z) max.z = point.z;
	}

	inline bool hit(const Ray& ray, float inTMin = EPSILON, float inTMax = INFINITY) const
	{
		for (int a = 0; a < 3; a++)
		{
			auto invD = 1.0f / ray.direction[a];
			auto t0 = (min[a] - ray.origin[a]) * invD;
			auto t1 = (max[a] - ray.origin[a]) * invD;

			if (invD < 0.0f)
			{
				std::swap(t0, t1);
			}

			auto tMin = t0 > inTMin ? t0 : inTMin;
			auto tMax = t1 < inTMax ? t1 : inTMax;

			if (tMax <= tMin)
			{
				return false;
			}
		}
		return true;
	}

	tuple min = point(std::numeric_limits<float>::max());
	tuple max = point(-std::numeric_limits<float>::max());
};

inline static BoundingBox surroundingBox(const BoundingBox& box0, const BoundingBox& box1)
{
	auto minX = std::min(box0.min.x, box1.min.x);
	auto minY = std::min(box0.min.y, box1.min.y);
	auto minZ = std::min(box0.min.z, box1.min.z);

	tuple min = point(minX, minY, minZ);

	auto maxX = std::max(box0.max.x, box1.max.x);
	auto maxY = std::max(box0.max.y, box1.max.y);
	auto maxZ = std::max(box0.max.z, box1.max.z);

	tuple max = point(maxX, maxY, maxZ);

	return BoundingBox(min, max);
}