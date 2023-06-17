#pragma once

#include "ray.h"
#include "matrix.h"
#include "material.h"
#include "intersection.h"
#include "boundingbox.h"

enum class ShapeType : uint8_t
{
	Sphere,
	Plane,
	Cube,
	Cylinder,
	Cone
};

class Shape : public std::enable_shared_from_this<Shape>
{
public:
	virtual ~Shape() = default;

	void setScale(const tuple& inScale)
	{
		scale = inScale;
	}

	void setScale(float x, float y, float z)
	{
		setScale(point(x, y, z));
	}

	void setScale(float value)
	{
		setScale(point(value, value, value));
	}

	void setRotation(float pitch, float yaw, float roll)
	{
		setRotation(point(pitch, yaw, roll));
	}

	void setRotation(const tuple& inRotation)
	{
		rotation = inRotation;
	}

	void setTranslation(float x, float y, float z)
	{
		translation = point(x, y, z);
	}

	void setTranslation(const tuple& inTranslation)
	{
		translation = inTranslation;
	}

	virtual void setTransform(const matrix4& inTransform)
	{
		transform = inTransform;

		// Optimization: Cache inversed matrix
		inversedTransform = inverse(transform);
	}

	virtual std::vector<Intersection> intersect(const Ray& ray)
	{
		auto localRay = transformRay(ray, inversedTransform);
		return localIntersect(localRay);
	}

	virtual std::vector<Intersection> localIntersect(const Ray& transformedRay) { return {}; }

	tuple normalAt(const tuple& worldPosition, const Intersection intersection = {})
	{ 
		// Before Chapter 14 Groups
		//auto localPosition = inverse(transform) * worldPosition;
		//auto localNormal = localNormalAt(localPosition);
		//auto worldNormal = transpose(inverse(transform)) * localNormal;
		//worldNormal.w = 0.0f;

		//return normalize(worldNormal);
		// 
		// Chapter 14 Groups
		auto localPosition = worldToObject(worldPosition);
		auto localNormal = localNormalAt(localPosition, intersection);

		if (material.sinNormalPerturb)
		{
			material.sinNormalPerturb(localNormal, localPosition, 0.1f, 50.0f, 0.0f);
		}

		if (material.cosNormalPerturb)
		{
			material.cosNormalPerturb(localNormal, localPosition, 0.1f, 50.0f, RTC_PIDIV4);
		}

		if (material.noiseNormalPerturb)
		{
			material.noiseNormalPerturb(localNormal, localPosition, 0.1, 8, 1.0, 5.0);
		}

		return normalToWorld(localNormal);
	}

	virtual tuple localNormalAt(const tuple& localPosition, const Intersection intersection = {}) const 
	{
		return {}; 
	}

	virtual bool boundingBox(BoundingBox& outputBox) = 0;

	virtual bool contains(const std::shared_ptr<Shape>& shape)
	{
		// Chapter 16 Constructive Solid Geometry (CSG)
		// If A is any other shape, the includes operator should return true if A is equal to B.
		return (this == shape.get());
	}

	tuple worldToObject(const tuple& worldPosition)
	{
		auto localPosition = worldPosition;

		if (parent != nullptr)
		{
			localPosition = parent->worldToObject(localPosition);
		}

		// Optimization: Using cached inversed transform
		return inversedTransform * localPosition;
	}

	tuple normalToWorld(const tuple& localNormal)
	{
		// Optimization: Using cached inversed transform
		auto worldNormal = transpose(inversedTransform) * localNormal;
		worldNormal.w = 0.0f;
		worldNormal = normalize(worldNormal);

		// Group and CSG
		if (parent != nullptr)
		{
			worldNormal = parent->normalToWorld(worldNormal);
		}

		return worldNormal;
	}

	auto getMaterial() const
	{
		if (parent != nullptr && useParentMaterial)
		{
			return parent->material;
		}
		return material;
	}

	matrix4 transform;
	matrix4 inversedTransform;
	tuple scale{ 1.0f, 1.0f, 1.0f, 1.0f };
	tuple rotation{ 0.0f, 0.0f, 0.0f, 1.0f };
	tuple translation{ 0.0f, 0.0f, 0.0f, 1.0f };
	Material material;
	BoundingBox aabb;

	std::shared_ptr<Shape> parent;

protected:
	bool useParentMaterial = false;
};

class TestShape : public Shape
{
public:
	virtual std::vector<Intersection> localIntersect(const Ray& transformedRay) override
	{
		savedRay = transformedRay;
		return {}; 
	}

	virtual tuple localNormalAt(const tuple& localPosition, const Intersection intersection = {}) const override
	{ 
		return localPosition; 
	}

	virtual bool boundingBox(BoundingBox& outputBox) override
	{
		return true;
	}

	Ray savedRay;
};

inline static std::shared_ptr<TestShape> testShape()
{
	return std::make_shared<TestShape>();
}

inline static bool boundingBoxOfShapes(const std::vector<std::shared_ptr<Shape>>& objects, BoundingBox& outputBox)
{
	if (objects.empty())
	{
		return false;
	}

	BoundingBox tempBox;
	bool firstBox = true;

	for (const auto& object : objects)
	{
		if (!object->boundingBox(tempBox))
		{
			return false;
		}
		outputBox = firstBox ? tempBox : surroundingBox(outputBox, tempBox);
		firstBox = false;
	}

	return true;
}