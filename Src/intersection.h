#pragma once

#include "ray.h"

struct Intersection
{
	float t = 0.0f;
	std::shared_ptr<class Shape> shape;
	std::shared_ptr<class Sphere> object;
	float a = 0.0f;
	float b = 0.0f;
	float u = 0.0f;
	float v = 0.0f;
};

struct HitResult
{
	float t = 0.0f;
	std::shared_ptr<class Shape> shape;
	tuple position;
	tuple viewDirection;
	tuple normal;
	bool inside = false;
	tuple overPosition;
	tuple underPosition;
	tuple reflectVector;
	float n1 = 1.0f;
	float n2 = 1.0f;
	tuple backgroundColor;
	float time = 0.0f;
	float a = 0.0f;
	float b = 0.0f;
	float u = 0.0f;
	float v = 0.0f;
};

inline static bool operator==(const Intersection& a, const Intersection& b)
{
	return Math::equal(a.t, b.t) && (a.shape == b.shape);
}

// No long needed, just preserve for old test cases
std::vector<Intersection> intersect(const std::shared_ptr<class Sphere>& sphere, const Ray& ray);

// No long needed, just preserve for old test cases
tuple normalAt(const std::shared_ptr<class Sphere>& sphere, const tuple& worldPoint);

// No long needed, just preserve for old test cases
tuple normalAt(const std::shared_ptr<class Shape>& shape, const tuple& worldPoint);

inline static bool compare(const Intersection& a, const Intersection& b)
{
	return a.t < b.t;
}

std::vector<Intersection> sortIntersections(const std::initializer_list<Intersection>& intersectionList);

void sortIntersections(std::vector<Intersection>& intersectionList);

Intersection hit(const std::vector<Intersection>& intersections);

std::vector<Intersection> intersectWorld(const class World& world, const Ray& ray);

HitResult prepareComputations(const Intersection& intersection, const Ray& ray, const std::vector<Intersection>& intersections = {});

Intersection intersectionWithUV(float t, const std::shared_ptr<Shape>& shape, float a, float b, float u, float v);