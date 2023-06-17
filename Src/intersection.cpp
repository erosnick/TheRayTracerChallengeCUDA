#include "sphere.h"
#include "world.h"

#include "intersection.h"

std::vector<Intersection> intersect(const std::shared_ptr<Shape>& sphere, const Ray& ray)
{
	return sphere->intersect(ray);
}

tuple normalAt(const std::shared_ptr<Sphere>& sphere, const tuple& worldPoint)
{
	auto objectPoint = inverse(sphere->transform) * worldPoint;
	auto objectNormal = objectPoint - sphere->center;

	auto worldNormal = transpose(inverse(sphere->transform)) * objectNormal;
	worldNormal.w = 0.0f;

	return  normalize(worldNormal);
}

tuple normalAt(const std::shared_ptr<Shape>& shape, const tuple& worldPoint)
{
	return shape->normalAt(worldPoint);
}

std::vector<Intersection> sortIntersections(const std::initializer_list<Intersection>& intersectionList)
{
	std::vector<Intersection> result;

	std::for_each(intersectionList.begin(), intersectionList.end(), [&](const Intersection& intersection)
	{
		result.emplace_back(intersection);
	});

	std::sort(result.begin(), result.end(), compare);

	return result;
}

void sortIntersections(std::vector<Intersection>& intersectionList)
{
	std::sort(intersectionList.begin(), intersectionList.end(), compare);
}

Intersection hit(const std::vector<Intersection>& intersections)
{
	for (size_t i = 0; i < intersections.size(); i++)
	{
		if (intersections[i].t > 0.0f)
		{
			return intersections[i];
		}
	}

	return {};
}

std::vector<Intersection> intersectWorld(const World& world, const Ray& ray)
{
	std::vector<Intersection> result;

	for (const auto& shape : world.getObjects())
	{
		// Old method
		//auto intersection = intersect(object, ray);
		auto intersections= shape->intersect(ray);

		result.insert(result.end(), intersections.begin(), intersections.end());
	}

	sortIntersections(result);

	return result;
}

HitResult prepareComputations(const Intersection& intersection, const Ray& ray, const std::vector<Intersection>& intersections)
{
	// Instantiate a data structure for storing some precomputed values
	HitResult hitResult;

	// Copy the intersection's properties, for convenience
	hitResult.t = intersection.t;
	hitResult.shape = intersection.shape;

	hitResult.u = intersection.u;
	hitResult.v = intersection.v;
	hitResult.a = intersection.a;
	hitResult.b = intersection.b;

	// Precompute some useful values
	hitResult.position = ray.at(hitResult.t);
	hitResult.viewDirection = -ray.direction;

	// Old method
	//hitResult.normal = normalAt(hitResult.object, hitResult.position);
	hitResult.normal = hitResult.shape->normalAt(hitResult.position, intersection);

	if (dot(hitResult.normal, hitResult.viewDirection) < 0.0f)
	{
		hitResult.inside = true;
		hitResult.normal = -hitResult.normal;
	}
	else
	{
		hitResult.inside = false;
	}

	hitResult.time = ray.time;

	// After computing and (if appropriate) negating the normal vector...
	hitResult.overPosition = hitResult.position + hitResult.normal * EPSILON;
	hitResult.underPosition = hitResult.position - hitResult.normal * EPSILON;
	hitResult.reflectVector = reflect(ray.direction, hitResult.normal);

	// For refractive material
	auto containers = std::vector<std::shared_ptr<Shape>>();

	for (const auto& i : intersections)
	{
		if (i == intersection)
		{
			if (containers.empty())
			{
				hitResult.n1 = 1.0f;
			}
			else
			{
				auto shape = containers.back();
				hitResult.n1 = shape->material.refractiveIndex;
			}
		}

		if (auto result = std::find(containers.begin(), containers.end(), i.shape); result != containers.end())
		{
			containers.erase(result);
		}
		else
		{
			containers.emplace_back(i.shape);
		}

		if (i == intersection)
		{
			if (containers.empty())
			{
				hitResult.n2 = 1.0f;
			}
			else
			{
				auto shape = containers.back();
				hitResult.n2 = shape->material.refractiveIndex;
			}
			break;
		}
	}

	return hitResult;
}

Intersection intersectionWithUV(float t, const std::shared_ptr<Shape>& shape, float a, float b, float u, float v)
{
	Intersection intersection;
	intersection.t = t;
	intersection.shape = shape;
	intersection.a = a;
	intersection.b = b;
	intersection.u = u;
	intersection.v = v;
	return intersection;
}