#pragma once

#include <memory>
#include <string>

#include "plane.h"
#include "sphere.h"
#include "light.h"
#include "colors.h"

class World
{
public:
	World() {}

	void setName(const std::string& inName)
	{
		name = inName;
	}

	void addLight(const Light& light)
	{
		lights.emplace_back(light);
	}

	void addObject(const std::shared_ptr<Shape>& object)
	{
		objects.emplace_back(object);
	}

	int32_t lightCount() const { return static_cast<int32_t>(lights.size()); }
	int32_t objectCount() const { return static_cast<int32_t>(objects.size()); }

	const auto& getLights() const { return lights; }
	const auto& getObjects() const { return objects; }

	const auto& getLight(int32_t index) const { return lights[index]; }
	const auto& getObject(int32_t index) const { return objects[index]; }

	auto& getLight(int32_t index) { return lights[index]; }
	auto& getObject(int32_t index) { return objects[index]; }

	auto getName() const { return name; }
private:
	std::vector<std::shared_ptr<Shape>> objects;
	std::vector<Light> lights;
	std::string name;
};

inline static World defaultWorld()
{
	World world;

	world.addLight(pointLight(point(-10.0f, 10.0f, -10.0f), Colors::White));

	auto sphere1 = createSphere();

	sphere1->material.color = color(0.8f, 1.0f, 0.6f);
	sphere1->material.diffuse = 0.7f;
	sphere1->material.specular = 0.2f;

	world.addObject(sphere1);

	auto sphere2 = createSphere(scale(0.5f, 0.5f, 0.5f));

	world.addObject(sphere2);

	auto plane = createPlane();

	plane->material.metallic = 0.5f;
	plane->setTransform(translate(0.0f, -1.0f, 0.0f));

	world.addObject(plane);

	return world;
}