#pragma once

#include "tuple.h"
#include "pattern.h"
#include "texture.h"

#include <functional>

struct Material
{
	tuple color{ 1.0f, 1.0f, 1.0f, 0.0f };
	float ambient = 0.1f;
	float diffuse = 0.9f;
	float specular = 0.9f;
	float shininess = 200.0f;
	float metallic = 0.0f;		
	float roughness = 0.0f;
	float transparency = 0.0f;
	float refractiveIndex = 1.0f;
	bool castShadow = true;
	tuple emission = Colors::Black;
	std::shared_ptr<Pattern> pattern;
	std::shared_ptr<Texture> texture;
	std::function<void(tuple& normal, const tuple& position, float amplitude, float frequency, float phase)> sinNormalPerturb;
	std::function<void(tuple& normal, const tuple& position, float amplitude, float frequency, float phase)> cosNormalPerturb;
	std::function<void(tuple& normal, const tuple& position, double scale, int octaves, double persistence, double lacunarity)> noiseNormalPerturb;
};

inline static bool operator==(const Material& a, const Material& b)
{
	return a.color == b.color &&
	Math::equal(a.ambient, b.ambient) &&
	Math::equal(a.diffuse, b.diffuse) &&
	Math::equal(a.specular, b.specular) &&
	Math::equal(a.shininess, b.shininess);
}

namespace Materials
{
	const Material White{ Colors::White, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Black{ Colors::Black, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Grey{ Colors::Grey, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Red{ Colors::Red, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Green{ Colors::Green, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Blue{ Colors::Blue, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Yellow{ Colors::Yellow, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material CornFlower{ Colors::CornFlower, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material DarkRed{ Colors::DarkRed, 0.1f, 0.9f, 0.3f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const Material Glass{ Colors::Dark, 0.1f, 0.9f, 1.0f, 300.0f, 1.0f, 1.0f, 1.0f, 1.52f, false };
	const Material Water{ Colors::Dark, 0.1f, 0.9f, 1.0f, 200.0f, 1.0f, 1.0f, 1.0f, 1.33f, false };
	const Material Diamond{ Colors::Dark, 0.1f, 0.9f, 1.0f, 200.0f, 1.0f, 1.0f, 1.0f, 2.417f, false };
	const Material Mirror{ Colors::Black, 0.1f, 0.9f, 1.0f, 200.0f, 1.0f, 0.0f, 0.0f, 1.0f };
}