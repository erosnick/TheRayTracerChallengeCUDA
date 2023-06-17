#pragma once

#include "tuple.h"

enum class LightType : uint8_t
{
	Point,
	Directional,
	Spot
};

struct Light
{
	LightType type = LightType::Point;
	tuple position;
	tuple direction;
	tuple intensity = Colors::White;
	float constant = 1.0f;
	float linear = 0.022f;
	float quadratic = 0.0019f;
	float cutOff = std::cos(Math::radians(12.5f));
	float outerCutOff = std::cos(Math::radians(17.5f));
};

inline static bool operator==(const Light& a, const Light& b)
{
	return (a.position == b.position) && 
		   (a.intensity == b.intensity);
}

inline static Light pointLight(const tuple& position, const tuple& intensity)
{
	return { LightType::Point, position, vector(0.0f), intensity };
}

inline static Light spotLight(const tuple& position, const tuple& direction, const tuple& intensity)
{
	return { LightType::Spot, position, direction, intensity };
}