#pragma once

#include "tuple.h"

#include "colors.h"
#include "matrix.h"
#include "transforms.h"

enum class PatternType : uint8_t
{
	Strip,
	Gradient,
	Ring,
	Checker,
	RadialGradient,
	Blend
};

class Pattern
{
public:
	Pattern() {}

	Pattern(const tuple& inColor1, const tuple& inColor2)
	: color1(inColor1), color2(inColor2)
	{}

	Pattern(const std::shared_ptr<Pattern>& inPattern1, const std::shared_ptr<Pattern>& inPattern2)
		: pattern1(inPattern1), pattern2(inPattern2)
	{}

	void setTransform(const matrix4& inTransform)
	{
		transform = inTransform;

		// Optimization: Cache inversed matrix
		inversedTransform = inverse(transform);
	}

	virtual tuple colorAt(const tuple& worldPosition)
	{
		return Colors::White;
	}

	tuple colorAt(const tuple& worldPosition, const matrix4& inversedObjectTransform)
	{
		//auto objectPosition = inverse(objectTransform) * worldPosition;
		// Optimization: Using cached inversed object transform
		auto objectPosition = inversedObjectTransform * worldPosition;
		auto patternPosition = inversedTransform * objectPosition;

		return colorAt(patternPosition);
	}

	tuple color1 = Colors::Black;
	tuple color2 = Colors::White;

	std::shared_ptr<Pattern> pattern1;
	std::shared_ptr<Pattern> pattern2;

	matrix4 transform;
	matrix4 inversedTransform;
};

// Chapter 11 Reflection and Refraction
// Test #7: Finding the Refracted Color Page 158~159
// Remember that the test pattern will return a color based on the point of
// intersection, which means the test can inspect the returned color to determine
// whether or not the ray was refracted.Sneaky!
class TestPattern : public Pattern
{
public:
	virtual tuple colorAt(const tuple& worldPosition) override
	{
		return color(worldPosition.x, worldPosition.y, worldPosition.z);
	}
};

class StripPattern : public Pattern
{
public:
	StripPattern(const tuple& inColor1, const tuple& inColor2)
	: Pattern(inColor1, inColor2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		if (Math::equal(std::fmodf(std::floor(worldPosition.x), 2.0f), 0.0f))
		{
			return color1;
		}

		return color2;
	}
};

inline static std::shared_ptr<Pattern> createStripPattern(const tuple& color1 = Colors::White, const tuple& color2 = Colors::Black)
{
	return std::make_shared<StripPattern>(color1, color2);
}

class GradientPattern : public Pattern
{
public:
	GradientPattern(const tuple& inColor1, const tuple& inColor2)
	: Pattern(inColor1, inColor2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		auto x = (worldPosition.x + 1.0f) * 0.5f;
		x = worldPosition.x;
		auto distance = color2 - color1;
		auto fraction = x - std::floorf(x);

		return color1 + distance * fraction;
	}
};

inline static std::shared_ptr<Pattern> createGradientPattern(const tuple& color1 = Colors::White, const tuple& color2 = Colors::Black)
{
	return std::make_shared<GradientPattern>(color1, color2);
}

class RingPattern : public Pattern
{
public:
	RingPattern(const tuple& inColor1, const tuple& inColor2)
	: Pattern(inColor1, inColor2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		auto distance = std::sqrtf(worldPosition.x * worldPosition.x + worldPosition.z * worldPosition.z);
		if (Math::equal(std::fmodf(std::floorf(distance), 2.0f), 0.0f))
		{
			return color1;
		}

		return color2;
	}
};

inline static std::shared_ptr<Pattern> createRingPattern(const tuple& color1 = Colors::White, const tuple& color2 = Colors::Black)
{
	return std::make_shared<RingPattern>(color1, color2);
}

class CheckerPattern : public Pattern
{
public:
	CheckerPattern(const tuple& inColor1, const tuple& inColor2)
	: Pattern(inColor1, inColor2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		auto y = 0.0f;

		if (std::fabsf(worldPosition.y) > EPSILON)
		{
			y = worldPosition.y;
		}

		auto sum = std::floorf(worldPosition.x) + std::floorf(y) + std::floorf(worldPosition.z);
		if (Math::equal(std::fmodf(sum, 2.0f), 0.0f))
		{
			return color1;
		}

		return color2;
	}
};

inline static std::shared_ptr<Pattern> createCheckerPattern(const tuple& color1 = Colors::White, const tuple& color2 = Colors::Black)
{
	return std::make_shared<CheckerPattern>(color1, color2);
}

class RadialGradientPattern : public Pattern
{
public:
	RadialGradientPattern(const tuple& inColor1, const tuple& inColor2)
	: Pattern(inColor1, inColor2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		auto x = (worldPosition.x + 1.0f) * 0.5f;
		auto z = (worldPosition.z + 1.0f) * 0.5f;
		x = worldPosition.x;
		z = worldPosition.z;

		auto distance = std::sqrtf(x * x + z * z);
		auto colorDistance = color2 - color1;
		auto fraction = distance - std::floorf(distance);

		return color1 + colorDistance * fraction;
	}
};

inline static std::shared_ptr<Pattern> createRadialGradientPattern(const tuple& color1 = Colors::White, const tuple& color2 = Colors::Black)
{
	return std::make_shared<RadialGradientPattern>(color1, color2);
}

class NestedPattern : public Pattern
{
public:
	NestedPattern(const std::shared_ptr<Pattern>& inPattern1, const std::shared_ptr<Pattern>& inPattern2)
	: Pattern(inPattern1, inPattern2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		auto y = 0.0f;

		if (std::fabsf(worldPosition.y) > EPSILON)
		{
			y = worldPosition.y;
		}

		// Optimization: Using cached inversed pattern transform
		//auto patternPosition1 = inverse(pattern1->transform) * worldPosition;
		//auto patternPosition2 = inverse(pattern2->transform) * worldPosition;
		auto patternPosition1 = pattern1->inversedTransform * worldPosition;
		auto patternPosition2 = pattern2->inversedTransform * worldPosition;

		auto sum = std::floorf(worldPosition.x) + std::floorf(y) + std::floorf(worldPosition.z);

		if (Math::equal(std::fmodf(sum, 2.0f), 0.0f))
		{
			return pattern1->colorAt(patternPosition1);
		}

		return pattern2->colorAt(patternPosition2);
	}
};

inline static std::shared_ptr<Pattern> createNestedPattern(const std::shared_ptr<Pattern>& pattern1, const std::shared_ptr<Pattern>& pattern2)
{
	return std::make_shared<NestedPattern>(pattern1, pattern2);
}

class BlendPattern : public Pattern
{
public:
	BlendPattern(const std::shared_ptr<Pattern>& inPattern1, const std::shared_ptr<Pattern>& inPattern2)
	: Pattern(inPattern1, inPattern2)
	{}

	tuple colorAt(const tuple& worldPosition) override
	{
		if (pattern1 != nullptr && pattern2 != nullptr)
		{
			// Optimization: Using cached inversed pattern transform
			//auto patternPosition1 = inverse(pattern1->transform) * worldPosition;
			//auto patternPosition2 = inverse(pattern2->transform) * worldPosition;
			auto patternPosition1 = pattern1->inversedTransform * worldPosition;
			auto patternPosition2 = pattern2->inversedTransform * worldPosition;

			return pattern1->colorAt(patternPosition1) * 0.5f + pattern2->colorAt(patternPosition2) * 0.5f;
		}

		return Colors::Black;
	}
};

inline static std::shared_ptr<Pattern> createBlendPattern(const std::shared_ptr<Pattern>& pattern1, const std::shared_ptr<Pattern>& pattern2)
{
	return std::make_shared<BlendPattern>(pattern1, pattern2);
}