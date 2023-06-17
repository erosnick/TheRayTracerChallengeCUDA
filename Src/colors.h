#pragma once

#include "tuple.h"

namespace Colors
{
	inline static tuple RGB(int32_t red, int32_t green, int32_t blue)
	{
		return color(red / 255.0f, green / 255.0f, blue / 255.0f);
	}

	const tuple White{ 1.0f, 1.0f, 1.0f, 0.0f };
	const tuple HalfWhite{ 0.5f, 0.5f, 0.5f, 0.0f };
	const tuple Black{ 0.0f, 0.0f, 0.0f, 0.0f };
	const tuple Dark{ 0.1f, 0.1f, 0.1f, 0.0f };
	const tuple Grey{ 0.9f, 0.9f, 0.9f, 0.0f };

	const tuple Red{ 1.0f, 0.0f, 0.0f, 0.0f };
	const tuple DarkRed{ 0.6f, 0.5f, 0.5f, 0.0f };
	const tuple Green{ 0.0f, 1.0f, 0.0f, 0.0f };
	const tuple DarkGreen = RGB(29, 72, 29);
	const tuple Blue{ 0.0f, 0.0f, 1.0f, 0.0f };
	const tuple Yellow{ 1.0f, 1.0f, 0.0f, 0.0f };
	const tuple CornFlower{ 0.4f, 0.6f, 0.9f, 0.0f };
	const tuple Background{ 0.235294f, 0.67451f, 0.843137f, 0.0f };
	const tuple Golden = RGB(224, 188, 78);
	const tuple Pink = RGB(255, 174, 201);
	const tuple Purple = RGB(220, 154, 238);
}