#pragma once

#include "matrix.h"

inline float determinant(const matrix2& m);
inline float determinant(const matrix3& m);
inline float determinant(const matrix4& m);

inline matrix2 subMatrix(const matrix3& m, int32_t row, int32_t column);
inline matrix3 subMatrix(const matrix4& m, int32_t row, int32_t column);

inline float minor(const matrix3& m, int32_t row, int32_t column);
inline float minor(const matrix4& m, int32_t row, int32_t column);

inline float cofactor(const matrix3& m, int32_t row, int32_t column);
inline float cofactor(const matrix4& m, int32_t row, int32_t column);

inline matrix4 transpose(const matrix4& m);
inline matrix4 inverse(const matrix4& m);

inline matrix4 identity();
inline matrix4 translate(float x, float y, float z);
inline matrix4 scale(float x, float y, float z);
inline matrix4 rotateX(float radian);
inline matrix4 rotateY(float radian);
inline matrix4 rotateZ(float radian);
inline matrix4 shearing(float xy, float xz, float yx, float yz, float zx, float zy);

inline float determinant(const matrix2& m)
{
	return m(0, 0) * m(1, 1) -
		m(0, 1) * m(1, 0);
}

inline float determinant(const matrix3& m)
{
	auto result = 0.0f;

	for (auto column = 0; column < 3; column++) {
		result = result + m(0, column) * cofactor(m, 0, column);
	}

	return result;
}

inline float determinant(const matrix4& m)
{
	auto result = 0.0f;

	for (auto column = 0; column < 4; column++)
	{
		result = result + m(0, column) * cofactor(m, 0, column);
	}

	return result;
}

inline matrix2 subMatrix(const matrix3& m, int32_t row, int32_t column)
{
	matrix2 result;

	int32_t index = 0;

	for (int32_t i = 0; i < 3; i++)
	{
		if (i == row) continue;
		std::vector<float> subRow;
		for (int32_t j = 0; j < 3; j++)
		{
			if (j == column) continue;
			subRow.push_back(m(i, j));
		}
		result(index, 0) = subRow[0];
		result(index, 1) = subRow[1];
		index++;
	}

	return result;
}

inline matrix3 subMatrix(const matrix4& m, int32_t row, int32_t column)
{
	matrix3 result;

	int32_t index = 0;

	for (int32_t i = 0; i < 4; i++)
	{
		if (i == row) continue;
		std::vector<float> subRow;
		for (int32_t j = 0; j < 4; j++)
		{
			if (j == column) continue;
			subRow.push_back(m(i, j));
		}
		result(index, 0) = subRow[0];
		result(index, 1) = subRow[1];
		result(index, 2) = subRow[2];
		index++;
	}

	return result;
}

template<typename R, typename T, int D>
inline R subMatrix(const T& m, int32_t row, int32_t column)
{
	R result;
	int32_t index = 0;

	for (int32_t i = 0; i < D; i++)
	{
		if (i == row) continue;
		std::vector<float> subRow;
		for (int32_t j = 0; j < D; j++)
		{
			if (j == column) continue;
			subRow.push_back(m(i, j));
		}

		for (int k = 0; k < D - 1; k++)
		{
			result(index, k) = subRow[k];
		}
		index++;
	}

	return result;
}

inline float minor(const matrix3& m, int32_t row, int32_t column)
{
	return determinant(subMatrix(m, row, column));
}

inline float minor(const matrix4& m, int32_t row, int32_t column)
{
	return determinant(subMatrix(m, row, column));
}

inline float cofactor(const matrix3& m, int32_t row, int32_t column)
{
	float result = minor(m, row, column);

	if (!((row + column) % 2 == 0))
	{
		result = -result;
	}

	return result;
}

inline float cofactor(const matrix4& m, int32_t row, int32_t column)
{
	float result = minor(m, row, column);

	if (!((row + column) % 2 == 0))
	{
		result = -result;
	}

	return result;
}

inline matrix4 transpose(const matrix4& m)
{
	matrix4 result;

	result(0, 0) = m.column(0)[0];
	result(0, 1) = m.column(0)[1];
	result(0, 2) = m.column(0)[2];
	result(0, 3) = m.column(0)[3];

	result(1, 0) = m.column(1)[0];
	result(1, 1) = m.column(1)[1];
	result(1, 2) = m.column(1)[2];
	result(1, 3) = m.column(1)[3];

	result(2, 0) = m.column(2)[0];
	result(2, 1) = m.column(2)[1];
	result(2, 2) = m.column(2)[2];
	result(2, 3) = m.column(2)[3];

	result(3, 0) = m.column(3)[0];
	result(3, 1) = m.column(3)[1];
	result(3, 2) = m.column(3)[2];
	result(3, 3) = m.column(3)[3];

	return result;
}

inline matrix4 inverse(const matrix4& m)
{
	// Taken form glm::mat4::inverse()
	auto Coef00 = m(2, 2) * m(3, 3) - m(3, 2) * m(2, 3);
	auto Coef02 = m(1, 2) * m(3, 3) - m(3, 2) * m(1, 3);
	auto Coef03 = m(1, 2) * m(2, 3) - m(2, 2) * m(1, 3);

	auto Coef04 = m(2, 1) * m(3, 3) - m(3, 1) * m(2, 3);
	auto Coef06 = m(1, 1) * m(3, 3) - m(3, 1) * m(1, 3);
	auto Coef07 = m(1, 1) * m(2, 3) - m(2, 1) * m(1, 3);

	auto Coef08 = m(2, 1) * m(3, 2) - m(3, 1) * m(2, 2);
	auto Coef10 = m(1, 1) * m(3, 2) - m(3, 1) * m(1, 2);
	auto Coef11 = m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2);

	auto Coef12 = m(2, 0) * m(3, 3) - m(3, 0) * m(2, 3);
	auto Coef14 = m(1, 0) * m(3, 3) - m(3, 0) * m(1, 3);
	auto Coef15 = m(1, 0) * m(2, 3) - m(2, 0) * m(1, 3);

	auto Coef16 = m(2, 0) * m(3, 2) - m(3, 0) * m(2, 2);
	auto Coef18 = m(1, 0) * m(3, 2) - m(3, 0) * m(1, 2);
	auto Coef19 = m(1, 0) * m(2, 2) - m(2, 0) * m(1, 2);

	auto Coef20 = m(2, 0) * m(3, 1) - m(3, 0) * m(2, 1);
	auto Coef22 = m(1, 0) * m(3, 1) - m(3, 0) * m(1, 1);
	auto Coef23 = m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1);

	tuple Fac0(Coef00, Coef00, Coef02, Coef03);
	tuple Fac1(Coef04, Coef04, Coef06, Coef07);
	tuple Fac2(Coef08, Coef08, Coef10, Coef11);
	tuple Fac3(Coef12, Coef12, Coef14, Coef15);
	tuple Fac4(Coef16, Coef16, Coef18, Coef19);
	tuple Fac5(Coef20, Coef20, Coef22, Coef23);

	tuple Vec0(m(1, 0), m(0, 0), m(0, 0), m(0, 0));
	tuple Vec1(m(1, 1), m(0, 1), m(0, 1), m(0, 1));
	tuple Vec2(m(1, 2), m(0, 2), m(0, 2), m(0, 2));
	tuple Vec3(m(1, 3), m(0, 3), m(0, 3), m(0, 3));

	tuple Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
	tuple Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
	tuple Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
	tuple Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

	tuple SignA(+1, -1, +1, -1);
	tuple SignB(-1, +1, -1, +1);

	matrix4 Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

	tuple Row0(Inverse(0, 0), Inverse(1, 0), Inverse(2, 0), Inverse(3, 0));

	tuple Dot0(m.row(0) * Row0);

	auto Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

	auto OneOverDeterminant = 1.0f / Dot1;

	return Inverse * OneOverDeterminant;
}

inline matrix4 translate(float x, float y, float z)
{
	matrix4 result(1.0f);

	result(0, 3) = x;
	result(1, 3) = y;
	result(2, 3) = z;

	return result;
}

inline matrix4 translate(const tuple& translation)
{
	return translate(translation.x, translation.y, translation.z);
}

inline matrix4 identity()
{
	matrix4 result(1.0f);

	return result;
}

inline matrix4 scale(float x, float y, float z)
{
	matrix4 result(1.0f);

	result(0, 0) = x;
	result(1, 1) = y;
	result(2, 2) = z;

	return result;
}

inline matrix4 scale(const tuple& value)
{
	return scale(value.x, value.y, value.z);
}

inline matrix4 scale(float value)
{
	return scale(value, value, value);
}

inline matrix4 rotateX(float radian)
{
	auto result = matrix4(1.0f);

	result(1, 1) =  std::cos(radian);
	result(1, 2) = -std::sin(radian);
	result(2, 1) =  std::sin(radian);
	result(2, 2) =  std::cos(radian);

	return result;
}

inline matrix4 rotateY(float radian)
{
	auto result = matrix4(1.0f);

	result(0, 0) =  std::cos(radian);
	result(0, 2) =  std::sin(radian);
	result(2, 0) = -std::sin(radian);
	result(2, 2) =  std::cos(radian);

	return result;
}

inline matrix4 rotateZ(float radian)
{
	auto result = matrix4(1.0f);

	result(0, 0) =  std::cos(radian);
	result(0, 1) = -std::sin(radian);
	result(1, 0) =  std::sin(radian);
	result(1, 1) =  std::cos(radian);

	return result;
}

inline matrix4 rotate(const tuple& value)
{
	return rotateZ(value.z) * rotateY(value.y) * rotateX(value.x);
}

inline matrix4 shearing(float xy, float xz, float yx, float yz, float zx, float zy)
{
	auto result = matrix4(1.0);

	result(0, 1) = xy;
	result(0, 2) = xz;
	result(1, 0) = yx;
	result(1, 2) = yz;
	result(2, 0) = zx;
	result(2, 1) = zy;

	return result;
}

inline matrix4 viewTransform(const tuple& from, const tuple& to, const tuple& up)
{
	auto forward = normalize(to - from);
	auto left = cross(forward, normalize(up));
	auto trueUp = cross(left, forward);

	auto orientation = matrix4(left.x,     left.y,     left.z,     0.0f,
									  trueUp.x,   trueUp.y,   trueUp.z,   0.0f,
								     -forward.x, -forward.y, -forward.z,  0.0f,
									  0.0f,		  0.0f,       0.0f,       1.0f);

	orientation(0, 3) = -dot(left, from);
	orientation(1, 3) = -dot(trueUp, from);
	orientation(2, 3) =  dot(forward, from);

	return orientation;
	//return orientation * translate(-from.x, -from.y, -from.z);
}