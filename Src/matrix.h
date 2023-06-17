#pragma once

#include "tuple.h"

struct matrix2
{
	matrix2()
	: matrix2(1.0f)
	{
	}

	matrix2(float value)
	{
		data[0][0] = value;
		data[0][1] = 0.0f;

		data[1][0] = 0.0f;
		data[1][1] = value;
	}

	matrix2(float m00, float m01, float m10, float m11)
	{
		data[0][0] = m00;
		data[0][1] = m01;

		data[1][0] = m10;
		data[1][1] = m11;
	}

	float operator()(int32_t row, int32_t column) const
	{
		return data[row][column];
	}

	float& operator()(int32_t row, int32_t column)
	{
		return data[row][column];
	}

	float data[2][2];
};

struct matrix3
{
	matrix3()
	: matrix3(1.0f)
	{}

	matrix3(float value)
	{
		data[0][0] = value;
		data[0][1] = 0.0f;
		data[0][2] = 0.0f;

		data[1][0] = 0.0f;
		data[1][1] = value;
		data[1][2] = 0.0f;

		data[2][0] = 0.0f;
		data[2][1] = 0.0f;
		data[2][2] = value;
	}

	matrix3(float m00, float m01, float m02, 
			float m10, float m11, float m12, 
			float m20, float m21, float m22)
	{
		data[0][0] = m00;
		data[0][1] = m01;
		data[0][2] = m02;

		data[1][0] = m10;
		data[1][1] = m11;
		data[1][2] = m12;

		data[2][0] = m20;
		data[2][1] = m21;
		data[2][2] = m22;
	}

	float operator()(int32_t row, int32_t column) const
	{
		return data[row][column];
	}

	float& operator()(int32_t row, int32_t column)
	{
		return data[row][column];
	}

	tuple row(int32_t index) const
	{
		return tuple(data[index][0], data[index][1], data[index][2], 0.0f);
	}

	tuple column(int32_t index) const
	{
		return tuple(data[0][index], data[1][index], data[2][index], 0.0f);
	}

	float data[3][3];
};

struct matrix4
{
	matrix4()
	: matrix4(1.0f)
	{
	}

	matrix4(float value)
	{
		data[0][0] = value;
		data[0][1] = 0.0f;
		data[0][2] = 0.0f;
		data[0][3] = 0.0f;

		data[1][0] = 0.0f;
		data[1][1] = value;
		data[1][2] = 0.0f;
		data[1][3] = 0.0f;
	
		data[2][0] = 0.0f;
		data[2][1] = 0.0f;
		data[2][2] = value;
		data[2][3] = 0.0f;
	
		data[3][0] = 0.0f;
		data[3][1] = 0.0f;
		data[3][2] = 0.0f;
		data[3][3] = value;
	}

	matrix4(float m00, float m01, float m02, float m03, 
			float m10, float m11, float m12, float m13, 
			float m20, float m21, float m22, float m23, 
			float m30, float m31, float m32, float m33)
	{
		data[0][0] = m00;
		data[0][1] = m01;
		data[0][2] = m02;
		data[0][3] = m03;

		data[1][0] = m10;
		data[1][1] = m11;
		data[1][2] = m12;
		data[1][3] = m13;

		data[2][0] = m20;
		data[2][1] = m21;
		data[2][2] = m22;
		data[2][3] = m23;

		data[3][0] = m30;
		data[3][1] = m31;
		data[3][2] = m32;
		data[3][3] = m33;

	}

	matrix4(const tuple& row0, const tuple& row1, const tuple& row2, const tuple& row3)
	: matrix4(row0[0], row0[1], row0[2], row0[3],
			  row1[0], row1[1], row1[2], row1[3],
			  row2[0], row2[1], row2[2], row2[3],
			  row3[0], row3[1], row3[2], row3[3])
	{
	}

	float operator()(int32_t row, int32_t column) const
	{
		return data[row][column];
	}

	float& operator()(int32_t row, int32_t column)
	{
		return data[row][column];
	}

	tuple row(int32_t index) const
	{
		return tuple(data[index][0], data[index][1], data[index][2], data[index][3]);
	}

	tuple column(int32_t index) const
	{
		return tuple(data[0][index], data[1][index], data[2][index], data[3][index]);
	}

	float data[4][4];
};

inline bool operator==(const matrix2& a, const matrix2& b)
{
	return Math::equal(a(0, 0), b(0, 0)) &&
		   Math::equal(a(0, 1), b(0, 1)) &&
		   Math::equal(a(1, 0), b(1, 0)) &&
		   Math::equal(a(1, 1), b(1, 1));
}

inline bool operator==(const matrix3& a, const matrix3& b)
{
	return equal(a.row(0), b.row(0)) &&
		   equal(a.row(1), b.row(1)) &&
		   equal(a.row(2), b.row(2));
}

inline bool operator==(const matrix4& a, const matrix4& b)
{
	return equal(a.row(0), b.row(0)) &&
		   equal(a.row(1), b.row(1)) &&
		   equal(a.row(2), b.row(2)) &&
		   equal(a.row(3), b.row(3));
}

inline bool operator!=(const matrix4& a, const matrix4& b)
{
	return !(a == b);
}

inline matrix4 operator*(const matrix4& a, const matrix4& b)
{
	matrix4 result;

	result(0, 0) = dot(a.row(0), b.column(0));
	result(0, 1) = dot(a.row(0), b.column(1));
	result(0, 2) = dot(a.row(0), b.column(2));
	result(0, 3) = dot(a.row(0), b.column(3));

	result(1, 0) = dot(a.row(1), b.column(0));
	result(1, 1) = dot(a.row(1), b.column(1));
	result(1, 2) = dot(a.row(1), b.column(2));
	result(1, 3) = dot(a.row(1), b.column(3));

	result(2, 0) = dot(a.row(2), b.column(0));
	result(2, 1) = dot(a.row(2), b.column(1));
	result(2, 2) = dot(a.row(2), b.column(2));
	result(2, 3) = dot(a.row(2), b.column(3));

	result(3, 0) = dot(a.row(3), b.column(0));
	result(3, 1) = dot(a.row(3), b.column(1));
	result(3, 2) = dot(a.row(3), b.column(2));
	result(3, 3) = dot(a.row(3), b.column(3));

	return result;
}

inline tuple operator*(const matrix4& a, const tuple& b)
{
	return tuple(dot(a.row(0), b),
				 dot(a.row(1), b),
				 dot(a.row(2), b),
				 dot(a.row(3), b));
}