#pragma once

#include "constants.h"

#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef CBRT
#define     cbrt(x)  ((x) > 0.0 ? pow((double)(x), 1.0f / 3.0f) : \
			  		 ((x) < 0.0 ? -pow((double)-(x), 1.0f / 3.0f) : 0.0f))
#endif

namespace Math
{
	HOST_DEVICE inline static bool equal(float a, float b)
	{
		return std::fabsf(a - b) < EPSILON;
	}

	inline static bool isZero(float a)
	{
		return std::fabsf(a) <= EPSILON;
	}

	inline static bool isZero(double a)
	{
		return std::fabs(a) <= EPSILON_HIGH_PRECISION;
	}

	inline static bool isZeroHighPrecision(float a)
	{
		return std::fabsf(a) <= EPSILON_HIGH_PRECISION;
	}

	inline static bool between(float a, float left, float right)
	{
		return a > left && a < right;
	}

	template<typename T>
	inline static T clamp(T value, T min, T max)
	{
		T result = value;

		if (result < min)
		{
			result = min;
		}

		if (result > max)
		{
			result = max;
		}

		return result;
	}

	inline static float radians(float angle)
	{
		return (RTC_PI / 180.0f) * angle;
	}

	inline static double randomDouble()
	{
		static std::uniform_real_distribution<double> distribution(0.0, 1.0);
		static std::mt19937_64 generator;
		return distribution(generator);
	}

	inline static float randomFloat()
	{
		static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
		static std::mt19937_64 generator;
		return distribution(generator);
	}

	inline static float randomFloat(float min, float max)
	{
		static std::uniform_real_distribution<float> distribution(min, max);
		static std::mt19937_64 generator;
		return distribution(generator);
	}

	inline static int32_t randomInt(int32_t min, int32_t max)
	{
		static std::uniform_int_distribution<int32_t> distribution(min, max);
		static std::mt19937_64 generator;
		return distribution(generator);
	}

	__device__ inline static float randomFloatCUDA(curandState* state, float min, float max)
	{
		curand_init(1234, 0, 0, state);
		return curand_uniform(state) * (max - min) + min;
	}

	inline static float max(float a, float b, float c)
	{
		return std::max(std::max(a, b), c);
	}

	inline static float min(float a, float b, float c)
	{
		return std::min(std::min(a, b), c);
	}

	inline static int solveQuadric(double c[3], double s[2])
	{
		double p, q, D;

		/* normal form: x^2 + px + q = 0 */

		p = c[1] / (2 * c[2]);
		q = c[0] / c[2];

		D = p * p - q;

		if (isZero(D)) {
			s[0] = -p;
			return 1;
		}
		else if (D > 0) {
			double sqrt_D = std::sqrt(D);

			s[0] = sqrt_D - p;
			s[1] = -sqrt_D - p;
			return 2;
		}
		else /* if (D < 0) */
			return 0;
	}

	inline static int solveCubic(double c[4], double s[3])
	{
		int     i, num;
		double  sub;
		double  A, B, C;
		double  sq_A, p, q;
		double  cb_p, D;

		/* normal form: x^3 + Ax^2 + Bx + C = 0 */

		A = c[2] / c[3];
		B = c[1] / c[3];
		C = c[0] / c[3];

		/*  substitute x = y - A/3 to eliminate quadric term:
		x^3 +px + q = 0 */

		sq_A = A * A;
		p = 1.0f / 3 * (-1.0f / 3 * sq_A + B);
		q = 1.0f / 2 * (2.0f / 27 * A * sq_A - 1.0f / 3 * A * B + C);

		/* use Cardano's formula */

		cb_p = p * p * p;
		D = q * q + cb_p;

		if (isZero(D)) {
			if (isZero(q)) { /* one triple solution */
				s[0] = 0;
				num = 1;
			}
			else { /* one single and one double solution */
				double u = cbrt(-q);
				s[0] = 2 * u;
				s[1] = -u;
				num = 2;
			}
		}
		else if (D < 0) { /* Casus irreducibilis: three real solutions */
			double phi = 1.0 / 3 * std::acos(-q / std::sqrt(-cb_p));
			double t = 2 * std::sqrt(-p);

			s[0] = t * std::cos(phi);
			s[1] = -t * std::cos(phi + RTC_PI / 3);
			s[2] = -t * std::cos(phi - RTC_PI / 3);
			num = 3;
		}
		else { /* one real solution */
			double sqrt_D = std::sqrt(D);
			double u = cbrt(sqrt_D - q);
			double v = -cbrt(sqrt_D + q);

			s[0] = u + v;
			num = 1;
		}

		/* resubstitute */

		sub = 1.0f / 3 * A;

		for (i = 0; i < num; ++i)
			s[i] -= sub;

		return num;
	}

	inline static int solveQuartic(double c[5], double s[4])
	{
		double  coeffs[4];
		double  z, u, v, sub;
		double  A, B, C, D;
		double  sq_A, p, q, r;
		int     i, num;

		/* normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0 */

		A = c[3] / c[4];
		B = c[2] / c[4];
		C = c[1] / c[4];
		D = c[0] / c[4];

		/*  substitute x = y - A/4 to eliminate cubic term:
		x^4 + px^2 + qx + r = 0 */

		sq_A = A * A;
		p = -3.0f / 8 * sq_A + B;
		q = 1.0f / 8 * sq_A * A - 1.0f / 2 * A * B + C;
		r = -3.0f / 256 * sq_A * sq_A + 1.0f / 16 * sq_A * B - 1.0f / 4 * A * C + D;

		if (isZero(r)) {
			/* no absolute term: y(y^3 + py + q) = 0 */

			coeffs[0] = q;
			coeffs[1] = p;
			coeffs[2] = 0;
			coeffs[3] = 1;

			num = solveCubic(coeffs, s);

			s[num++] = 0;
		}
		else {
			/* solve the resolvent cubic ... */

			coeffs[0] = 1.0f / 2 * r * p - 1.0f / 8 * q * q;
			coeffs[1] = -r;
			coeffs[2] = -1.0f / 2 * p;
			coeffs[3] = 1;

			(void)solveCubic(coeffs, s);

			/* ... and take the one real solution ... */

			z = s[0];

			/* ... to build two quadric equations */

			u = z * z - r;
			v = 2 * z - p;

			if (isZero(u))
				u = 0;
			else if (u > 0)
				u = std::sqrt(u);
			else
				return 0;

			if (isZero(v))
				v = 0;
			else if (v > 0)
				v = std::sqrt(v);
			else
				return 0;

			coeffs[0] = z - u;
			coeffs[1] = q < 0 ? -v : v;
			coeffs[2] = 1;

			num = solveQuadric(coeffs, s);

			coeffs[0] = z + u;
			coeffs[1] = q < 0 ? v : -v;
			coeffs[2] = 1;

			num += solveQuadric(coeffs, s + num);
		}

		/* resubstitute */

		sub = 1.0f / 4 * A;

		for (i = 0; i < num; ++i)
			s[i] -= sub;

		return num;
	}
}