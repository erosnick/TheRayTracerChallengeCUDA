#pragma once

#include "tuple.h"
#include "material.h"
#include "light.h"
#include "colors.h"
#include "intersection.h"
#include "camera.h"
#include "canvas.h"
#include "world.h"

#include <thread>
#include <atomic>
#include <execution>
#include <mutex>

tuple lighting(const Material& material, const Light& light, const tuple& position, const tuple& viewDirection, const tuple& normal, float inShadow = false);
tuple lighting(const Material& material, const std::shared_ptr<Shape>& shape, const Light& light, const tuple& position, const tuple& viewDirection, const tuple& normal, float inShadow);
tuple lightingPBR(const Material& material, const std::shared_ptr<Shape>& shape, const Light& light, 
				   const tuple& position, const tuple& viewDirection, const tuple& normal, float inShadow, float u = 0.0f, float v = 0.0f);

tuple shadeHit(const World& world, const HitResult& hitResult, int32_t depth = 1);
tuple colorAt(const World& world, const Ray& ray, int32_t depth = 1);
Canvas render(const Camera& camera, const World& world, int32_t maxDepth, int32_t samplesPerPixel = 1);
std::vector<bool> isShadowed(const World& world, const tuple& position, float time = 0.0f);
tuple reflectedColor(const World& world, const HitResult& hitResult, int32_t depth);
tuple refractedColor(const World& world, const HitResult& hitResult, int32_t depth);
float schlick(const HitResult& hitResult);

// ----------------------------------------------------------------------------
float distributionGGX(tuple N, tuple H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = std::max(dot(N, H), 0.0f);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
	denom = RTC_PI * denom * denom;

	return nom / denom;
}
// ----------------------------------------------------------------------------
float geometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;

	float nom = NdotV;
	float denom = NdotV * (1.0f - k) + k;

	return nom / denom;
}
// ----------------------------------------------------------------------------
float geometrySmith(tuple N, tuple V, tuple L, float roughness)
{
	float NdotV = std::max(dot(N, V), 0.0f);
	float NdotL = std::max(dot(N, L), 0.0f);
	float ggx2 = geometrySchlickGGX(NdotV, roughness);
	float ggx1 = geometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
inline static tuple fresnelSchlick(float cosTheta, tuple F0)
{
	return F0 + (1.0 - F0) * std::powf(Math::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

tuple lighting(const Material& material, const Light& light, const tuple& position, const tuple& viewDirection, const tuple& normal, float inShadow)
{
	// Combine the surface color with the light's color/intensity
	auto effectiveColor = material.color * light.intensity;

	if (material.pattern != nullptr)
	{
		effectiveColor = material.pattern->colorAt(position) * light.intensity;
	}

	// Find the direction to the light source
	auto lightDirection = normalize(light.position - position);

	// Compute the ambient contribution
	auto ambient = effectiveColor * material.ambient;

	if (inShadow)
	{
		return ambient;
	}

	// lightDotNormal represents the cosine of the angle between the
	// light vector and the normal vector.A negative number means the
	// light is on the other side of the surface
	auto lightDotNormal = dot(lightDirection, normal);

	auto diffuse = Colors::Black;
	auto specular = Colors::Black;

	if (lightDotNormal < 0.0f)
	{
		diffuse = Colors::Black;
		specular = Colors::Black;
	}
	else
	{
		// Compute the diffuse contribution
		diffuse = effectiveColor * material.diffuse * lightDotNormal;

		// reflectDotEye represents the cosine of the angle between the
		// reflection vector and the eye vector.A negative number means the
		// light reflects away from the eye.
		auto reflectVector = reflect(-lightDirection, normal);
		auto reflectDotEye = dot(reflectVector, viewDirection);

		if (reflectDotEye <= 0.0f)
		{
			specular = Colors::Black;
		}
		else
		{
			// Compute the specular contribution
			auto factor = pow(reflectDotEye, material.shininess);
			specular = light.intensity * material.specular * factor;
		}
	}

	// Add the three contributions together to get the final shading
	return ambient + diffuse + specular;
}

tuple lighting(const Material& material, const std::shared_ptr<Shape>& shape, const Light& light, const tuple& position, const tuple& viewDirection, const tuple& normal, float inShadow)
{
	// Combine the surface color with the light's color/intensity
	auto effectiveColor = material.color * light.intensity;

	if (material.pattern != nullptr)
	{
		//effectiveColor = material.pattern->colorAt(position) * light.intensity;
		effectiveColor = material.pattern->colorAt(position, shape->transform) * light.intensity;
	}

	// Find the direction to the light source
	auto lightDirection = normalize(light.position - position);

	// Compute the ambient contribution
	auto ambient = effectiveColor * material.ambient;

	if (inShadow)
	{
		return ambient;
	}

	// lightDotNormal represents the cosine of the angle between the
	// light vector and the normal vector.A negative number means the
	// light is on the other side of the surface
	auto lightDotNormal = dot(lightDirection, normal);

	auto diffuse = Colors::Black;
	auto specular = Colors::Black;

	if (lightDotNormal < 0.0f)
	{
		diffuse = Colors::Black;
		specular = Colors::Black;
	}
	else
	{
		// Compute the diffuse contribution
		diffuse = effectiveColor * material.diffuse * lightDotNormal;

		// reflectDotEye represents the cosine of the angle between the
		// reflection vector and the eye vector.A negative number means the
		// light reflects away from the eye.
		auto reflectVector = reflect(-lightDirection, normal);
		auto reflectDotEye = dot(reflectVector, viewDirection);

		if (reflectDotEye <= 0.0f)
		{
			specular = Colors::Black;
		}
		else
		{
			// Compute the specular contribution
			auto factor = pow(reflectDotEye, material.shininess);
			specular = light.intensity * material.specular * factor;
		}
	}

	float distance = length(light.position - position);
	float attenuation = 1.0f / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

	// Add the three contributions together to get the final shading
	return ambient + (diffuse + specular) * attenuation;
}

tuple lightingPBR(const Material& material, const std::shared_ptr<Shape>& shape, const Light& light,
				   const tuple& position, const tuple& viewDirection, const tuple& normal, float inShadow, float u, float v)
{
	auto albedo = material.color;

	if (material.pattern)
	{
		albedo = material.pattern->colorAt(position, shape->inversedTransform);
	}

	if (material.texture)
	{
		albedo = material.texture->colorAt(position, u, v);
	}

	//albedo = pow(albedo, point(2.2f));

	// Compute the ambient contribution
	auto ambient = albedo * material.ambient;

	if (inShadow && material.emission == Colors::Black)
	{
		return ambient;
	}

	// calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
	// of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
	tuple F0 = color(0.04f);
	F0 = lerp(F0, albedo, material.metallic);

	// reflectance equation
	tuple Lo = color(0.0f);

	// calculate per-light radiance
	tuple L = normalize(light.position - position);
	tuple H = normalize(viewDirection + L);
	//float distance = length(light.position - position);
	//float attenuation = 1.0f / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
	float distance = length(light.position - position);
	float attenuation = 1.0f / (distance * distance);

	float intensity = 1.0f;

	if (light.type == LightType::Spot)
	{
		// spotlight (soft edges)
		float theta = dot(L, normalize(-light.direction));
		float epsilon = (light.cutOff - light.outerCutOff);
		intensity = Math::clamp((theta - light.outerCutOff) / epsilon, 0.0f, 1.0f);
	}

	tuple radiance = light.intensity * attenuation * intensity;

	tuple N = normalize(normal);

	// Cook-Torrance BRDF
	float NDF = distributionGGX(N, H, material.roughness);
	float G = geometrySmith(N, viewDirection, L, material.roughness);
	tuple F = fresnelSchlick(Math::clamp(dot(H, viewDirection), 0.0f, 1.0f), F0);

	tuple numerator = NDF * G * F;
	float denominator = 4.0f * std::max(dot(N, viewDirection), 0.0f) * std::max(dot(N, L), 0.0f) + 0.0001f; // + 0.0001 to prevent divide by zero
	tuple specular = numerator / denominator;

	// kS is equal to Fresnel
	tuple kS = F;
	// for energy conservation, the diffuse and specular light can't
	// be above 1.0 (unless the surface emits light); to preserve this
	// relationship the diffuse component (kD) should equal 1.0 - kS.
	tuple kD = point(1.0) - kS;
	// multiply kD by the inverse metalness such that only non-metals 
	// have diffuse lighting, or a linear blend if partly metal (pure metals
	// have no diffuse light).
	kD *= 1.0f - material.metallic;

	// scale light by NdotL
	float NdotL = std::max(dot(N, L), 0.0f);

	// add to outgoing radiance Lo
	// note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
	Lo += (kD * albedo / RTC_PI + specular) * radiance * NdotL;

	tuple finalColor = ambient + Lo + material.emission;

	// HDR tonemapping
	finalColor = finalColor / (finalColor + point(1.0));

	// gamma correct
	//finalColor = pow(finalColor, point(1.0f / 2.2f));

	return finalColor;
}

tuple shadeHit(const World& world, const HitResult& hitResult, int32_t depth)
{
	tuple finalColor;

	auto shadowResult = isShadowed(world, hitResult.overPosition, hitResult.time);

	for (int32_t i = 0; i < world.lightCount(); i++)
	{
		//finalColor += lighting(hitResult.shape->material, world.getLights()[i], hitResult.position, hitResult.viewDirection, hitResult.normal, shadowResult[i]);
		//finalColor += lighting(hitResult.shape->getMaterial(), hitResult.shape, world.getLights()[i], hitResult.position, hitResult.viewDirection, hitResult.normal, shadowResult[i]);
		finalColor += lightingPBR(hitResult.shape->getMaterial(),
								   hitResult.shape, world.getLights()[i],
								   hitResult.position, 
								   hitResult.viewDirection, 
								   hitResult.normal, shadowResult[i], hitResult.u, hitResult.v);
	}

	auto reflected = reflectedColor(world, hitResult, depth);
	auto refracted = refractedColor(world, hitResult, depth);

	auto material = hitResult.shape->getMaterial();

	if (material.metallic > 0.0f && material.transparency > 0.0f)
	{
		auto reflectance = schlick(hitResult);
		return finalColor + reflected * reflectance + refracted * (1.0f - reflectance);
	}

	return finalColor + reflected + refracted;
}

inline static tuple computeBackgroundColor(const Ray& ray)
{
	auto t = 0.5f * (ray.direction.y + 1.0f);
	return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

tuple colorAt(const World& world, const Ray& ray, int32_t depth)
{
	auto intersections = intersectWorld(world, ray);

	auto intersection = hit(intersections);

	auto backgroundColor = Colors::Black;// computeBackgroundColor(ray);

	if (intersection.t > 0.0f)
	{
		auto hitResult = prepareComputations(intersection, ray, intersections);
		hitResult.backgroundColor = backgroundColor;
		return shadeHit(world, hitResult, depth);
	}

	return backgroundColor;
}

Canvas render(const Camera& camera, const World& world, int32_t maxDepth, int32_t samplesPerPixel)
{
	auto image = Canvas(camera.imageWidth, camera.imageHeight);

	// Total number of iterations or tasks
	int32_t pixelCount = camera.imageWidth * camera.imageHeight;

	// Variable to track completed iterations
	std::atomic_int completedIterations = 0;

	// Mutex to ensure thread-safe access to completedIterations
	std::mutex mtx;

	std::vector<int32_t> imageHorizontalIterator;
	std::vector<int32_t> imageVerticalIterator;

	imageHorizontalIterator.resize(camera.imageWidth);
	imageVerticalIterator.resize(camera.imageHeight);

	for (int32_t i = 0; i < camera.imageWidth; i++)
	{
		imageHorizontalIterator[i] = i;
	}

	for (int32_t i = 0; i < camera.imageHeight; i++)
	{
		imageVerticalIterator[i] = i;
	}

	std::cout << "Start Rendering...\n";

	AriaCore::Timer timer("Counter");

	std::for_each(std::execution::par, imageVerticalIterator.begin(), imageVerticalIterator.end(),
	[&](int32_t y)
	{
		// Update completedIterations atomically
		completedIterations.fetch_add(camera.imageWidth, std::memory_order_relaxed);
		printf("\rScanlines remaining: %.0f%%(%.0fs)",  completedIterations / static_cast<float>(pixelCount) * 100.0f, timer.Elapsed());

		std::for_each(std::execution::par, imageHorizontalIterator.begin(), imageHorizontalIterator.end(),
		[&, y](int32_t x)
			{
				auto finalColor = Colors::Black;
				for (auto sample = 0; sample < samplesPerPixel; sample++)
				{
					auto rx = Math::randomFloat();
					auto ry = Math::randomFloat();
					auto ray = camera.rayForPixel(static_cast<float>(x + rx), static_cast<float>(y + ry));
					finalColor += colorAt(world, ray, maxDepth);
					if (x == 0 && y == 2)
					{
						//finalColor = Colors::Green;
					}
				}
				image.writePixel(x, y, finalColor / static_cast<float>(samplesPerPixel));
			});
	});

	printf("\nRendering done.\n");

	return image;
}

inline std::vector<bool> isShadowed(const World & world, const tuple & position, float time)
{
	std::vector<bool> shadowResult(world.lightCount(), false);

	for (int32_t i = 0; i < world.lightCount(); i++)
	{
		auto toLight = world.getLight(i).position - position;

		auto distance = length(toLight);
		auto direction = normalize(toLight);

		auto ray = Ray(position, direction, time);
		auto intersections = intersectWorld(world, ray);
		auto intersection = hit(intersections);

		if (intersection.t > 0.0f && intersection.t < distance)
		{
			shadowResult[i] = true;

			if (!intersection.shape->getMaterial().castShadow)
			{
				shadowResult[i] = false;
			}
		}
	}

	return shadowResult;
}

tuple reflectedColor(const World& world, const HitResult& hitResult, int32_t depth)
{
	if (Math::equal(hitResult.shape->getMaterial().metallic, 0.0f) || depth == 0)
	{
		return Colors::Black;
	}

	auto reflectedRay = Ray(hitResult.overPosition, hitResult.reflectVector, hitResult.time);
	auto color = colorAt(world, reflectedRay, depth - 1);

	return color * hitResult.shape->getMaterial().metallic;
}

tuple refractedColor(const World& world, const HitResult& hitResult, int32_t depth)
{
	if (Math::equal(hitResult.shape->getMaterial().transparency, 0.0f) || depth == 0)
	{
		return Colors::Black;
	}

	// Find the ratio of first index of refraction to the second.
	// (Yup, this is inverted from the definition of Snell's Law.)
	auto nRatio = hitResult.n1 / hitResult.n2;

	// cos(thetaI) is the same as the dot product of the two vectors
	auto cosI = dot(hitResult.viewDirection, hitResult.normal);

	// Find sin(thetaT) ^ 2 via trigonometric identity
	auto sin2T = nRatio * nRatio * (1.0f - cosI * cosI);

	if (sin2T > 1.0f)
	{
		return Colors::Black;
	}

	// Find cos(thetaT) via trigonometric identity
	auto cosT = std::sqrtf(1.0f - sin2T);

	// Compute the direction of the refracted ray
	auto direction = hitResult.normal * (nRatio * cosI - cosT) - hitResult.viewDirection * nRatio;

	// Create the refracted ray
	auto refractedRay = Ray(hitResult.underPosition, direction, hitResult.time);

	// Find the color of the refracted ray, making sure to multiply
	// by the transparency value to account for any opacity
	auto color = colorAt(world, refractedRay, depth - 1);

	return color * hitResult.shape->getMaterial().transparency;
}

float schlick(const HitResult& hitResult)
{
	// Find the cosine of the angle between the eye and normal vectors
	auto cos = dot(hitResult.viewDirection, hitResult.normal);

	// Total internal reflection can only occur if n1 > n2
	if (hitResult.n1 > hitResult.n2)
	{
		auto n = hitResult.n1 / hitResult.n2;
		auto sin2T = n * n * (1.0f - cos * cos);
		if (sin2T > 1.0f)
		{
			return 1.0f;
		}

		// Compute cosine of theta_t using trig identity
		auto cosT = std::sqrtf(1.0f - sin2T);

		// When n1 > n2, use cos(theta_t) instead
		cos = cosT;
	}

	auto r0 = std::powf(((hitResult.n1 - hitResult.n2) / (hitResult.n1 + hitResult.n2)), 2.0f);

	return r0 + (1.0f - r0) * std::powf((1.0f - cos), 5.0f);
}