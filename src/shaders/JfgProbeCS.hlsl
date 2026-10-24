//
// RT64 - JFG PORT: the irradiance probes of the ray-traced indirect light (FramebufferRenderer::recordJfgProbes).
//
// A grid of probes around the camera (JFG_PROBES_X x Y x Z, JfgFrame.gi.y apart), each kept in the slot its world cell
// falls on (wrapping), so the grid follows the camera without moving what it has learnt. One thread group per probe,
// one ray per thread over the whole sphere (a Fibonacci sphere turned differently each frame):
//   a ray that gets out (nothing within the probe reach) sees the sky (JfgFrame.sky, upwards only);
//   a ray that hits the level or a character takes the light at that point: the frame's lights and the directional
//   light, each with a shadow ray, and the probes' own light there (so light bounces again over the frames), times an
//   albedo, as the game's surfaces have no albedo of their own.
// The light is gathered as an ambient cube (six directions) and blended into what the probe had, most of the old kept
// (JfgFrame.gi.z), which is the denoising: over frames and, where a pixel reads eight probes, across space.
//

#include "shared/rt64_jfg_light.h"

struct JfgProbeCB {
    uint lightCount;
    uint levelIndexCount; // the scene's level part (geometry 0) has this many indices, the objects' part follows it
    uint levelPart;       // 1 when the scene has a level part (geometry 0 is the level)
    uint frame;
};

[[vk::push_constant]] ConstantBuffer<JfgProbeCB> gConstants : register(b0, space0);
RaytracingAccelerationStructure jfgScene : register(t1, space0);
StructuredBuffer<JfgLight> jfgLights : register(t2, space0);
StructuredBuffer<JfgFrame> jfgFrame : register(t3, space0);
ByteAddressBuffer jfgWorldPos : register(t4, space0);
ByteAddressBuffer jfgIndices : register(t5, space0);
RWStructuredBuffer<JfgProbe> jfgProbes : register(u6, space0);

#define JFG_SHADOW_LIFT 40.0f
#define JFG_PROBE_RAYS 32
#define JFG_ALBEDO 0.6f

#include "JfgProbes.hlsli"

groupshared float4 gsCube[JFG_PROBE_RAYS][6];

bool jfgOpen(float3 origin, float3 target) {
    const float3 toTarget = target - origin;
    const float distanceToTarget = length(toTarget);
    if (distanceToTarget <= 1.0f) {
        return true;
    }

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = toTarget / distanceToTarget;
    ray.TMin = 0.5f;
    ray.TMax = distanceToTarget - 1.0f;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(jfgScene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    return query.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
}

float3 jfgVertex(uint index) {
    const uint v = jfgIndices.Load(index * 4);
    return asfloat(jfgWorldPos.Load3(v * 16));
}

// Rotates v about a unit axis by an angle (Rodrigues).
float3 jfgRotate(float3 v, float3 axis, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
}

float jfgHash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return float(x & 0xFFFFFF) / 16777216.0f;
}

[numthreads(JFG_PROBE_RAYS, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
    const JfgFrame f = jfgFrame[0];
    const uint slot = groupId.x;
    const uint ray = threadId.x;
    const int3 cell = jfgProbeCellOfSlot(slot, f);
    const float spacing = f.gi.y;
    const float3 probePos = (float3(cell) + 0.5f) * spacing;

    // This ray's direction: a Fibonacci sphere, turned about an axis and by an angle that change every frame.
    const float i = float(ray) + 0.5f;
    const float z = 1.0f - 2.0f * i / float(JFG_PROBE_RAYS);
    const float r = sqrt(max(1.0f - z * z, 0.0f));
    const float phi = i * 2.39996323f;
    const float3 axis = normalize(float3(jfgHash(gConstants.frame * 3 + 1) - 0.5f, jfgHash(gConstants.frame * 3 + 2) - 0.5f, jfgHash(gConstants.frame * 3 + 3) - 0.5f) + 1e-4f);
    const float3 dir = normalize(jfgRotate(float3(r * cos(phi), z, r * sin(phi)), axis, jfgHash(gConstants.frame * 7 + 11) * 6.2831853f));

    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    RayDesc rayDesc;
    rayDesc.Origin = probePos;
    rayDesc.Direction = dir;
    rayDesc.TMin = 0.5f;
    rayDesc.TMax = spacing * 24.0f;
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(jfgScene, RAY_FLAG_NONE, 0xFF, rayDesc);
    while (query.Proceed()) { }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // The surface hit: its triangle from the scene's indices (the objects' part after the level's).
        const uint geometry = query.CommittedGeometryIndex();
        const uint base = ((geometry == 0) && (gConstants.levelPart != 0)) ? 0 : gConstants.levelIndexCount;
        const uint first = base + query.CommittedPrimitiveIndex() * 3;
        const float3 a = jfgVertex(first);
        const float3 b = jfgVertex(first + 1);
        const float3 c = jfgVertex(first + 2);
        float3 n = cross(b - a, c - a);
        n = (dot(n, n) > 1e-8f) ? normalize(n) : -dir;
        n = (dot(n, dir) > 0.0f) ? -n : n; // the side the ray came from
        const float3 hit = probePos + dir * query.CommittedRayT();
        const float3 p = hit + n * 1.5f;

        float3 light = float3(0.0f, 0.0f, 0.0f);
        for (uint l = 0; l < gConstants.lightCount; l++) {
            const JfgLight jl = jfgLights[l];
            const float radius = jl.posRadius.w;
            const float3 toLight = jl.posRadius.xyz - hit;
            const float d = length(toLight);
            if ((d < radius) && (d > 0.001f)) {
                const float facing = saturate(dot(n, toLight / d));
                if (facing > 0.0f) {
                    const float inner = min(jl.colorInner.w, radius - 1.0f);
                    const float reach = saturate((radius - d) / (radius - inner));
                    const bool castsShadows = jl.shadow.w < 0.5f;
                    const float3 lightPos = jl.posRadius.xyz + float3(0.0f, castsShadows ? JFG_SHADOW_LIFT : 0.0f, 0.0f);
                    if (!castsShadows || jfgOpen(p, lightPos)) {
                        light += jl.colorInner.rgb * reach * facing;
                    }
                }
            }
        }

        if (f.sunDir.w > 0.5f) {
            const float facing = saturate(dot(n, f.sunDir.xyz));
            if ((facing > 0.0f) && jfgOpen(p, p + f.sunDir.xyz * (rayDesc.TMax + spacing * 8.0f))) {
                light += f.sunColor.rgb * facing;
            }
        }

        // Light bounced before (the probes there), so the light goes on bouncing over the frames.
        light += jfgProbeIrradiance(p, n, f);
        radiance = light * JFG_ALBEDO;
    }
    else if (dir.y > 0.0f) {
        radiance = f.sky.rgb * f.sky.w;
    }

    // Into the ambient cube: each of the six directions weighted by how much the ray goes its way.
    const float3 w = dir * dir;
    gsCube[ray][0] = (dir.x > 0.0f) ? float4(radiance * w.x, w.x) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gsCube[ray][1] = (dir.x < 0.0f) ? float4(radiance * w.x, w.x) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gsCube[ray][2] = (dir.y > 0.0f) ? float4(radiance * w.y, w.y) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gsCube[ray][3] = (dir.y < 0.0f) ? float4(radiance * w.y, w.y) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gsCube[ray][4] = (dir.z > 0.0f) ? float4(radiance * w.z, w.z) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gsCube[ray][5] = (dir.z < 0.0f) ? float4(radiance * w.z, w.z) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    GroupMemoryBarrierWithGroupSync();

    if (ray < 6) {
        float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint k = 0; k < JFG_PROBE_RAYS; k++) {
            sum += gsCube[k][ray];
        }

        float3 estimate = (sum.w > 1e-4f) ? (sum.rgb / sum.w) : float3(0.0f, 0.0f, 0.0f);
        estimate = all(isfinite(estimate)) ? max(estimate, 0.0f) : float3(0.0f, 0.0f, 0.0f);
        const JfgProbe old = jfgProbes[slot];
        const bool same = all(old.cell.xyz == cell) && (old.cell.w == JFG_PROBE_WRITTEN) && all(isfinite(old.cube[ray].rgb));
        const float keep = same ? saturate(f.gi.z) : 0.0f;
        jfgProbes[slot].cube[ray] = float4(lerp(estimate, old.cube[ray].rgb, keep), 1.0f);
    }

    GroupMemoryBarrierWithGroupSync();
    if (ray == 0) {
        jfgProbes[slot].cell = int4(cell, JFG_PROBE_WRITTEN);
    }
}
