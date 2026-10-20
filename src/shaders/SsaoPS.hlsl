//
// RT64 - JFG PORT: screen-space ambient occlusion for Jet Force Gemini.
//
// Two full-screen passes (FullScreenVS) at the point of a frame where the game has finished its 3D scene and starts
// drawing 2D (see FramebufferRenderer::submitRasterScene), so the HUD is drawn afterwards and stays untouched:
//   this pass        scene depth -> occlusion buffer (r: how much light reaches the pixel, g: its distance from the
//                    camera). View-space positions are rebuilt from depth with the scene's own projection, the surface
//                    normal from neighbouring depths, and occlusion is the share of nearby points that lie above the
//                    surface (Alchemy-style AO).
//   SSAO_BLUR (BloomPS.hlsl)   occlusion buffer -> scene colour: a blur that leaves out neighbours at another distance
//                    (so edges stay sharp) to smooth out the per-pixel sample rotation, multiplied into the picture.
//
// Parameters (RasterParams): padding.x = strength, padding.y = radius (game units), padding.z = tan(half the vertical
// field of view), screenScale.x / screenScale.y = near / far plane. All as floats; a 0 field of view or near plane
// means the scene's projection was not found and the game's usual ones are used.
//

#include "shared/rt64_raster_params.h"

#include "FbRendererCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

#if defined(MULTISAMPLING)
Texture2DMS<float> gBackgroundDepth : register(t2, space3);
float loadDepth(int2 p) {
    return gBackgroundDepth.Load(p, 0);
}
void depthSize(out uint w, out uint h) {
    uint samples;
    gBackgroundDepth.GetDimensions(w, h, samples);
}
#else
Texture2D<float> gBackgroundDepth : register(t2, space3);
float loadDepth(int2 p) {
    return gBackgroundDepth.Load(int3(p, 0));
}
void depthSize(out uint w, out uint h) {
    gBackgroundDepth.GetDimensions(w, h);
}
#endif

// The game's usual projection: near 10, far 15000, tan(30 degrees).
float nearZ() {
    const float n = gConstants.screenScale.x;
    return (n > 0.0f) ? n : 10.0f;
}
float farZ() {
    const float f = gConstants.screenScale.y;
    return (f > nearZ()) ? f : 15000.0f;
}
float tanHalfFovY() {
    const float t = asfloat(gConstants.padding.z);
    return (t > 0.0f) ? t : 0.57735f;
}
static const int SampleCount = 16;

// Distance from the camera along the view direction.
float linearDepth(float d, float n, float f) {
    const float ndcZ = d * 2.0f - 1.0f;
    return (2.0f * n * f) / ((f + n) - ndcZ * (f - n));
}

float3 viewPosition(int2 p, float2 size, float aspect, float n, float f, float tanHalf) {
    const float z = linearDepth(loadDepth(p), n, f);
    const float2 ndc = float2(((float(p.x) + 0.5f) / size.x) * 2.0f - 1.0f, 1.0f - ((float(p.y) + 0.5f) / size.y) * 2.0f);
    return float3(ndc.x * z * aspect * tanHalf, ndc.y * z * tanHalf, z);
}

void PSMain(
      in float4 vertexPosition : SV_Position
    , in float2 vertexUV : TEXCOORD0
    , [[vk::location(0)]] [[vk::index(0)]] out float4 resultColor : SV_TARGET0
)
{
    const float strength = asfloat(gConstants.padding.x);
    const float radius = asfloat(gConstants.padding.y);
    const float n = nearZ();
    const float f = farZ();
    const float tanHalf = tanHalfFovY();

    uint width, height;
    depthSize(width, height);
    const float2 size = float2(width, height);
    const float aspect = size.x / size.y;
    const int2 p = int2(vertexPosition.xy);
    const int2 maxP = int2(width - 1, height - 1);
    resultColor = float4(1.0f, 0.0f, 0.0f, 1.0f);

    const float d = loadDepth(p);
    if (d >= 0.99999f) {
        return; // nothing drawn here (sky, background): no occlusion, and distance 0 tells the blur to leave it out
    }

#define VIEW_POSITION(q) viewPosition(q, size, aspect, n, f, tanHalf)
    const float3 P = VIEW_POSITION(p);
    resultColor.g = P.z;

    // Surface normal from the neighbours closest in depth on each axis (avoids smearing across silhouettes).
    const float3 Pr = VIEW_POSITION(min(p + int2(1, 0), maxP));
    const float3 Pl = VIEW_POSITION(max(p - int2(1, 0), int2(0, 0)));
    const float3 Pd = VIEW_POSITION(min(p + int2(0, 1), maxP));
    const float3 Pu = VIEW_POSITION(max(p - int2(0, 1), int2(0, 0)));
    const float3 dx = (abs(Pr.z - P.z) < abs(P.z - Pl.z)) ? (Pr - P) : (P - Pl);
    const float3 dy = (abs(Pd.z - P.z) < abs(P.z - Pu.z)) ? (Pd - P) : (P - Pu);
    float3 N = normalize(cross(dy, dx));
    if (dot(N, -P) < 0.0f) {
        N = -N;
    }

    // Sample disc: radius in pixels from the world radius at this distance.
    const float pixelsPerUnit = (size.y * 0.5f) / (P.z * tanHalf);
    const float discRadius = clamp(radius * pixelsPerUnit, 2.0f, size.y * 0.08f);

    // Per-pixel rotation (interleaved gradient noise); the blur pass smooths out the pattern it leaves.
    const float noise = frac(52.9829189f * frac(dot(float2(p), float2(0.06711056f, 0.00583715f))));
    const float rotation = noise * 6.2831853f;
    const float radius2 = radius * radius;

    // A sample has to lie clearly above the surface's plane to count: by a share of the distance (depth precision and
    // the steps between the flat polygons of low-poly models grow with it) and at a steeper angle than grazing, so
    // flat and gently curved surfaces (faces, walls seen at an angle) do not darken themselves.
    const float planeBias = 0.002f * P.z;
    const float angleBias = 0.2f;

    float occlusion = 0.0f;
    [unroll]
    for (int i = 0; i < SampleCount; i++) {
        // Spiral of samples, denser near the centre.
        const float t = (float(i) + 0.5f) / float(SampleCount);
        const float angle = t * 7.0f * 6.2831853f + rotation;
        const float2 offset = float2(cos(angle), sin(angle)) * (t * discRadius);
        const int2 q = clamp(p + int2(round(offset)), int2(0, 0), maxP);
        const float3 S = VIEW_POSITION(q);
        const float3 v = S - P;
        const float vv = dot(v, v);
        if (vv < radius2 && vv > 1e-4f) {
            const float vlen = sqrt(vv);
            const float cosine = (dot(v, N) - planeBias) / vlen;
            occlusion += max(0.0f, cosine - angleBias) * (1.0f - vlen / radius);
        }
    }
#undef VIEW_POSITION

    // Fades out for surfaces very close to the camera (within about two radii), where the samples spread over a few
    // large polygons and the result is mostly noise.
    const float closeFade = saturate((P.z - 0.5f * radius) / (1.5f * radius));
    resultColor.r = saturate(1.0f - strength * 4.0f * closeFade * occlusion / float(SampleCount));
}
