//
// RT64 - JFG PORT: depth-based distance fog for Jet Force Gemini (experimental).
//
// Drawn as a full-screen triangle (FullScreenVS) into the colour target with an alpha blend, right before the HUD of a
// world scene (see FramebufferRenderer::submitRasterScene), so the HUD stays untouched. The distance comes from the
// depth buffer with the game's own near and far planes (10, 15000), as in SsaoPS.hlsl. Nothing drawn (depth at the
// far plane: sky, backdrops, the cleared widescreen sides) gets no fog, so the sky keeps its own colours.
//
// Parameters (RasterParams): padding.x / padding.y = where the fog starts / is thickest (game units, as floats),
// padding.z = the most fog there is (0..1), screenScale.xy and screenOffset.x = the fog colour (0..1).
//

#include "shared/rt64_raster_params.h"

#include "FbRendererCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

#if defined(MULTISAMPLING)
Texture2DMS<float> gBackgroundDepth : register(t2, space3);
float loadDepth(int2 p) {
    return gBackgroundDepth.Load(p, 0);
}
#else
Texture2D<float> gBackgroundDepth : register(t2, space3);
float loadDepth(int2 p) {
    return gBackgroundDepth.Load(int3(p, 0));
}
#endif

static const float NearZ = 10.0f;
static const float FarZ = 15000.0f;

// Distance from the camera along the view direction.
float linearDepth(float d) {
    const float ndcZ = d * 2.0f - 1.0f;
    return (2.0f * NearZ * FarZ) / ((FarZ + NearZ) - ndcZ * (FarZ - NearZ));
}

void PSMain(
      in float4 vertexPosition : SV_Position
    , in float2 vertexUV : TEXCOORD0
    , [[vk::location(0)]] [[vk::index(0)]] out float4 resultColor : SV_TARGET0
)
{
    const float start = asfloat(gConstants.padding.x);
    const float end = max(asfloat(gConstants.padding.y), start + 1.0f);
    const float density = asfloat(gConstants.padding.z);
    const float3 color = float3(gConstants.screenScale.x, gConstants.screenScale.y, gConstants.screenOffset.x);

    const float d = loadDepth(int2(vertexPosition.xy));
    if (d >= 0.99999f) {
        resultColor = float4(color, 0.0f);
        return;
    }

    const float t = saturate((linearDepth(d) - start) / (end - start));
    resultColor = float4(color, density * t * t * (3.0f - 2.0f * t));
}
