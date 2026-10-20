//
// RT64 - JFG PORT: vignette for Jet Force Gemini.
//
// Drawn as a full-screen triangle (FullScreenVS) into the colour target with a multiplying blend (as SsaoPS.hlsl),
// right before the HUD of a world scene and after bloom (see FramebufferRenderer::submitRasterScene), so the HUD is
// not darkened.
//
// Parameters (RasterParams): padding.x = strength (0..1, how dark the corners get), padding.y = size (0..1, how far
// from the centre the darkening starts, 1 = the corners), padding.z = the colour target's width / height.
//

#include "shared/rt64_raster_params.h"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

void PSMain(
      in float4 vertexPosition : SV_Position
    , in float2 vertexUV : TEXCOORD0
    , [[vk::location(0)]] [[vk::index(0)]] out float4 resultColor : SV_TARGET0
)
{
    const float strength = asfloat(gConstants.padding.x);
    const float size = asfloat(gConstants.padding.y);
    const float aspect = asfloat(gConstants.padding.z);

    // Distance from the centre, round on screen: 0 in the middle, 1 in the corners.
    const float2 c = (vertexUV - 0.5f) * float2(aspect, 1.0f);
    const float r = length(c) / length(float2(0.5f * aspect, 0.5f));
    const float t = saturate((r - size) / max(1.0f - size, 0.001f));
    const float v = 1.0f - strength * t * t * (3.0f - 2.0f * t);
    resultColor = float4(v, v, v, 1.0f);
}
