//
// RT64 - JFG PORT: one face of a light's shadow cube (see JfgShadow.hlsli and FramebufferRenderer::recordJfgShadows).
//
// Drawn with the frame's shadow index list (the level's and the characters' triangles), whose indices are the RSP's
// vertex numbers, so the world position comes straight from RSPProcessCS's output. Depth only: no pixel shader.
// Parameters (RasterParams): padding.xyz = the light's position, shadow lift included (float bits), renderIndex = the
// face, screenScale = near, far.
//

#include "shared/rt64_jfg_light.h"
#include "shared/rt64_raster_params.h"

#include "JfgShadow.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

ByteAddressBuffer jfgWorldPos : register(t68, space0);
StructuredBuffer<JfgFrame> jfgFrame : register(t70, space0);

// The directional light's map (renderIndex = JFG_SUN_FACE): orthographic, padding.xyz = its centre, screenScale.x =
// half its size, screenScale.y = half its depth range.
void VSMain(in uint iVertexId : SV_VertexID, out float4 oPosition : SV_POSITION) {
    if (gConstants.renderIndex == JFG_SUN_FACE) {
        float3 right, up, forward;
        jfgSunBasis(jfgFrame[0].sunDir.xyz, right, up, forward);
        const float3 v = asfloat(jfgWorldPos.Load3(iVertexId * 16)) - asfloat(gConstants.padding);
        const float size = gConstants.screenScale.x, range = gConstants.screenScale.y;
        oPosition = float4(dot(v, right) / size, dot(v, up) / size, saturate((dot(v, forward) + range) / (2.0f * range)), 1.0f);
        return;
    }

    const float3 lightPos = asfloat(gConstants.padding);
    const float3 v = asfloat(jfgWorldPos.Load3(iVertexId * 16)) - lightPos;
    const float3 p = jfgShadowFace(v, gConstants.renderIndex);
    const float n = gConstants.screenScale.x;
    const float f = gConstants.screenScale.y;
    oPosition = float4(p.x, p.y, p.z * (f / (f - n)) - (f * n) / (f - n), p.z);
}
