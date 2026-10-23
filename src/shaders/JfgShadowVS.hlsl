//
// RT64 - JFG PORT: one face of a light's shadow cube (see JfgShadow.hlsli and FramebufferRenderer::recordJfgShadows).
//
// Drawn with the frame's shadow index list (the level's and the characters' triangles), whose indices are the RSP's
// vertex numbers, so the world position comes straight from RSPProcessCS's output. Depth only: no pixel shader.
// Parameters (RasterParams): padding.xyz = the light's position, shadow lift included (float bits), renderIndex = the
// face, screenScale = near, far.
//

#include "shared/rt64_raster_params.h"

#include "JfgShadow.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

ByteAddressBuffer jfgWorldPos : register(t68, space0);

void VSMain(in uint iVertexId : SV_VertexID, out float4 oPosition : SV_POSITION) {
    const float3 lightPos = asfloat(gConstants.padding);
    const float3 v = asfloat(jfgWorldPos.Load3(iVertexId * 16)) - lightPos;
    const float3 p = jfgShadowFace(v, gConstants.renderIndex);
    const float n = gConstants.screenScale.x;
    const float f = gConstants.screenScale.y;
    oPosition = float4(p.x, p.y, p.z * (f / (f - n)) - (f * n) / (f - n), p.z);
}
