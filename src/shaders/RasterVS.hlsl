//
// RT64
//

#include "shared/rt64_other_mode.h"
#include "shared/rt64_raster_params.h"
#include "shared/rt64_render_flags.h"

#include "FbRendererCommon.hlsli"
#include "Library.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

// JFG PORT: the world position of every vertex the RSP processed (RSPProcessCS, the workload's jfgWorldPosBuffer,
// float4 per vertex; FramebufferRendererDescriptorCommonSet::jfgWorldPos). Indexed draws read it by SV_VertexID.
ByteAddressBuffer jfgWorldPos : register(t68, space0);
// ... and a character's normal in the world (jfgWorldNormBuffer; w = object slot + 1, 0 = not a character, 255 = the
// level's smooth direction).
ByteAddressBuffer jfgWorldNorm : register(t71, space0);

LIBRARY_EXPORT void RasterVS(const RenderParams rp, in float4 iPosition, in float2 iUV, in float4 iColor, in uint iVertexId, out float4 oPosition, out float2 oUV, out float4 oSmoothColor, out float4 oFlatColor, out float4 oWorldPos, out float4 oWorldNorm) {
    float4 ndcPos = iPosition;
    
    // Skip any sort of transformation on the coordinates when rendering rects.
    if (!renderFlagRect(rp.flags)) {
        ndcPos.xy -= float2(FbParams.resolution / 2);
        ndcPos.xy /= float2(FbParams.resolution.x / 2.0f, FbParams.resolution.y / -2.0f);
        ndcPos.xyz *= ndcPos.w;
    }
    
    // Apply screen scale and offset.
    ndcPos.xy = (ndcPos.xy * gConstants.screenScale) + gConstants.screenOffset * ndcPos.w;
    
    // Output a fixed depth value for the entire triangle.
    const OtherMode otherMode = { rp.omL, rp.omH };
    const bool copyMode = (otherMode.cycleType() == G_CYC_COPY);
    const bool zSourcePrim = (otherMode.zSource() == G_ZS_PRIM);
    if (!copyMode && zSourcePrim) {
        const uint instanceIndex = instanceRenderIndices[gConstants.renderIndex].instanceIndex;
        ndcPos.z = instanceRDPParams[instanceIndex].primDepth.x * ndcPos.w;
    }

    oPosition = ndcPos;
    oUV = iUV;
    oSmoothColor = iColor;
    oFlatColor = iColor;

    // JFG PORT: w = 1 marks a valid world position (indexed draws only).
    oWorldPos = float4(0.0f, 0.0f, 0.0f, 0.0f);
    oWorldNorm = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (gConstants.padding.x & RASTER_JFG_INDEXED) {
        oWorldPos = float4(asfloat(jfgWorldPos.Load3(iVertexId * 16)), 1.0f);
        oWorldNorm = asfloat(jfgWorldNorm.Load4(iVertexId * 16));
    }
}

#if defined(DYNAMIC_RENDER_PARAMS)
RenderParams getRenderParams() {
    uint instanceIndex = instanceRenderIndices[gConstants.renderIndex].instanceIndex;
    return DynamicRenderParams[instanceIndex];
}
#elif defined(SPEC_CONSTANT_RENDER_PARAMS)
#   include "RenderParamsSpecConstants.hlsli"
#endif

#if defined(DYNAMIC_RENDER_PARAMS) || defined(SPEC_CONSTANT_RENDER_PARAMS)
void VSMain(
    in float4 iPosition : POSITION
    , in float2 iUV : TEXCOORD
    , in float4 iColor : COLOR
    , in uint iVertexId : SV_VertexID
    , out float4 oPosition : SV_POSITION
    , out float2 oUV : TEXCOORD
    , out float4 oSmoothColor : COLOR0
#if defined(DYNAMIC_RENDER_PARAMS) || defined(VERTEX_FLAT_COLOR)
    , out float4 oFlatColor : COLOR1
#endif
    , out float4 oWorldPos : TEXCOORD1 // JFG PORT: last, so the interface locations match RasterPS's
    , out float4 oWorldNorm : TEXCOORD2
)
{
#if !defined(DYNAMIC_RENDER_PARAMS) && !defined(VERTEX_FLAT_COLOR)
    float4 oFlatColor;
#endif
    RasterVS(getRenderParams(), iPosition, iUV, iColor, iVertexId, oPosition, oUV, oSmoothColor, oFlatColor, oWorldPos, oWorldNorm);
}
#endif