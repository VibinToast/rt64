//
// RT64
//

#include "shared/rt64_blender.h"
#include "shared/rt64_color_combiner.h"
#include "shared/rt64_jfg_light.h"
#include "shared/rt64_raster_params.h"

#include "Depth.hlsli"
#include "FbRendererCommon.hlsli"
#include "Library.hlsli"
#include "Random.hlsli"
#include "TextureSampler.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

// JFG PORT: the frame's lights for the per-pixel level lighting (FramebufferRenderer::setJfgLights), padding.y of them.
StructuredBuffer<JfgLight> jfgLights : register(t69, space0);
// ... and the shadow casters (characters and objects): xyz = a sphere's centre, resting on the ground, w = its radius
// (worked out from the size of the game's own shadow for it).
StructuredBuffer<float4> jfgCasters : register(t70, space0);
// ... and how the frame's characters and objects are lit, by slot (FramebufferRenderer::setJfgLights).
StructuredBuffer<JfgObjectLight> jfgObjects : register(t72, space0);

// JFG PORT: closest approach of segments p1-q1 and p2-q2: the distance, and where it is along p1-q1 (0-1).
float2 jfgSegmentDistance(float3 p1, float3 q1, float3 p2, float3 q2) {
    const float3 d1 = q1 - p1;
    const float3 d2 = q2 - p2;
    const float3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    const float c = dot(d1, r);
    const float b = dot(d1, d2);
    const float denom = a * e - b * b;
    float s = (denom > 1e-6f) ? saturate((b * f - c * e) / denom) : 0.0f;
    float t = (b * s + f) / e;
    if (t < 0.0f) {
        t = 0.0f;
        s = (a > 1e-6f) ? saturate(-c / a) : 0.0f;
    }
    else if (t > 1.0f) {
        t = 1.0f;
        s = (a > 1e-6f) ? saturate((b - c) / a) : 0.0f;
    }

    return float2(length(p1 + d1 * s - (p2 + d2 * t)), s);
}

// JFG PORT: how much of a light reaching p is left after the casters between them. Each caster stands as a capsule
// as tall as a character (3 x its radius, 0.6 x its radius thick, from the ground up), so a low torch throws a long
// shadow; it blocks the part of the line from p to the light that passes within its thickness, softly over half of
// it either side. A caster the light sits in (a lamp or torch object with a shadow of its own) doesn't block it, nor
// one the pixel is on. The light is taken 40 units higher: torch lights sit at the bowl, as low as a character's
// waist, which made every shadow a thin streak to the edge of the light.
float jfgShadowVisibility(float3 p, float3 lightPos, uint casterCount, float darkness) {
    lightPos.y += 40.0f; // the game puts a torch's light at its bowl; shadows are thrown from the flame above it
    if (distance(p, lightPos) < 0.001f) {
        return 1.0f;
    }

    float visibility = 1.0f;
    for (uint c = 0; c < casterCount; c++) {
        const float4 caster = jfgCasters[c];
        const float thick = caster.w * 0.6f;
        const float ground = caster.y - caster.w;
        const float3 bottom = float3(caster.x, ground + thick, caster.z);
        const float3 top = float3(caster.x, ground + caster.w * 3.0f - thick, caster.z);
        if ((jfgSegmentDistance(lightPos, lightPos, bottom, top).x < thick * 1.5f) ||
            (jfgSegmentDistance(p, p, bottom, top).x < thick)) {
            continue;
        }

        const float2 hit = jfgSegmentDistance(p, lightPos, bottom, top);
        if ((hit.y <= 0.0f) || (hit.y >= 1.0f)) {
            continue; // closest behind the pixel or beyond the light
        }

        visibility *= 1.0f - darkness * (1.0f - smoothstep(thick * 0.5f, thick * 1.5f, hit.x));
    }

    return visibility;
}

#if defined(MULTISAMPLING)
Texture2DMS<float> gBackgroundDepth : register(t2, space3);

float sampleBackgroundDepth(int2 pixelPos, uint sampleCount) {
    float v = gBackgroundDepth.Load(pixelPos, 0);
    if (sampleCount >= 2) {
        v += gBackgroundDepth.Load(pixelPos, 1);
    }

    if (sampleCount >= 4) {
        v += gBackgroundDepth.Load(pixelPos, 2);
        v += gBackgroundDepth.Load(pixelPos, 3);
    }

    if (sampleCount >= 8) {
        v += gBackgroundDepth.Load(pixelPos, 4);
        v += gBackgroundDepth.Load(pixelPos, 5);
        v += gBackgroundDepth.Load(pixelPos, 6);
        v += gBackgroundDepth.Load(pixelPos, 7);
    }

    return v / float(sampleCount);
}
#elif defined(DYNAMIC_RENDER_PARAMS) || defined(SPEC_CONSTANT_RENDER_PARAMS) || defined(LIBRARY)
Texture2D<float> gBackgroundDepth : register(t2, space3);

float sampleBackgroundDepth(int2 pixelPos, uint sampleCount) {
    return gBackgroundDepth.Load(int3(pixelPos, 0));
}
#endif

LIBRARY_EXPORT bool RasterPS(const RenderParams rp, float4 vertexPosition, float2 vertexUV, float4 vertexSmoothColor, float4 vertexFlatColor,
    float4 vertexWorldPos, float4 vertexWorldNorm, bool isFrontFace, out float4 resultColor, out float4 resultAlpha) 
{
    const OtherMode otherMode = { rp.omL, rp.omH };
#if defined(DYNAMIC_RENDER_PARAMS)
    if ((otherMode.cycleType() != G_CYC_COPY) && renderFlagCulling(rp.flags) && isFrontFace) {
        return false;
    }
#endif
    
    const uint instanceIndex = instanceRenderIndices[gConstants.renderIndex].instanceIndex;
    const float4 vertexColor = renderFlagSmoothShade(rp.flags) ? vertexSmoothColor : float4(vertexFlatColor.rgb, vertexSmoothColor.a);

    // JFG PORT: the surface's direction from how the world position changes across the screen: one direction per
    // triangle (the level has no normals). Worked out here, outside any branch, as screen-space derivatives need.
    // ddy x ddx points towards the camera for every visible surface.
    const float3 jfgNormal = normalize(cross(ddy(vertexWorldPos.xyz), ddx(vertexWorldPos.xyz)));
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    const bool depthClampNear = renderFlagNoN(rp.flags);
    const bool depthDecal = (otherMode.zMode() == ZMODE_DEC);
    const bool zSourcePrim = (otherMode.zSource() == G_ZS_PRIM);
    int2 pixelPosSeed = floor(vertexPosition.xy);
    uint randomSeed = initRand(FrParams.frameCount, instanceIndex * pixelPosSeed.y * pixelPosSeed.x, 16); // TODO: Review seed.

    // The F3D family of microcodes has a bug where polygons are clipped earlier than expected despite the depth being computed
    // in a wider range of values. Not accounting for this can lead to geometry showing up that should otherwise be invisible.
    // This max range was determined by using a test ROM against an LLE implementation.
    // FIXME: This should be turned off for non-F3D microcodes.
    const bool simulateDepthClipF3D = true;
    const float MaxDepth = simulateDepthClipF3D && !renderFlagRect(rp.flags) && !zSourcePrim ? (1022.0f / 1024.0f) : 1.0f;
    
    // FIXME: This can be implemented by checking feature support for the API and embedding the depth bounds into the pipeline.
    const bool pipelineDepthBounds = false;
    
    // Handle no-nearclipping by clamping the minimum depth and manually clipping above the maximum.
    if (depthClampNear) {
        // Since depth clip is disabled on the PSO so near clip can be ignored, we manually clip any values above the allowed depth.
        if (vertexPosition.z > MaxDepth) {
            return false;
        }
    }
    // Do depth clipping manually on the fragment shader if pipeline's depth bounds are not being used due to lack of hardware support.
    else if (!pipelineDepthBounds) {
        if ((vertexPosition.z < 0.0f) || (vertexPosition.z > MaxDepth)) {
            return false;
        }
    }

    if (depthDecal) {
        // Sample the depth buffer for this pixel to compare for the decal check.
        int2 pixelPos = floor(vertexPosition.xy);
        uint sampleCount = 1U << renderFlagSampleCount(rp.flags);
        float surfaceDepth = sampleBackgroundDepth(pixelPos, sampleCount);

        // Calculate the decal depth tolerance based on the depth derivatives (or the prim dz value if prim depth source is enabled).
        float dz;
        if (zSourcePrim) {
            dz = instanceRDPParams[instanceIndex].primDepth.y;
        }
        else {
            dz = (abs(ddx(vertexPosition.z)) + abs(ddy(vertexPosition.z))) * FbParams.resolutionScale.y;
        }

        // Perform the decal depth tolerance check.
        const float DepthTolerance = max(CoplanarDepthTolerance(surfaceDepth), dz);
        const float pixelDepth = select(depthClampNear, max(vertexPosition.z, 0.0f), vertexPosition.z);
        if (abs(pixelDepth - surfaceDepth) > DepthTolerance) {
            return false;
        }
    }
    
    float2 ddxVertexUV = ddx(vertexUV);
    float2 ddyVertexUV = ddy(vertexUV);
    float2 lowResUV = vertexUV;
    if (!renderFlagUpscale2D(rp.flags)) {
        float2 screenPos = floor(vertexPosition.xy);
        screenPos.x += FbParams.horizontalMisalignment;
        lowResUV -= fmod(screenPos.xy, FbParams.resolutionScale.yy) * float2(ddxVertexUV.x, ddyVertexUV.y);
    }
    
    int tileIndex0 = 0;
    int tileIndex1 = 1;
    float lodFraction;
    float lodScale = 1.0f;
    if (!renderFlagUpscaleLOD(rp.flags)) {
        lodScale = FbParams.resolutionScale.y;
    }
    
    computeLOD(otherMode, instanceRenderIndices[gConstants.renderIndex].rdpTileCount, instanceRDPParams[instanceIndex].primLOD, lodScale, ddxVertexUV, ddyVertexUV, tileIndex0, tileIndex1, lodFraction);

    float4 texVal0 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 texVal1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (renderFlagUsesTexture0(rp.flags)) {
        const uint globalTileIndex = instanceRenderIndices[gConstants.renderIndex].rdpTileIndex + tileIndex0;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }
        
        const GPUTile gpuTile = GPUTiles[globalTileIndex];
        const float2 textureUV = gpuTileFlagHighRes(gpuTile.flags) ? vertexUV : lowResUV;
        texVal0 = sampleTexture(otherMode, rp.flags, textureUV, ddxVertexUV, ddyVertexUV, rdpTile, gpuTile, false);
    }
    
    if (renderFlagUsesTexture1(rp.flags)) {
        const bool oneCycleHardwareBug = (otherMode.cycleType() == G_CYC_1CYCLE);
        const uint globalTileIndex = instanceRenderIndices[gConstants.renderIndex].rdpTileIndex + (oneCycleHardwareBug ? tileIndex0 : tileIndex1);
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = oneCycleHardwareBug ? renderCMS0(rp.flags) : renderCMS1(rp.flags);
            rdpTile.cmt = oneCycleHardwareBug ? renderCMT0(rp.flags) : renderCMT1(rp.flags);
            rdpTile.nativeSampler = oneCycleHardwareBug ? renderFlagNativeSampler0(rp.flags) : renderFlagNativeSampler1(rp.flags);
        }
        
        const GPUTile gpuTile = GPUTiles[globalTileIndex];
        const float2 textureUV = gpuTileFlagHighRes(gpuTile.flags) ? vertexUV : lowResUV;
        const uint nativeSampler = renderFlagNativeSampler1(rp.flags);
        texVal1 = sampleTexture(otherMode, rp.flags, textureUV, ddxVertexUV, ddyVertexUV, rdpTile, gpuTile, oneCycleHardwareBug);
    }
    
    // Color combiner.
    float4 shadeColor = vertexColor;

    // JFG PORT: the game's dynamic lights on the level, per pixel. The game adds them to the level's vertex colours on
    // the CPU (trackLightAsm, skipped while this is on), so they go into the shade colour here, before the combiner
    // multiplies it with the texture: full strength within the inner radius, linear to nothing at the radius.
    // JFG PORT: characters and objects, lit as the game lights them but per pixel with the model's own normals
    // (vertexWorldNorm.w = object slot + 1): base colour (in the vertex colour) x (ambient, scaled with the level's,
    // + each of the game's directions by how much the surface faces it), then the frame's lights by the object's
    // share of them.
    if ((vertexWorldNorm.w > 0.5f) && (vertexWorldPos.w > 0.5f)) {
        const JfgObjectLight object = jfgObjects[uint(vertexWorldNorm.w + 0.5f) - 1];
        const float3 n = normalize(vertexWorldNorm.xyz);
        // The first setting is the level's own light (its ambient and one direction, from the level header), so the
        // level ambient scales both; the others come from nearby lights (torches, lamps) and are left as they are.
        const float levelAmbient = float(gConstants.padding.x >> 24) / 128.0f;
        float3 brightness = object.params.x * levelAmbient;
        for (uint k = 0; k < uint(object.params.y + 0.5f); k++) {
            const float scale = (k == 0) ? levelAmbient : 1.0f;
            brightness += object.colors[k].rgb * object.dirs[k].w * scale * saturate(dot(n, normalize(object.dirs[k].xyz)));
        }

        float3 added = float3(0.0f, 0.0f, 0.0f);
        const float share = object.params.z;
        if (share > 0.0f) {
            const uint lightCount = gConstants.padding.y & 0xFFFF;
            const uint casterCount = gConstants.padding.y >> 16;
            const float darkness = float((gConstants.padding.x >> 16) & 0xFF) / 255.0f;
            const float facing = saturate(asfloat(gConstants.padding.z));
            for (uint i = 0; i < lightCount; i++) {
                const JfgLight light = jfgLights[i];
                const float radius = light.posRadius.w;
                const float3 toLight = light.posRadius.xyz - vertexWorldPos.xyz;
                const float d = length(toLight);
                if (d < radius) {
                    const float inner = min(light.colorInner.w, radius - 1.0f);
                    const float towards = saturate(dot(n, toLight / max(d, 0.001f)));
                    const float visibility = (darkness > 0.0f) ? jfgShadowVisibility(vertexWorldPos.xyz, light.posRadius.xyz, casterCount, darkness) : 1.0f;
                    added += light.colorInner.rgb * saturate((radius - d) / (radius - inner)) * lerp(1.0f, towards, facing) * visibility;
                }
            }
        }

        shadeColor.rgb = saturate(shadeColor.rgb * (saturate(brightness) + added * share));
    }

    // Facing (padding.z): how much a surface turned away from a light gets less of it, which the game never does.
    if ((gConstants.padding.x & RASTER_JFG_LEVEL) && (vertexWorldPos.w > 0.5f)) {
        const float facing = saturate(asfloat(gConstants.padding.z));
        const uint lightCount = gConstants.padding.y & 0xFFFF;
        const uint casterCount = gConstants.padding.y >> 16;
        const float darkness = float((gConstants.padding.x >> 16) & 0xFF) / 255.0f;
        const float ambient = float(gConstants.padding.x >> 24) / 128.0f; // the level's own colours are its ambient light
        float3 added = float3(0.0f, 0.0f, 0.0f);
        float painted = 1.0f; // how much of the level's own colours a shadow leaves
        for (uint i = 0; i < lightCount; i++) {
            const JfgLight light = jfgLights[i];
            const float radius = light.posRadius.w;
            const float3 toLight = light.posRadius.xyz - vertexWorldPos.xyz;
            const float d = length(toLight);
            if (d < radius) {
                const float inner = min(light.colorInner.w, radius - 1.0f);
                const float towards = saturate(dot(jfgNormal, toLight / max(d, 0.001f)));
                const float visibility = (darkness > 0.0f) ? jfgShadowVisibility(vertexWorldPos.xyz, light.posRadius.xyz, casterCount, darkness) : 1.0f;
                const float reach = saturate((radius - d) / (radius - inner));
                added += light.colorInner.rgb * reach * lerp(1.0f, towards, facing) * visibility;
                // The level's colours already hold the glow of its torches and lamps, so a shadow from one takes that
                // too: as much as the light reaches there, up to three quarters of the colour.
                painted *= 1.0f - 0.75f * reach * (1.0f - visibility);
            }
        }

        shadeColor.rgb = saturate(shadeColor.rgb * ambient * painted + added);
    }
    float4 combinerColor;
    float alphaCompareValue;
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = instanceRDPParams[instanceIndex].primColor;
    ccInputs.shadeColor = shadeColor;
    ccInputs.envColor = instanceRDPParams[instanceIndex].envColor;
    ccInputs.keyCenter = instanceRDPParams[instanceIndex].keyCenter;
    ccInputs.keyScale = instanceRDPParams[instanceIndex].keyScale;
    ccInputs.lodFraction = lodFraction;
    ccInputs.primLodFrac = instanceRDPParams[instanceIndex].primLOD.x;
    ccInputs.noise = nextRand(randomSeed);
    ccInputs.K4 = (instanceRDPParams[instanceIndex].convertK[4] / 255.0f);
    ccInputs.K5 = (instanceRDPParams[instanceIndex].convertK[5] / 255.0f);
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);
    
#if 0
    // Alpha dither.
    // TODO: To avoid increasing the alpha values here, the only viable choice would be to drop the precision down to 5-bit.
    // Since we'd rather keep the full precision of the alpha channel from the texture, this step is ignored for now.
    if (otherMode.alphaDither() != G_AD_DISABLE) {
        uint rgbDither = (otherMode.rgbDither() >> G_MDSFT_RGBDITHER) & 0x3;
        uint alphaDither = (otherMode.alphaDither() >> G_MDSFT_ALPHADITHER) & 0x3;
        float alphaDitherFloat = (AlphaDitherValue(rgbDither, alphaDither, floor(vertexPosition.xy), randomSeed) / 255.0f);
        if (!otherMode.alphaCvgSel()) {
            combinerColor.a += alphaDitherFloat;
        }
        
        shadeColor.a += alphaDitherFloat;
        alphaCompareValue += alphaDitherFloat;
    }
#endif
    
    // Alpha compare.
    if (otherMode.alphaCompare() == G_AC_DITHER) {
        if (alphaCompareValue < nextRand(randomSeed)) {
            return false;
        }
    }
    else if (otherMode.alphaCompare() == G_AC_THRESHOLD) {
        if (alphaCompareValue < instanceRDPParams[instanceIndex].blendColor.a) {
            return false;
        }
    }
    
    // Compute coverage estimation.
    const bool usesHDR = renderFlagUsesHDR(rp.flags);
    const float cvgRange = usesHDR ? 65535.0f : 255.0f;
    float resultCvg = (8.0f / cvgRange) * (otherMode.cvgXAlpha() ? combinerColor.a : 1.0f);
    
    // Discard all pixels without coverage.
    const float CoverageThreshold = 1.0f / cvgRange;
    if (resultCvg < CoverageThreshold) {
        return false;
    }
    
    // Add the blender if it can be replicated with simple emulation.
    Blender::Inputs blInputs;
    blInputs.blendColor = instanceRDPParams[instanceIndex].blendColor;
    blInputs.fogColor = instanceRDPParams[instanceIndex].fogColor;
    blInputs.shadeAlpha = shadeColor.a;
    resultColor = Blender::run(otherMode, rp.flags, blInputs, combinerColor, false);
    resultAlpha = 1.0f;
    
    // When using alpha blending, we store the blending factor into the dedicated output so the main one can be used for coverage.
    const bool alphaBlend = (otherMode.cycleType() != G_CYC_COPY) && Blender::usesAlphaBlend(otherMode);
    if (alphaBlend) {
        resultAlpha.a = resultColor.a;
    }
    
    // Preserve the value in the destination.
    if (otherMode.cvgDst() == CVG_DST_SAVE) {
        resultColor.a = 0.0f;
    }
    // Write a full coverage value regardless of the computed coverage.
    else if (otherMode.cvgDst() == CVG_DST_FULL) {
        resultColor.a = 7.0f / cvgRange;
    }
    // Write the coverage value clamped to the full value allowed.
    else if (otherMode.cvgDst() == CVG_DST_CLAMP) {
        resultColor.a = min(resultCvg, 7.0f / cvgRange);
    }
    // Write out the computed coverage. It'll be added on wrap mode.
    else {
        resultColor.a = resultCvg;
    }
    
    // Add highlight color to the last step.
    uint highlightColorUint = instanceRenderIndices[gConstants.renderIndex].highlightColor;
    if (highlightColorUint > 0) {
        float4 highlightColor = RGBA32ToFloat4(highlightColorUint);
        resultColor = lerp(resultColor, highlightColor, highlightColor.a);
    }
    
#ifdef DYNAMIC_RENDER_PARAMS
    if (FrParams.viewUbershaders) {
        resultColor.rgb = lerp(resultColor.rgb, float3(1.0f, 0.0f, 0.0f), 0.5f);
    }
#endif

    // JFG PORT: RT64_JFG_DEBUG_WORLD=1: the world position as colour, repeating every 256 units; magenta where a draw
    // has none (raw triangles, rectangles).
    if (gConstants.padding.x & RASTER_JFG_DEBUG_WORLD) {
        resultColor.rgb = (vertexWorldPos.w > 0.5f) ? frac(vertexWorldPos.xyz / 256.0f) : float3(1.0f, 0.0f, 1.0f);
    }

    // JFG PORT: the surface direction as colour (x red, y green, z blue; up is green), magenta without a position.
    if (gConstants.padding.x & RASTER_JFG_DEBUG_NORMAL) {
        resultColor.rgb = (vertexWorldPos.w > 0.5f) ? (jfgNormal * 0.5f + 0.5f) : float3(1.0f, 0.0f, 1.0f);
    }

    // JFG PORT: the casters' shadows as red, at full darkness, from every light that reaches the surface.
    if ((gConstants.padding.x & RASTER_JFG_DEBUG_SHADOW) && (vertexWorldPos.w > 0.5f)) {
        const uint lightCount = gConstants.padding.y & 0xFFFF;
        const uint casterCount = gConstants.padding.y >> 16;
        float shadow = 0.0f;
        for (uint i = 0; i < lightCount; i++) {
            const JfgLight light = jfgLights[i];
            if (distance(light.posRadius.xyz, vertexWorldPos.xyz) < light.posRadius.w) {
                shadow = max(shadow, 1.0f - jfgShadowVisibility(vertexWorldPos.xyz, light.posRadius.xyz, casterCount, 1.0f));
            }
        }

        resultColor.rgb = lerp(resultColor.rgb, float3(1.0f, 0.0f, 0.0f), shadow);
    }
    
    return true;
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
void PSMain(
      in float4 vertexPosition : SV_POSITION
    , in float2 vertexUV : TEXCOORD
    , in float4 vertexSmoothColor : COLOR0
#if defined(DYNAMIC_RENDER_PARAMS) || defined(VERTEX_FLAT_COLOR)
    , nointerpolation in float4 vertexFlatColor : COLOR1
#endif
    , in float4 vertexWorldPos : TEXCOORD1 // JFG PORT: last, as in RasterVS
    , in float4 vertexWorldNorm : TEXCOORD2
#if defined(DYNAMIC_RENDER_PARAMS)
    , bool isFrontFace : SV_IsFrontFace
#endif
    , [[vk::location(0)]] [[vk::index(0)]] out float4 pixelColor : SV_TARGET0
    , [[vk::location(0)]] [[vk::index(1)]] out float4 pixelAlpha : SV_TARGET1
)
{
#if !defined(DYNAMIC_RENDER_PARAMS)
#if !defined(VERTEX_FLAT_COLOR)
    float4 vertexFlatColor = 0.0f;
#endif
    bool isFrontFace = false;
#endif
    float4 resultColor;
    float4 resultAlpha;
    float resultDepth;
    if (!RasterPS(getRenderParams(), vertexPosition, vertexUV, vertexSmoothColor, vertexFlatColor, vertexWorldPos, vertexWorldNorm, isFrontFace, resultColor, resultAlpha)) {
        discard;
    }

    pixelColor = resultColor;
    pixelAlpha = resultAlpha;
}
#endif