//
// RT64
//

#include "shared/rt64_blender.h"
#include "shared/rt64_color_combiner.h"
#include "shared/rt64_jfg_light.h"
#include "shared/rt64_raster_params.h"

#include "Depth.hlsli"
#include "JfgShadow.hlsli"
#include "FbRendererCommon.hlsli"
#include "Library.hlsli"
#include "Random.hlsli"
#include "TextureSampler.hlsli"

[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

// JFG PORT: the frame's lights for the per-pixel level lighting (FramebufferRenderer::setJfgLights), padding.y of them.
StructuredBuffer<JfgLight> jfgLights : register(t69, space0);
// ... and how the frame's characters and objects are lit, by slot (FramebufferRenderer::setJfgLights).
StructuredBuffer<JfgObjectLight> jfgObjects : register(t72, space0);
// ... and what the lighting needs once per frame: camera, directional light, materials, shadow fade.
StructuredBuffer<JfgFrame> jfgFrame : register(t70, space0);
// ... and the shadow maps of the nearest lights: a cube each, one atlas row of six faces (JfgShadow.hlsli).
Texture2D<float> jfgShadowMap : register(t73, space0);

// JFG PORT: how much of a light reaches p, from its shadow cube (FramebufferRenderer::recordJfgShadows): the face that
// looks towards p, p's distance along it against the nearest surface the light sees there, over 3 x 3 texels for a
// soft edge. p is pushed a little towards the light first, along its surface, so a surface doesn't shadow itself. A
// light without a cube (beyond the nearest few) casts no shadows. darkness: how dark a full shadow is (0-1).
float jfgShadowVisibility(float3 p, float3 n, JfgLight light, float darkness) {
    const uint row = uint(light.shadow.x + 0.5f);
    if ((row == 0) || (darkness <= 0.0f)) {
        return 1.0f;
    }

    uint width, height;
    jfgShadowMap.GetDimensions(width, height);
    const int tile = int(width / 6);
    const float3 lightPos = light.posRadius.xyz + float3(0.0f, JFG_SHADOW_LIFT, 0.0f);
    const float3 toLight = lightPos - p;
    const float3 side = (dot(n, toLight) < 0.0f) ? -n : n;
    const float3 v = (p + side * 1.5f + normalize(toLight) * 1.0f) - lightPos;
    const uint face = jfgShadowFaceOf(v);
    const float3 f = jfgShadowFace(v, face);
    const float nearPlane = light.shadow.y;
    const float farPlane = light.shadow.z;
    if (f.z <= nearPlane) {
        return 1.0f;
    }

    // NDC x right, y up; the viewport puts y = +1 at the tile's top row.
    const float2 ndc = f.xy / f.z;
    const int2 texel = int2(float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f) * float(tile));
    const int2 origin = int2(int(face) * tile, int(row - 1) * tile);
    const float bias = 0.5f + f.z * 0.01f;
    float lit = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int2 t = clamp(texel + int2(dx, dy), int2(0, 0), int2(tile - 1, tile - 1));
            const float stored = jfgShadowMap.Load(int3(origin + t, 0));
            lit += ((f.z - bias) <= jfgShadowDistance(stored, nearPlane, farPlane)) ? 1.0f : 0.0f;
        }
    }

    return 1.0f - darkness * jfgFrame[0].shadowFade[row - 1] * (1.0f - lit / 9.0f);
}

// JFG PORT: the surface direction tilted by the texture's detail: the texture's brightness taken as a height (height =
// how many world units full white stands out), its change across the screen against how the surface spans the screen
// (dpdx, dpdy) gives the slope (surface gradient, Mikkelsen, "Bump Mapping Unparametrized Surfaces on the GPU").
float3 jfgDetailNormal(float3 n, float3 dpdx, float3 dpdy, float2 heightDeriv, float height) {
    const float3 r1 = cross(dpdy, n);
    const float3 r2 = cross(n, dpdx);
    const float det = dot(dpdx, r1);
    if ((height <= 0.0f) || (abs(det) < 1e-10f)) {
        return n;
    }

    const float3 grad = sign(det) * (heightDeriv.x * r1 + heightDeriv.y * r2) * height;
    return normalize(abs(det) * n - grad);
}

// JFG PORT: a light's shine (Blinn-Phong) on a surface facing nb, seen from the camera; shine = strength x how shiny
// the texture is there.
float3 jfgShine(float3 nb, float3 p, float3 toLight, float3 color, float shine, float sharpness) {
    if ((shine <= 0.0f) || (dot(nb, toLight) <= 0.0f)) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    const float3 v = normalize(jfgFrame[0].camera.xyz - p);
    const float3 h = normalize(toLight + v);
    return color * shine * pow(saturate(dot(nb, h)), sharpness);
}

// JFG PORT: the shadow of the frame light at pos (a character's own light direction, from the same torch or lamp).
float jfgShadowAt(float3 p, float3 n, float3 pos, uint lightCount, float darkness) {
    for (uint i = 0; i < lightCount; i++) {
        const JfgLight light = jfgLights[i];
        if (distance(light.posRadius.xyz, pos) < 4.0f) {
            return jfgShadowVisibility(p, n, light, darkness);
        }
    }

    return 1.0f;
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
    // JFG PORT: the texture's brightness one step along u and along v (the surface detail's slope, see below), sampled
    // as the texture itself is so it changes smoothly between texels.
    const bool jfgDetailOn = (jfgFrame[0].material.x > 0.0f) &&
        (((gConstants.padding.x & RASTER_JFG_LEVEL) != 0) || ((jfgFrame[0].extra.x > 0.0f) && (vertexWorldNorm.w > 0.5f)));
    float2 jfgTexGrad = float2(0.0f, 0.0f);
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
        if (jfgDetailOn) {
            const float3 toLum = float3(0.299f, 0.587f, 0.114f);
            const float lum = dot(texVal0.rgb, toLum);
            const float4 alongU = sampleTexture(otherMode, rp.flags, textureUV + float2(1.0f, 0.0f), ddxVertexUV, ddyVertexUV, rdpTile, gpuTile, false);
            const float4 alongV = sampleTexture(otherMode, rp.flags, textureUV + float2(0.0f, 1.0f), ddxVertexUV, ddyVertexUV, rdpTile, gpuTile, false);
            jfgTexGrad = float2(dot(alongU.rgb, toLum) - lum, dot(alongV.rgb, toLum) - lum);
        }
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
    
    // JFG PORT: what the lighting below needs from the texture (outside any branch, for the screen derivatives): its
    // brightness, taken as a height for the surface detail and as how shiny it is, and the surface's span on screen.
    const JfgFrame jfgF = jfgFrame[0];
    const float jfgTexLum = dot(texVal0.rgb, float3(0.299f, 0.587f, 0.114f));
    // The slope across the screen, from the slope along the texture (chain rule: the texture's own screen derivatives).
    const float2 jfgLumDeriv = float2(dot(jfgTexGrad, ddxVertexUV), dot(jfgTexGrad, ddyVertexUV));
    const float3 jfgDpdx = ddx(vertexWorldPos.xyz);
    const float3 jfgDpdy = ddy(vertexWorldPos.xyz);
    const float jfgCameraDistance = distance(jfgF.camera.xyz, vertexWorldPos.xyz);
    const float jfgHeight = jfgF.material.x * saturate(1.5f - jfgCameraDistance / 1500.0f); // fades out with distance
    const float jfgShineStrength = jfgF.material.y * lerp(0.25f, 1.0f, jfgTexLum);
    const float jfgSharpness = max(jfgF.material.z, 1.0f);
    float3 jfgSpecular = float3(0.0f, 0.0f, 0.0f);

    // Color combiner.
    float4 shadeColor = vertexColor;

    // JFG PORT: the game's dynamic lights on the level, per pixel. The game adds them to the level's vertex colours on
    // the CPU (trackLightAsm, skipped while this is on), so they go into the shade colour here, before the combiner
    // multiplies it with the texture: full strength within the inner radius, linear to nothing at the radius.
    // JFG PORT: characters and objects, lit as the game lights them but per pixel with the model's own normals
    // (vertexWorldNorm.w = object slot + 1): base colour (in the vertex colour) x (ambient, scaled with the level's,
    // + each of the game's directions by how much the surface faces it, shadowed by its light's shadow map when that
    // light has one), the directional light, then the frame's lights by the object's share of them.
    if ((vertexWorldNorm.w > 0.5f) && (vertexWorldPos.w > 0.5f)) {
        const JfgObjectLight object = jfgObjects[uint(vertexWorldNorm.w + 0.5f) - 1];
        const float3 n = normalize(vertexWorldNorm.xyz);
        const float3 nb = jfgDetailNormal(n, jfgDpdx, jfgDpdy, jfgLumDeriv, jfgHeight * jfgF.extra.x); // their own share
        const uint lightCount = gConstants.padding.y & 0xFFFF;
        const float darkness = float((gConstants.padding.x >> 16) & 0xFF) / 255.0f;
        // The first setting is the level's own light (its ambient and one direction, from the level header), so the
        // level ambient scales both; the others come from nearby lights (torches, lamps) and are left as they are.
        const float levelAmbient = float(gConstants.padding.x >> 24) / 128.0f;
        float3 brightness = object.params.x * levelAmbient;
        for (uint k = 0; k < uint(object.params.y + 0.5f); k++) {
            const float scale = (k == 0) ? levelAmbient : 1.0f;
            const float3 dir = normalize(object.dirs[k].xyz);
            const float shadow = (object.positions[k].w > 0.5f) ? jfgShadowAt(vertexWorldPos.xyz, n, object.positions[k].xyz, lightCount, darkness) : 1.0f;
            const float3 light = object.colors[k].rgb * object.dirs[k].w * scale * shadow;
            brightness += light * saturate(dot(nb, dir));
            jfgSpecular += jfgShine(nb, vertexWorldPos.xyz, dir, light, jfgShineStrength, jfgSharpness);
        }

        if (jfgF.sunDir.w > 0.5f) {
            const float3 sun = jfgF.sunColor.rgb * jfgF.sunColor.w;
            brightness += sun * saturate(dot(nb, jfgF.sunDir.xyz));
            jfgSpecular += jfgShine(nb, vertexWorldPos.xyz, jfgF.sunDir.xyz, sun, jfgShineStrength, jfgSharpness);
        }

        float3 added = float3(0.0f, 0.0f, 0.0f);
        const float share = object.params.z;
        if (share > 0.0f) {
            const float facing = saturate(asfloat(gConstants.padding.z));
            for (uint i = 0; i < lightCount; i++) {
                const JfgLight light = jfgLights[i];
                const float radius = light.posRadius.w;
                const float3 toLight = light.posRadius.xyz - vertexWorldPos.xyz;
                const float d = length(toLight);
                if (d < radius) {
                    const float inner = min(light.colorInner.w, radius - 1.0f);
                    const float3 l = toLight / max(d, 0.001f);
                    const float towards = saturate(dot(nb, l));
                    const float visibility = jfgShadowVisibility(vertexWorldPos.xyz, n, light, darkness);
                    const float3 lit = light.colorInner.rgb * saturate((radius - d) / (radius - inner)) * visibility;
                    added += lit * lerp(1.0f, towards, facing);
                    jfgSpecular += jfgShine(nb, vertexWorldPos.xyz, l, lit * share, jfgShineStrength, jfgSharpness);
                }
            }
        }

        shadeColor.rgb = saturate(shadeColor.rgb * (saturate(brightness) + added * share));
    }

    // The level: its own colours are its ambient (and hold its torches' painted glow), the frame's lights and the
    // directional light are added. Facing (padding.z): how much a surface turned away from a light gets less of it,
    // which the game never does; the surface detail tilts each light whatever facing is set to.
    if ((gConstants.padding.x & RASTER_JFG_LEVEL) && (vertexWorldPos.w > 0.5f)) {
        const float facing = saturate(asfloat(gConstants.padding.z));
        const uint lightCount = gConstants.padding.y & 0xFFFF;
        const float darkness = float((gConstants.padding.x >> 16) & 0xFF) / 255.0f;
        const float ambient = float(gConstants.padding.x >> 24) / 128.0f; // the level's own colours are its ambient light
        const float3 nb = jfgDetailNormal(jfgNormal, jfgDpdx, jfgDpdy, jfgLumDeriv, jfgHeight);
        float3 added = float3(0.0f, 0.0f, 0.0f);
        float painted = 1.0f; // how much of the level's own colours a shadow (and the detail) leaves
        for (uint i = 0; i < lightCount; i++) {
            const JfgLight light = jfgLights[i];
            const float radius = light.posRadius.w;
            const float3 toLight = light.posRadius.xyz - vertexWorldPos.xyz;
            const float d = length(toLight);
            if (d < radius) {
                const float inner = min(light.colorInner.w, radius - 1.0f);
                const float3 l = toLight / max(d, 0.001f);
                const float towards = saturate(dot(jfgNormal, l));
                const float relief = saturate(1.0f + dot(nb, l) - dot(jfgNormal, l)); // the detail's light and shade
                const float visibility = jfgShadowVisibility(vertexWorldPos.xyz, jfgNormal, light, darkness);
                const float reach = saturate((radius - d) / (radius - inner));
                const float3 lit = light.colorInner.rgb * reach * visibility;
                added += lit * lerp(1.0f, towards, facing) * relief;
                jfgSpecular += jfgShine(nb, vertexWorldPos.xyz, l, lit, jfgShineStrength, jfgSharpness);
                // The level's colours already hold the glow of its torches and lamps, so a shadow from one takes that
                // too (as much as the light reaches there, up to material.w of the colour), and the detail shades it.
                painted *= (1.0f - jfgF.material.w * reach * (1.0f - visibility)) * lerp(1.0f, relief, reach * 0.5f);
            }
        }

        if (jfgF.sunDir.w > 0.5f) {
            added += jfgF.sunColor.rgb * saturate(dot(nb, jfgF.sunDir.xyz));
            jfgSpecular += jfgShine(nb, vertexWorldPos.xyz, jfgF.sunDir.xyz, jfgF.sunColor.rgb, jfgShineStrength, jfgSharpness);
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
    resultColor.rgb += jfgSpecular; // JFG PORT: shine goes on top of the texture, not into it
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

    // JFG PORT: the lights' shadows as red, at full darkness, from every light that reaches the surface.
    if ((gConstants.padding.x & RASTER_JFG_DEBUG_SHADOW) && (vertexWorldPos.w > 0.5f)) {
        const uint lightCount = gConstants.padding.y & 0xFFFF;
        float shadow = 0.0f;
        for (uint i = 0; i < lightCount; i++) {
            const JfgLight light = jfgLights[i];
            if (distance(light.posRadius.xyz, vertexWorldPos.xyz) < light.posRadius.w) {
                shadow = max(shadow, 1.0f - jfgShadowVisibility(vertexWorldPos.xyz, jfgNormal, light, 1.0f));
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