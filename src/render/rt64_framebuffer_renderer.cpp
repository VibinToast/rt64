//
// RT64
//

#include "rt64_framebuffer_renderer.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unordered_set>

#include "stb/stb_image.h"

#include "../include/rt64_extended_gbi.h"
#include "shared/rt64_jfg_bloom.h"

#include "common/rt64_elapsed_timer.h"
#include "common/rt64_math.h"
#include "hle/rt64_color_converter.h"
#include "gbi/rt64_f3d.h"
#include "shared/rt64_framebuffer_params.h"
#include "shared/rt64_raster_params.h"

#include "rt64_descriptor_sets.h"
#include "rt64_render_worker.h"

// TODO: Move to shared.

namespace interop {
    struct BicubicCB {
        uint2 InputResolution;
        uint2 OutputResolution;
    };

    struct HistogramAverageCB {
        uint pixelCount;
        float minLuminance;
        float luminanceRange;
        float timeDelta;
        float tau;
    };

    struct HistogramSetCB {
        float luminanceValue;
    };

    struct LuminanceHistogramCB {
        uint inputWidth;
        uint inputHeight;
        float minLuminance;
        float oneOverLuminanceRange;
    };

    struct TextureCB {
        uint2 TextureSize;
        float2 TexelSize;
    };
};

// JFG PORT: screen-space ambient occlusion settings, written by the port's options (src/jfg/jfg_options.cpp, push_render_options).
std::atomic<int> rt64_jfg_ssao{ 0 };
std::atomic<float> rt64_jfg_ssao_strength{ 1.0f };
std::atomic<float> rt64_jfg_ssao_radius{ 150.0f };
std::atomic<int> rt64_jfg_bloom{ 0 };
std::atomic<float> rt64_jfg_bloom_intensity{ 0.6f };
std::atomic<float> rt64_jfg_bloom_threshold{ 0.7f };
std::atomic<float> rt64_jfg_bloom_radius{ 6.0f };
std::atomic<int> rt64_jfg_paused{ 0 }; // 1 while the game's pause screen is up (set by the game side)
std::atomic<float> rt64_jfg_fov_tan{ 0.0f }; // tan(half the game's vertical field of view), 0 = not known yet (SsaoPS.hlsl)
std::atomic<int> rt64_jfg_fog{ 0 };            // depth-based fog (FogPS.hlsl), experimental
std::atomic<int> rt64_jfg_fog_blocked{ 0 };    // 1 while the game's own fog is within the far plane (set by the game side)
std::atomic<uint32_t> rt64_jfg_fog_drawn{ 0 };     // scenes the depth-based fog was drawn in (for the F9 menu's status line)
std::atomic<uint32_t> rt64_jfg_fog_game_scenes{ 0 }; // world scenes whose depth-writing draws blend with the game's fog colour
std::atomic<float> rt64_jfg_fog_start{ 1500.0f };
std::atomic<float> rt64_jfg_fog_end{ 9000.0f };
std::atomic<float> rt64_jfg_fog_density{ 0.5f };
std::atomic<uint32_t> rt64_jfg_fog_color{ 0xA0B4C8u }; // 0xRRGGBB
std::atomic<int> rt64_jfg_vignette{ 0 };       // vignette (VignettePS.hlsl)
std::atomic<int> rt64_jfg_pixel_lights{ 0 };   // the game's lights on the level per pixel (RasterPS, src/jfg/jfg_lighting.cpp)
std::atomic<int> rt64_jfg_debug_world{ 0 };    // show the world position as colour (RasterPS; F9 Lighting tab, RT64_JFG_DEBUG_WORLD)
std::atomic<int> rt64_jfg_debug_shadow{ 0 };   // paint the casters' shadows red (RasterPS; F9 Lighting tab)
std::atomic<int> rt64_jfg_debug_normal{ 0 };   // show the surface direction as colour (RasterPS; F9 Lighting tab)
std::atomic<float> rt64_jfg_facing{ 0.0f };    // how much the level lighting depends on facing the light, 0-1 (RasterPS)
std::atomic<float> rt64_jfg_shadows{ 0.0f };   // how dark characters' and objects' shadows from the lights are, 0-1 (RasterPS)
std::atomic<int> rt64_jfg_shadow_lights{ 4 };  // how many of the nearest lights get a shadow map, 0-4 (recordJfgShadows)
std::atomic<int> rt64_jfg_shadow_size{ 512 };  // each shadow map face's size in texels
std::atomic<int> rt64_jfg_material_files{ 0 };  // material maps found in materials/ (for the port's lighting tool)
std::atomic<int> rt64_jfg_material_loaded{ 0 }; // ... and loaded so far

// JFG PORT: material maps decoded by their loading threads, waiting to be uploaded by the renderer. Not members: a
// loading thread may still finish after the renderer is gone.
namespace {
    struct JfgMaterialImage {
        uint64_t hash = 0;
        int kind = 0; // 0 normal, 1 specular, 2 emission
        std::vector<uint8_t> rgba;
        uint32_t width = 0, height = 0;
    };

    std::mutex jfgMaterialImageMutex;
    std::vector<JfgMaterialImage> jfgMaterialImages;
}

// JFG PORT: the lighting settings that go to the shaders each frame (src/jfg/jfg_lighting.cpp), as floats:
//   0-2 direction towards the directional light, 3 on (0/1), 4-6 its colour x intensity, 7 how much it lights
//   characters, 8 surface detail height, 9 shine strength, 10 shine sharpness, 11 how much a shadow takes from the
//   level's painted glow, 12 share of the surface detail on characters and objects, 13 the directional light casts
//   shadows (0/1), 14 the size of the area around the view it shadows, 15 how much authored material maps count,
//   16 the level's smooth surface directions on (0/1), 17 shine strength on characters, 18 its sharpness.
static std::mutex jfgParamsMutex;
uint32_t rt64_jfg_glow_for(uint64_t hash); // src/hle/rt64_state.cpp
static float jfgParams[20] = { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 16.0f, 0.75f };
void rt64_jfg_lighting_params(const float *params, uint32_t count) {
    std::lock_guard<std::mutex> lock(jfgParamsMutex);
    memcpy(jfgParams, params, sizeof(float) * std::min(count, 20U));
}
std::atomic<float> rt64_jfg_level_ambient{ 1.0f }; // a scale on the level's own colours, its ambient light (RasterPS)
std::atomic<float> rt64_jfg_vignette_strength{ 0.4f };
std::atomic<float> rt64_jfg_vignette_size{ 0.45f };

namespace RT64 {
    // Helper functions.
    
    RenderRect convertFixedRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, int32_t horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        if (!rect.isNull()) {
            auto computeOrigin = [=](uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return std::lround(((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage));
                }
                else {
                    return fbWidth / 2L;
                }
            };

            auto correctMisalignment = [=](int32_t coord, uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return int32_t(coord - (coord % std::lround(resScale[1]))) - horizontalMisalignment;
                }
                else {
                    return coord;
                }
            };

            int32_t left = static_cast<int32_t>(std::floor((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x));
            int32_t right = static_cast<int32_t>(std::ceil((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x));
            int32_t top = lround(rect.top(true) * resScale.y);
            int32_t bottom = lround(rect.bottom(true) * resScale.y);
            left = correctMisalignment(left, leftOrigin);
            right = correctMisalignment(right, rightOrigin);
            return RenderRect(left, top, right, bottom);
        }
        else {
            return RenderRect(0, 0, 0, 0);
        }
    }
    
    RenderViewport convertViewportRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, float horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        auto computeOrigin = [=](uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return ((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage);
            }
            else {
                return float(fbWidth) / 2;
            }
        };
        
        auto correctMisalignment = [=](float coord, uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return (coord - std::fmod(coord, resScale[1])) - horizontalMisalignment;
            }
            else {
                return coord;
            }
        };
        
        float left = std::round((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x);
        float right = std::round((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x);
        float top = std::round(rect.top(true) * resScale.y);
        float bottom = std::round(rect.bottom(true) * resScale.y);
        left = correctMisalignment(left, leftOrigin);
        right = correctMisalignment(right, rightOrigin);
        return RenderViewport(left, top, right - left, bottom - top);
    }

    hlslpp::float3 viewPositionFrom(hlslpp::float4x4 viewI) {
        return viewI[3].xyz;
    }

    hlslpp::float3 viewDirectionFrom(hlslpp::float4x4 viewI) {
        return hlslpp::normalize(viewI[2].xyz);
    }

    RenderColor toRenderColor(hlslpp::float4 v) {
        return { v.x, v.y, v.z, v.w };
    }

    static RenderRect viewportScissorIntersection(const RenderViewport &viewport, const RenderRect &scissor) {
        return RenderRect{
            std::max(static_cast<int32_t>(std::floor(viewport.x)), scissor.left),
            std::max(static_cast<int32_t>(std::floor(viewport.y)), scissor.top),
            std::min(static_cast<int32_t>(std::ceil(viewport.x + viewport.width)), scissor.right),
            std::min(static_cast<int32_t>(std::ceil(viewport.y + viewport.height)), scissor.bottom)
        };
    }

    // RasterScene

    RasterScene::RasterScene() { }

    // FramebufferRenderer
    
    FramebufferRenderer::FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);

        this->shaderLibrary = shaderLibrary;

        frameParams.frameCount = 0;
        frameParams.viewUbershaders = false;
        frameParams.ditherNoiseStrength = 1.0f;

        shaderUploader = std::make_unique<BufferUploader>(worker->device);
        descCommonSet = std::make_unique<FramebufferRendererDescriptorCommonSet>(shaderLibrary->samplerLibrary, worker->device->getCapabilities().raytracing, worker->device);

#   if RT_ENABLED
        if (rtSupport) {
            this->rtSupport = rtSupport;
            rtResources = std::make_unique<RaytracingResources>(worker, graphicsAPI);
    }
#   endif
    }

    FramebufferRenderer::~FramebufferRenderer() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
    }
    
    void FramebufferRenderer::resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling) {
        instanceDrawCallVector.clear();
        hitGroupVector.clear();
        bloomDescriptorSets.clear(); // JFG PORT: the previous workload has finished on the GPU by now
        bloomBuffersRetired.clear();
        bloomSnapshotRetired.clear();
        jfgShadowRetired.clear(); // JFG PORT
        jfgShadowRetiredViews.clear();
        jfgShadowRetiredFramebuffers.clear();
        jfgShadowIndices.clear();
        jfgFrame++;
        renderIndicesVector.clear();
        rspSmoothNormalVector.clear();
        frameParams.viewUbershaders = ubershadersVisible;
        frameParams.ditherNoiseStrength = ditherNoiseStrength;
        framebufferCount = 0;

        // Create dummy color target if it hasn't been created yet.
        if (dummyColorTarget == nullptr) {
            RenderTextureDesc dummyColorDesc = RenderTextureDesc::ColorTarget(4, 4, RenderTarget::colorBufferFormat(shaderLibrary->usesHDR), multisampling);
            dummyColorTarget = worker->device->createTexture(dummyColorDesc);
            dummyColorTarget->setName("Framebuffer Renderer Color Dummy");
            dummyColorTargetView = dummyColorTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyColorDesc.format));
            dummyColorTargetTransitioned = false;
        }

        // Create dummy depth target if it hasn't been created yet.
        if (dummyDepthTarget == nullptr) {
            RenderTextureDesc dummyDepthDesc = RenderTextureDesc::DepthTarget(4, 4, RenderFormat::D32_FLOAT, multisampling);
            dummyDepthTarget = worker->device->createTexture(dummyDepthDesc);
            dummyDepthTarget->setName("Framebuffer Renderer Depth Dummy");
            dummyDepthTargetView = dummyDepthTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyDepthDesc.format));
            dummyDepthTargetTransitioned = false;
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture) {
        assert(rtResources != nullptr);
        assert(rtShaderCache != nullptr);
        assert(blueNoiseTexture != nullptr);

        const int32_t stateIndex = rtShaderCache->getActiveState();
        rtState = &rtShaderCache->states[stateIndex];
        rtPipelineLayout = rtShaderCache->pipelineLayout.get();
        rtResources->resetBottomLevelAS();

        this->blueNoiseTexture = blueNoiseTexture;
    }
#endif

    void FramebufferRenderer::updateTextureCache(TextureCache *textureCache) {
        const std::unique_lock<std::mutex> textureMapLock(textureCache->textureMapMutex);
        textureCacheVersions = textureCache->textureMap.versions;
        textureCacheTextures = textureCache->textureMap.textures;
        textureCacheTextureReplacements = textureCache->textureMap.textureReplacements;
        textureCacheFreeSpaces = textureCache->textureMap.freeSpaces;
        textureCacheSize = static_cast<uint32_t>(textureCacheTextures.size());
        textureCacheGlobalVersion = textureCache->textureMap.globalVersion;
        textureCacheReplacementMapEnabled = textureCache->textureMap.replacementMapEnabled;
        scanJfgMaterials(textureCache); // JFG PORT: first time, and when the packs change
        dynamicTextureViewVector.clear();
        dynamicTextureBarrierVector.clear();
    }

    void FramebufferRenderer::createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, 
        TextureCache *textureCache, uint64_t submissionFrame)
    {
        for (uint32_t i = 0; i < callTileCount; i++) {
            const DrawCallTile &callTile = callTiles[i];
            if (!callTile.valid) {
                continue;
            }
            
            interop::GPUTile &gpuTile = dstGPUTiles[i];
            if (callTile.tileCopyUsed) {
                const auto &it = fbManager->tileCopies.find(callTile.tmemHashOrID);
                if (it != fbManager->tileCopies.end()) {
                    const FramebufferManager::TileCopy &tileCopy = it->second;
                    gpuTile.tcScale.x = static_cast<float>(tileCopy.usedWidth) / static_cast<float>(callTile.tileCopyWidth);
                    gpuTile.tcScale.y = static_cast<float>(tileCopy.usedHeight) / static_cast<float>(callTile.tileCopyHeight);
                    gpuTile.ulScale.x = tileCopy.ulScaleS ? gpuTile.tcScale.x : 1.0f;
                    gpuTile.ulScale.y = tileCopy.ulScaleT ? gpuTile.tcScale.y : 1.0f;
                    gpuTile.texelShift = tileCopy.texelShift;
                    gpuTile.texelMask = tileCopy.texelMask;
                    gpuTile.textureIndex = getTextureIndex(tileCopy);
                    gpuTile.textureDimensions = interop::float3(float(tileCopy.textureWidth), float(tileCopy.textureHeight), 1.0f);
                    gpuTile.flags.alphaIsCvg = !callTile.reinterpretTile;
                    gpuTile.flags.highRes = true;
                    gpuTile.flags.fromCopy = true;
                    gpuTile.flags.rawTMEM = false;
                    gpuTile.flags.hasMipmaps = false;
                    gpuTile.flags.jfgSoft = tileCopy.jfgSmallSource; // JFG PORT
                    gpuTile.flags.jfgGlow = 0;
                    gpuTile.jfgMaterial = 0;
                }
            }
            else {
                // Retrieve the texture from the cache or use a blank texture if not found.
                uint32_t textureIndex = 0;
                bool textureReplaced = false;
                bool hasMipmaps = false;
                bool shiftedByHalf = false;
                textureCache->useTexture(callTile.tmemHashOrID, submissionFrame, textureIndex, gpuTile.tcScale, gpuTile.textureDimensions, textureReplaced, hasMipmaps, shiftedByHalf);

                // Describe the GPU tile for a regular texture.
                gpuTile.ulScale.x = gpuTile.tcScale.x;
                gpuTile.ulScale.y = gpuTile.tcScale.y;
                gpuTile.texelShift = { 0, 0 };
                gpuTile.texelMask = { UINT_MAX, UINT_MAX };
                gpuTile.textureIndex = textureIndex;
                gpuTile.flags.alphaIsCvg = false;
                gpuTile.flags.highRes = textureReplaced;
                gpuTile.flags.fromCopy = false;
                gpuTile.flags.rawTMEM = !textureReplaced && callTile.rawTMEM;
                gpuTile.flags.hasMipmaps = hasMipmaps;
                gpuTile.flags.shiftedByHalf = shiftedByHalf;
                gpuTile.flags.jfgSoft = false; // JFG PORT
                gpuTile.flags.jfgGlow = rt64_jfg_glow_for(callTile.tmemHashOrID);
                gpuTile.jfgMaterial = jfgMaterialFor(callTile.tmemHashOrID, textureCache);
            }
        }
    }

    uint32_t FramebufferRenderer::getDestinationIndex() {
        uint32_t dstIndex;
        if (textureCacheFreeSpaces.empty()) {
            dstIndex = textureCacheSize++;
        }
        else {
            dstIndex = textureCacheFreeSpaces.back();
            textureCacheFreeSpaces.pop_back();
        }

        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(RenderTarget *renderTarget) {
        assert(renderTarget != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ renderTarget->getResolvedTexture(), dstIndex, renderTarget->getResolvedTextureView()});
        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(const FramebufferManager::TileCopy &tileCopy) {
        assert(tileCopy.texture != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ tileCopy.texture.get(), dstIndex, nullptr });
        dynamicTextureBarrierVector.emplace_back(RenderTextureBarrier(tileCopy.texture.get(), RenderTextureLayout::SHADER_READ));
        return dstIndex;
    }
    
    void FramebufferRenderer::updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        assert(worker != nullptr);
        assert(drawBuffers != nullptr);
        
        // JFG PORT: with material maps the set spans the whole array (they sit at its top, JFG_MATERIAL_BASE).
        const bool jfgMaterials = (rt64_jfg_material_files.load(std::memory_order_relaxed) > 0);
        const uint32_t jfgSetSize = jfgMaterials ? uint32_t(FramebufferRendererDescriptorTextureSet::UpperRange) : ((textureCacheSize + 1) * 3) / 2;
        const bool createSet = (descTextureSet == nullptr) || (descTextureSet->textureCacheSize < (textureCacheSize + 1)) ||
            (jfgMaterials && (descTextureSet->textureCacheSize < uint32_t(FramebufferRendererDescriptorTextureSet::UpperRange)));
        if (createSet) {
            descTextureSet = std::make_unique<FramebufferRendererDescriptorTextureSet>(worker->device, std::max(jfgSetSize, ((textureCacheSize + 1) * 3) / 2));
        }

        {
            std::lock_guard<std::mutex> lock(jfgMaterialMutex);
            if (createSet) {
                jfgMaterialPending.clear();
                for (uint32_t i = 0; i < uint32_t(jfgMaterialTextures.size()); i++) {
                    jfgMaterialPending.push_back(i);
                }
            }

            for (uint32_t slot : jfgMaterialPending) {
                if ((JFG_MATERIAL_BASE + slot) < descTextureSet->textureCacheSize) {
                    descTextureSet->setTexture(JFG_MATERIAL_BASE + slot, jfgMaterialTextures[slot]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
            }

            jfgMaterialPending.clear();
        }

        if (createSet || (descriptorTextureReplacementMapEnabled != textureCacheReplacementMapEnabled)) {
            descriptorTextureVersions.clear();
            descriptorTextureGlobalVersion = 0;
            descriptorTextureReplacementMapEnabled = textureCacheReplacementMapEnabled;
        }

        // JFG PORT: the world position of every vertex for the raster shaders (RasterVS).
        descCommonSet->setBuffer(descCommonSet->jfgWorldPos, outputBuffers->jfgWorldPosBuffer.buffer.get(), outputBuffers->jfgWorldPosBuffer.allocatedSize);
        descCommonSet->setBuffer(descCommonSet->jfgLights, jfgLightsBuffer.get(), sizeof(interop::JfgLight) * jfgLights.size(), RenderBufferStructuredView(sizeof(interop::JfgLight)));
        descCommonSet->setBuffer(descCommonSet->jfgFrame, jfgFrameBuffer.get(), sizeof(interop::JfgFrame) * jfgFrameData.size(), RenderBufferStructuredView(sizeof(interop::JfgFrame)));
        descCommonSet->setBuffer(descCommonSet->jfgWorldNorm, outputBuffers->jfgWorldNormBuffer.buffer.get(), outputBuffers->jfgWorldNormBuffer.allocatedSize);
        descCommonSet->setBuffer(descCommonSet->jfgObjects, jfgObjectsBuffer.get(), sizeof(interop::JfgObjectLight) * jfgObjects.size(), RenderBufferStructuredView(sizeof(interop::JfgObjectLight)));
        if (jfgShadowAtlas != nullptr) {
            descCommonSet->setTexture(descCommonSet->jfgShadowMap, jfgShadowAtlas.get(), RenderTextureLayout::DEPTH_READ, jfgShadowAtlasView.get());
        }

#   if RT_ENABLED
        if (raytracingEnabled) {
            descCommonSet->setBuffer(descCommonSet->posBuffer, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->normBuffer, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->velBuffer, outputBuffers->worldVelBuffer.buffer.get(), outputBuffers->worldVelBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->genTexCoordBuffer, outputBuffers->genTexCoordBuffer.buffer.get(), outputBuffers->genTexCoordBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->shadedColBuffer, outputBuffers->shadedColBuffer.buffer.get(), outputBuffers->shadedColBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcFogIndices, drawBuffers->fogIndicesBuffer.get(), drawBuffers->fogIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightIndices, drawBuffers->lightIndicesBuffer.get(), drawBuffers->lightIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightCounts, drawBuffers->lightCountsBuffer.get(), drawBuffers->lightCountsBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->indexBuffer, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->RSPFogVector, drawBuffers->rspFogBuffer.get(), drawBuffers->rspFogBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPFog)));
            descCommonSet->setBuffer(descCommonSet->RSPLightVector, drawBuffers->rspLightsBuffer.get(), drawBuffers->rspLightsBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPLight)));
            descCommonSet->setTexture(descCommonSet->gViewDirection, rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingPosition, rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingNormal, rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingSpecular, rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDiffuse, rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gInstanceId, rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReflection, rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gRefraction, rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gTransparent, rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFlow, rtResources->flowTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReactiveMask, rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gLockMask, rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDepth, rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDepth, rtResources->depthTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredDirectLight, rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredIndirectLight, rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL);

            const uint32_t hitBufferPixelCount = rtResources->textureWidth * rtResources->textureHeight * RaytracingResources::MaxHitQueries;
            descCommonSet->setBuffer(descCommonSet->gHitVelocityDistance, rtResources->hitVelocityDistanceBuffer.get(), hitBufferPixelCount * 16, rtResources->hitVelocityDistanceBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitColor, rtResources->hitColorBuffer.get(), hitBufferPixelCount * 4, rtResources->hitColorBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitNormalFog, rtResources->hitNormalFogBuffer.get(), hitBufferPixelCount * 8, rtResources->hitNormalFogBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitInstanceId, rtResources->hitInstanceIdBuffer.get(), hitBufferPixelCount * 2, rtResources->hitInstanceIdBufferView.get());
            descCommonSet->setAccelerationStructure(descCommonSet->SceneBVH, rtResources->topLevelAS.get());
            descCommonSet->setBuffer(descCommonSet->SceneLights, rtResources->lightsBuffer.get(), sizeof(interop::PointLight) * std::max(rtResources->rtParams.lightsCount, 1U), RenderBufferStructuredView(sizeof(interop::PointLight)));
            descCommonSet->setBuffer(descCommonSet->interleavedRasters, interleavedRastersBuffer.get(), sizeof(interop::InterleavedRaster) * std::max(interleavedRastersCount, 1U), RenderBufferStructuredView(sizeof(interop::InterleavedRaster)));
            descCommonSet->setTexture(descCommonSet->gBlueNoise, blueNoiseTexture, RenderTextureLayout::SHADER_READ);
            descCommonSet->setBuffer(descCommonSet->instanceExtraParams, drawBuffers->extraParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::ExtraParams)));
            descCommonSet->setBuffer(descCommonSet->RtParams, rtResources->rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        }
#   endif

        descCommonSet->setBuffer(descCommonSet->FrParams, frameParamsBuffer.get(), sizeof(interop::FrameParams));
        descCommonSet->setBuffer(descCommonSet->instanceRenderIndices, renderIndicesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderIndices)));
        descCommonSet->setBuffer(descCommonSet->instanceRDPParams, drawBuffers->rdpParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPParams)));
        descCommonSet->setBuffer(descCommonSet->RDPTiles, drawBuffers->rdpTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPTile)));
        descCommonSet->setBuffer(descCommonSet->GPUTiles, drawBuffers->gpuTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::GPUTile)));
        descCommonSet->setBuffer(descCommonSet->DynamicRenderParams, drawBuffers->renderParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderParams)));

        // Make sure the versions vector matches the texture cache size.
        descriptorTextureVersions.resize(textureCacheSize, 0);

        // Update texture vector with static textures from the cache and dynamic resource views.
        if (descriptorTextureGlobalVersion != textureCacheGlobalVersion) {
            const uint32_t textureVersionSize = static_cast<uint32_t>(textureCacheVersions.size());
            for (uint32_t i = 0; i < textureVersionSize; i++) {
                if (textureCacheVersions[i] == descriptorTextureVersions[i]) {
                    continue;
                }

                descriptorTextureVersions[i] = textureCacheVersions[i];
                if (textureCacheTextures[i] == nullptr) {
                    continue;
                }

                if (textureCacheReplacementMapEnabled && (textureCacheTextureReplacements[i] != nullptr)) {
                    descTextureSet->setTexture(i, textureCacheTextureReplacements[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else if (textureCacheTextures[i]->texture != nullptr) {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->tmem.get(), RenderTextureLayout::SHADER_READ);
                }
            }

            descriptorTextureGlobalVersion = textureCacheGlobalVersion;
        }

        for (const DynamicTextureView &dynamicView : dynamicTextureViewVector) {
            descTextureSet->setTexture(dynamicView.dstIndex, dynamicView.texture, RenderTextureLayout::SHADER_READ, dynamicView.textureView);
            descriptorTextureVersions[dynamicView.dstIndex] = 0;
        }
    }

    void FramebufferRenderer::updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (smoothDescSet == nullptr) {
            smoothDescSet = std::make_unique<RSPSmoothNormalDescriptorSet>(worker->device);
        }

        smoothDescSet->setBuffer(smoothDescSet->srcWorldPos, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcCol, drawBuffers->normalColorBuffer.get(), drawBuffers->normalColorBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint8_t) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        smoothDescSet->setBuffer(smoothDescSet->dstWorldNorm, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
    }

    void FramebufferRenderer::updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (vertexTestZSet == nullptr) {
            vertexTestZSet = std::make_unique<RSPVertexTestZDescriptorSet>(worker->device);
        }
        
        vertexTestZSet->setBuffer(vertexTestZSet->screenPos, outputBuffers->screenPosBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));
        vertexTestZSet->setBuffer(vertexTestZSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        vertexTestZSet->setBuffer(vertexTestZSet->dstFaceIndices, outputBuffers->testZIndexBuffer.buffer.get(), outputBuffers->testZIndexBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
    }

    void FramebufferRenderer::updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        updateShaderDescriptorSet(worker, drawBuffers, outputBuffers, raytracingEnabled);
        updateRSPVertexTestZSet(worker, drawBuffers, outputBuffers);

#   if RT_ENABLED
        if (raytracingEnabled) {
            updateRSPSmoothNormalSet(worker, drawBuffers, outputBuffers);
            rtResources->updateShaderSets(worker, shaderLibrary);
        }
#   endif
    }

    bool FramebufferRenderer::submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState) {
        if (depthState == readOnly) {
            return false;
        }
        
        RenderFramebuffer *renderFramebuffer = readOnly ? fbStorage->colorWriteDepthRead.get() : fbStorage->colorDepthWrite.get();
        const RenderTextureLayout depthReadState = RenderTextureLayout::DEPTH_READ;
        const RenderTextureLayout depthWriteState = RenderTextureLayout::DEPTH_WRITE;
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(fbStorage->depthTarget->texture.get(), readOnly ? depthReadState : RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(renderFramebuffer);
        depthState = readOnly;
        return true;
    }
    
    void FramebufferRenderer::submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState) {
        InstanceDrawCall::Type previousCallType = InstanceDrawCall::Type::Unknown;
        bool previousVertexTestZ = false;
        const RenderPipeline *previousPipeline = nullptr;
        RenderRect previousScissor;
        interop::RasterParams rasterParams = {};
        const bool jfgDebugWorld = rt64_jfg_debug_world.load(std::memory_order_relaxed) != 0; // JFG PORT: see RasterPS
        const bool jfgDebugNormal = rt64_jfg_debug_normal.load(std::memory_order_relaxed) != 0;
        const bool jfgDebugShadow = rt64_jfg_debug_shadow.load(std::memory_order_relaxed) != 0;
        const float jfgFacing = rt64_jfg_facing.load(std::memory_order_relaxed);
        const uint32_t jfgShadows = uint32_t(std::clamp(rt64_jfg_shadows.load(std::memory_order_relaxed), 0.0f, 1.0f) * 255.0f + 0.5f);
        const uint32_t jfgAmbient = uint32_t(std::clamp(rt64_jfg_level_ambient.load(std::memory_order_relaxed), 0.0f, 1.99f) * 128.0f + 0.5f);
        const bool jfgPixelLights = (rt64_jfg_pixel_lights.load(std::memory_order_relaxed) != 0) && (jfgLightCount > 0);
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descDummyFbSet = framebuffer.descDummyFbSet->get();

        auto switchToGraphicsPipeline = [&]() {
            previousCallType = InstanceDrawCall::Type::Unknown;
            previousVertexTestZ = false;
            previousPipeline = nullptr;
            previousScissor = RenderRect();
            worker->commandList->setGraphicsPipelineLayout(rendererPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(depthState ? descRealFbSet : descDummyFbSet, 3);
            worker->commandList->setViewports(framebuffer.viewport);
        };

        auto switchToDepthRead = [&]() {
            if (submitDepthAccess(worker, fbStorage, true, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            }
        };

        auto switchToDepthWrite = [&]() {
            if (submitDepthAccess(worker, fbStorage, false, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descDummyFbSet, 3);
            }
        };

        auto drawCallTriangles = [&](const InstanceDrawCall &drawCall) {
            if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                worker->commandList->drawIndexedInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0, 0);
            }
            else {
                worker->commandList->drawInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0);
            }
        };

        if (fbStorage->colorTarget != nullptr) {
            switchToGraphicsPipeline();
        }

        // JFG PORT: ambient occlusion and bloom, once per scene. The scene is scanned first for the HUD: the first draw
        // without any depth test after the 3D world (>= SsaoMinDepthWrites depth-writing draws), or the scene's end.
        // Bloom goes right before the HUD. Ambient occlusion goes right after the last depth-writing draw before it, so
        // what is drawn over the world without writing depth (gem and pickup sprites, particles, shots, water,
        // lens flares) is not darkened by the occlusion of the surfaces behind it. Such depth-tested, non-writing draws
        // that come earlier, between depth-writing ones, are moved to just after the occlusion pass (in their order);
        // the depth test still hides them behind anything in front, so only their blending over later opaque draws
        // behind them changes. Nothing is moved across a fill (clear) or a draw without depth test.
        const bool targetsPresent = (fbStorage->colorTarget != nullptr) && (fbStorage->depthTarget != nullptr);
        const bool ssaoOn = (rt64_jfg_ssao.load(std::memory_order_relaxed) != 0) && (ssaoPipeline != nullptr) && (ssaoBlurPipeline != nullptr) &&
            targetsPresent && (fbStorage->colorWriteDepthRead != nullptr);
        const bool bloomOn = (rt64_jfg_bloom.load(std::memory_order_relaxed) != 0) && (bloomCompositePipeline != nullptr) &&
            targetsPresent && (fbStorage->colorWriteDepthRead != nullptr);
        // JFG PORT: fog and vignette only on the game's own picture during play: not on the pause screen (its backdrop is drawn
        // once and kept, so it would get them again every frame) and not on smaller 3D scenes (decided after the scan below).
        const bool jfgPausedAny = (rt64_jfg_paused.load(std::memory_order_relaxed) != 0);
        const bool fogWanted = (rt64_jfg_fog.load(std::memory_order_relaxed) != 0) &&
            (fogPipeline != nullptr) && targetsPresent && (fbStorage->colorWriteDepthRead != nullptr) && !jfgPausedAny;
        const bool vignetteWanted = (rt64_jfg_vignette.load(std::memory_order_relaxed) != 0) && (vignettePipeline != nullptr) &&
            targetsPresent && (fbStorage->colorWriteDepthRead != nullptr) && !jfgPausedAny;
        bool fogScene = false;
        bool vignetteScene = false;
        const uint32_t SsaoMinDepthWrites = 12;
        const uint32_t sceneDrawCount = uint32_t(rasterScene.instanceIndices.size());
        uint32_t ssaoPosition = UINT32_MAX;
        uint32_t bloomPosition = UINT32_MAX;
        // JFG PORT: where fog, bloom and vignette go (right before the HUD), and the scene's picture is saved first (see
        // bloomSnapshots); UINT32_MAX when the scene gets none of them.
        uint32_t effectsPosition = UINT32_MAX;
        const uint32_t SsaoMarker = UINT32_MAX;
        const uint32_t EffectsMarker = UINT32_MAX - 1;
        thread_local std::vector<uint32_t> drawOrder;
        thread_local std::vector<uint32_t> deferredDraws;
        drawOrder.clear();
        deferredDraws.clear();
        // JFG PORT: in widescreen the colour buffer is wider than the game's 4:3 picture. The game only draws (and clears) its own
        // 4:3 area; what lies beyond is whatever was left from earlier frames, which showed up as old pictures at the sides of the
        // screen. The sides are cleared to black at the start of each 3D scene (below).
        const bool sidesWide = (fbStorage->colorTarget != nullptr) && (uint64_t(fbStorage->colorTarget->width) * 3ULL > uint64_t(fbStorage->colorTarget->height) * 4ULL + 6ULL);
        bool worldScene = false;
        const bool jfgPaused = sidesWide && (rt64_jfg_paused.load(std::memory_order_relaxed) != 0);
        int32_t sceneTop = INT32_MAX, sceneBottom = INT32_MIN; // vertical extent of the scene's depth-writing draws
        // The projection of the last perspective draw that wrote depth (the ambient occlusion pass rebuilds positions with it).
        float ssaoTanHalfFovY = 0.0f, ssaoNear = 0.0f, ssaoFar = 0.0f;
        if (ssaoOn || bloomOn || sidesWide || fogWanted || vignetteWanted) {
            uint32_t depthWrites = 0;
            uint32_t fogBlendWrites = 0; // depth-writing draws whose blender mixes in the fog colour (the game's own fog)
            uint32_t lastWritePosition = 0;
            uint32_t hudPosition = sceneDrawCount;
            for (uint32_t pos = 0; pos < sceneDrawCount; pos++) {
                const InstanceDrawCall &drawCall = instanceDrawCallVector[rasterScene.instanceIndices[pos]];
                const bool isTriangles = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) ||
                    (drawCall.type == InstanceDrawCall::Type::RawTriangles) || (drawCall.type == InstanceDrawCall::Type::RegularRect);
                if (!isTriangles || drawCall.triangles.scissor.isEmpty()) {
                    continue;
                }

                const interop::OtherMode otherMode = drawCall.triangles.shaderDesc.otherMode;
                if (otherMode.zUpd()) {
                    depthWrites++;
                    // Blender inputs (bits 16-31 of the other mode, GBL_c1/GBL_c2): first cycle's P at 14-15 and M at 6-7 of
                    // blenderInputs(); 3 is G_BL_CLR_FOG.
                    const uint32_t blender = otherMode.blenderInputs();
                    if ((((blender >> 14) & 3U) == 3U) || (((blender >> 6) & 3U) == 3U)) {
                        fogBlendWrites++;
                    }
                    sceneTop = std::min(sceneTop, drawCall.triangles.scissor.top);
                    sceneBottom = std::max(sceneBottom, drawCall.triangles.scissor.bottom);
                    lastWritePosition = pos;
                    if (drawCall.triangles.jfgNear > 0.0f) {
                        ssaoTanHalfFovY = drawCall.triangles.jfgTanHalfFovY;
                        ssaoNear = drawCall.triangles.jfgNear;
                        ssaoFar = drawCall.triangles.jfgFar;
                    }
                }
                else if (!otherMode.zCmp() && (depthWrites >= SsaoMinDepthWrites)) {
                    hudPosition = pos;
                    break;
                }
            }

            worldScene = depthWrites >= SsaoMinDepthWrites;
            // Only a scene that fills the screen's height is the game's picture; the pause screen's character panel is a smaller
            // 3D scene whose edges must not be blacked out.
            if (worldScene && sidesWide && !jfgPaused && (uint64_t(std::max(sceneBottom - sceneTop, 0)) * 10ULL < uint64_t(fbStorage->colorTarget->height) * 9ULL)) {
                worldScene = false;
            }
            if (depthWrites >= SsaoMinDepthWrites) {
                ssaoPosition = ssaoOn ? (lastWritePosition + 1) : UINT32_MAX;
                bloomPosition = bloomOn ? hudPosition : UINT32_MAX;
                const bool fullHeight = uint64_t(std::max(sceneBottom - sceneTop, 0)) * 10ULL >= uint64_t(fbStorage->colorTarget->height) * 9ULL;
                // The level's own fog shows when its scenery blends with the fog colour and the fog lies within the far plane.
                const bool gameFog = (fogBlendWrites > 0) && (rt64_jfg_fog_blocked.load(std::memory_order_relaxed) != 0);
                if (gameFog && fullHeight) {
                    rt64_jfg_fog_game_scenes.fetch_add(1, std::memory_order_relaxed);
                }

                fogScene = fogWanted && fullHeight && !gameFog;
                vignetteScene = vignetteWanted && fullHeight;
                if (bloomOn || fogScene || vignetteScene) {
                    effectsPosition = hudPosition;
                }
            }
        }

        if (ssaoPosition != UINT32_MAX) {
            // Draws that may move: from the last fill or no-depth-test draw before the occlusion pass onwards.
            uint32_t movableStart = 0;
            for (uint32_t pos = 0; pos < ssaoPosition; pos++) {
                const InstanceDrawCall &drawCall = instanceDrawCallVector[rasterScene.instanceIndices[pos]];
                const bool isTriangles = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) ||
                    (drawCall.type == InstanceDrawCall::Type::RawTriangles) || (drawCall.type == InstanceDrawCall::Type::RegularRect);
                if (drawCall.type == InstanceDrawCall::Type::FillRect) {
                    movableStart = pos + 1;
                }
                else if (isTriangles && !drawCall.triangles.shaderDesc.otherMode.zCmp() && !drawCall.triangles.shaderDesc.otherMode.zUpd()) {
                    movableStart = pos + 1;
                }
            }

            for (uint32_t pos = 0; pos < sceneDrawCount; pos++) {
                if (pos == ssaoPosition) {
                    drawOrder.emplace_back(SsaoMarker);
                    drawOrder.insert(drawOrder.end(), deferredDraws.begin(), deferredDraws.end());
                }

                if (pos == effectsPosition) {
                    drawOrder.emplace_back(EffectsMarker);
                }

                const uint32_t index = rasterScene.instanceIndices[pos];
                const InstanceDrawCall &drawCall = instanceDrawCallVector[index];
                const bool isTriangles = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) ||
                    (drawCall.type == InstanceDrawCall::Type::RawTriangles) || (drawCall.type == InstanceDrawCall::Type::RegularRect);
                if ((pos >= movableStart) && (pos < ssaoPosition) && isTriangles &&
                    drawCall.triangles.shaderDesc.otherMode.zCmp() && !drawCall.triangles.shaderDesc.otherMode.zUpd())
                {
                    deferredDraws.emplace_back(index);
                }
                else {
                    drawOrder.emplace_back(index);
                }
            }

            if (ssaoPosition == sceneDrawCount) {
                drawOrder.emplace_back(SsaoMarker);
                drawOrder.insert(drawOrder.end(), deferredDraws.begin(), deferredDraws.end());
            }

            if (effectsPosition == sceneDrawCount) {
                drawOrder.emplace_back(EffectsMarker);
            }
        }
        else {
            drawOrder.assign(rasterScene.instanceIndices.begin(), rasterScene.instanceIndices.end());
            if (effectsPosition != UINT32_MAX) {
                drawOrder.insert(drawOrder.begin() + effectsPosition, EffectsMarker);
            }
        }
        // JFG PORT: keep the scene as it is now, before fog, glow and vignette go on (see bloomSnapshots): the next scene for this
        // colour buffer starts from it, so the parts the game does not draw again do not get the effects added again and again.
        auto takeSnapshot = [&]() {
            RenderTarget *colorTarget = fbStorage->colorTarget;
            const uint32_t width = colorTarget->width;
            const uint32_t height = colorTarget->height;
            {
                const RenderFormat snapshotFormat = colorTarget->format;
                const uint32_t snapshotSamples = colorTarget->multisampling.sampleCount;
                BloomSnapshot *snapshot = nullptr;
                for (const std::unique_ptr<BloomSnapshot> &entry : bloomSnapshots) {
                    if (entry->target == colorTarget) {
                        snapshot = entry.get();
                        break;
                    }
                }

                if (snapshot == nullptr) {
                    if (bloomSnapshots.size() >= 6) {
                        for (std::unique_ptr<BloomSnapshot> &entry : bloomSnapshots) {
                            bloomSnapshotRetired.emplace_back(std::move(entry->texture));
                        }

                        bloomSnapshots.clear();
                    }

                    bloomSnapshots.emplace_back(std::make_unique<BloomSnapshot>());
                    snapshot = bloomSnapshots.back().get();
                    snapshot->target = colorTarget;
                }

                if (snapshot->targetTexture != colorTarget->texture.get()) {
                    snapshot->targetTexture = colorTarget->texture.get();
                    snapshot->valid = false;
                }

                if ((snapshot->texture == nullptr) || (snapshot->width != width) || (snapshot->height != height) ||
                    (snapshot->format != snapshotFormat) || (snapshot->samples != snapshotSamples))
                {
                    if (snapshot->texture != nullptr) {
                        bloomSnapshotRetired.emplace_back(std::move(snapshot->texture));
                    }

                    snapshot->valid = false;
                    RenderClearValue snapshotClear = RenderClearValue::Color(RenderColor(), snapshotFormat);
                    snapshot->texture = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, snapshotFormat, colorTarget->multisampling, &snapshotClear));
                    if (snapshot->texture != nullptr) {
                        snapshot->texture->setName("JFG Bloom Snapshot");
                        snapshot->width = width;
                        snapshot->height = height;
                        snapshot->format = snapshotFormat;
                        snapshot->samples = snapshotSamples;
                    }
                }

                if (snapshot->texture != nullptr) {
                    worker->commandList->barriers(RenderBarrierStage::COPY, {
                        RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COPY_SOURCE),
                        RenderTextureBarrier(snapshot->texture.get(), RenderTextureLayout::COPY_DEST)
                    });
                    worker->commandList->copyTexture(snapshot->texture.get(), colorTarget->texture.get());
                    snapshot->valid = true;
                    snapshot->frame = jfgFrame;
                }
            }

            // Back to drawing into the scene target (depth stays read-only).
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
            worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
            switchToGraphicsPipeline();
        };
        auto submitBloom = [&]() {
            RenderTarget *colorTarget = fbStorage->colorTarget;
            const uint32_t width = colorTarget->width;
            const uint32_t height = colorTarget->height;
            const uint32_t scale = std::max(1U, uint32_t(std::lround(height / 270.0)));
            BloomBuffers *buffers = getBloomBuffers(worker, (width + scale - 1) / scale, (height + scale - 1) / scale);
            if (buffers == nullptr) {
                return;
            }

            interop::BloomCB cb = {};
            cb.threshold = rt64_jfg_bloom_threshold.load(std::memory_order_relaxed);
            cb.knee = 0.1f;
            cb.intensity = rt64_jfg_bloom_intensity.load(std::memory_order_relaxed);
            cb.sigma = rt64_jfg_bloom_radius.load(std::memory_order_relaxed);
            cb.scale = scale;
            RenderDescriptorSet *sceneSet = makeBloomDescriptorSet(worker, colorTarget->texture.get(), colorTarget->textureView.get());
            RenderDescriptorSet *bufferSet[2] = {
                makeBloomDescriptorSet(worker, buffers->texture[0].get(), buffers->textureView[0].get()),
                makeBloomDescriptorSet(worker, buffers->texture[1].get(), buffers->textureView[1].get())
            };

            const RenderViewport smallViewport(0.0f, 0.0f, float(buffers->width), float(buffers->height));
            const RenderRect smallRect(0, 0, int32_t(buffers->width), int32_t(buffers->height));
            auto bloomPass = [&](RenderPipeline *pipeline, RenderTexture *input, RenderTextureLayout inputLayout, uint32_t output, RenderDescriptorSet *set) {
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                    RenderTextureBarrier(input, inputLayout),
                    RenderTextureBarrier(buffers->texture[output].get(), RenderTextureLayout::COLOR_WRITE)
                });
                worker->commandList->setFramebuffer(buffers->framebuffer[output].get());
                worker->commandList->setPipeline(pipeline);
                worker->commandList->setGraphicsPipelineLayout(bloomPipelineLayout);
                worker->commandList->setGraphicsDescriptorSet(set, 0);
                worker->commandList->setViewports(smallViewport);
                worker->commandList->setScissors(smallRect);
                worker->commandList->setGraphicsPushConstants(0, &cb);
                worker->commandList->drawInstanced(3, 1, 0, 0);
            };

            // Bright parts of the scene, downsampled into buffer 0; blurred across into 1 and down back into 0.
            bloomPass(bloomExtractPipeline, colorTarget->texture.get(), RenderTextureLayout::SHADER_READ, 0, sceneSet);
            cb.directionX = 1; cb.directionY = 0;
            bloomPass(bloomBlurPipeline, buffers->texture[0].get(), RenderTextureLayout::SHADER_READ, 1, bufferSet[0]);
            cb.directionX = 0; cb.directionY = 1;
            bloomPass(bloomBlurPipeline, buffers->texture[1].get(), RenderTextureLayout::SHADER_READ, 0, bufferSet[1]);

            // Back to the scene target (depth stays read-only), add the bloom.
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                RenderTextureBarrier(buffers->texture[0].get(), RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE)
            });
            worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
            worker->commandList->setPipeline(bloomCompositePipeline);
            worker->commandList->setGraphicsPipelineLayout(bloomPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(bufferSet[0], 0);
            worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(width), float(height)));
            worker->commandList->setScissors(RenderRect(0, 0, int32_t(width), int32_t(height)));
            worker->commandList->setGraphicsPushConstants(0, &cb);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            // Restore everything the scene's draws expect.
            switchToGraphicsPipeline();
        };
        // JFG PORT: ambient occlusion in two passes (see SsaoPS.hlsl): from the depth into the occlusion buffer, then blurred
        // (keeping edges) and multiplied into the scene.
        auto submitSsao = [&]() {
            RenderTarget *colorTarget = fbStorage->colorTarget;
            const uint32_t width = colorTarget->width;
            const uint32_t height = colorTarget->height;
            BloomBuffers *buffers = getBloomBuffers(worker, width, height, true);
            if (buffers == nullptr) {
                return;
            }

            switchToDepthRead();
            const RenderViewport fullViewport(0.0f, 0.0f, float(width), float(height));
            const RenderRect fullRect(0, 0, int32_t(width), int32_t(height));
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(buffers->texture[0].get(), RenderTextureLayout::COLOR_WRITE));
            worker->commandList->setFramebuffer(buffers->framebuffer[0].get());
            worker->commandList->setPipeline(ssaoPipeline);
            worker->commandList->setViewports(fullViewport);
            worker->commandList->setScissors(fullRect);
            interop::RasterParams ssaoParams = {};
            const float strength = rt64_jfg_ssao_strength.load(std::memory_order_relaxed);
            const float radius = rt64_jfg_ssao_radius.load(std::memory_order_relaxed);
            memcpy(&ssaoParams.padding.x, &strength, sizeof(float));
            memcpy(&ssaoParams.padding.y, &radius, sizeof(float));
            // The scene's own projection; without one, the field of view the game last set (camSetFOV) and its usual planes.
            const float fovTan = (ssaoNear > 0.0f) ? ssaoTanHalfFovY : rt64_jfg_fov_tan.load(std::memory_order_relaxed);
            memcpy(&ssaoParams.padding.z, &fovTan, sizeof(float));
            ssaoParams.screenScale.x = ssaoNear;
            ssaoParams.screenScale.y = ssaoFar;
            worker->commandList->setGraphicsPushConstants(0, &ssaoParams);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            // Back to the scene target (depth stays read-only), blur the occlusion onto it.
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                RenderTextureBarrier(buffers->texture[0].get(), RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE)
            });
            worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
            worker->commandList->setPipeline(ssaoBlurPipeline);
            worker->commandList->setGraphicsPipelineLayout(bloomPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(makeBloomDescriptorSet(worker, buffers->texture[0].get(), buffers->textureView[0].get()), 0);
            worker->commandList->setViewports(fullViewport);
            worker->commandList->setScissors(fullRect);
            interop::BloomCB blurParams = {};
            worker->commandList->setGraphicsPushConstants(0, &blurParams);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            // Restore everything the scene's draws expect.
            switchToGraphicsPipeline();
        };
        // JFG PORT: full-screen passes in the scene's own pipeline layout (see FogPS.hlsl / VignettePS.hlsl).
        auto submitFullScreen = [&](RenderPipeline *pipeline, const interop::RasterParams &params) {
            worker->commandList->setPipeline(pipeline);
            worker->commandList->setScissors(RenderRect(0, 0, int32_t(fbStorage->colorTarget->width), int32_t(fbStorage->colorTarget->height)));
            worker->commandList->setGraphicsPushConstants(0, &params);
            worker->commandList->drawInstanced(3, 1, 0, 0);
            previousPipeline = nullptr;
            previousScissor = RenderRect();
            previousCallType = InstanceDrawCall::Type::Unknown;
        };
        auto submitFog = [&]() {
            interop::RasterParams fogParams = {};
            const float start = rt64_jfg_fog_start.load(std::memory_order_relaxed);
            const float end = rt64_jfg_fog_end.load(std::memory_order_relaxed);
            const float density = rt64_jfg_fog_density.load(std::memory_order_relaxed);
            const uint32_t color = rt64_jfg_fog_color.load(std::memory_order_relaxed);
            memcpy(&fogParams.padding.x, &start, sizeof(float));
            memcpy(&fogParams.padding.y, &end, sizeof(float));
            memcpy(&fogParams.padding.z, &density, sizeof(float));
            fogParams.screenScale.x = float((color >> 16) & 0xFF) / 255.0f;
            fogParams.screenScale.y = float((color >> 8) & 0xFF) / 255.0f;
            fogParams.screenOffset.x = float(color & 0xFF) / 255.0f;
            submitFullScreen(fogPipeline, fogParams);
        };
        auto submitVignette = [&]() {
            interop::RasterParams vignetteParams = {};
            const float strength = rt64_jfg_vignette_strength.load(std::memory_order_relaxed);
            const float size = rt64_jfg_vignette_size.load(std::memory_order_relaxed);
            const float aspect = float(fbStorage->colorTarget->width) / float(std::max(fbStorage->colorTarget->height, 1U));
            memcpy(&vignetteParams.padding.x, &strength, sizeof(float));
            memcpy(&vignetteParams.padding.y, &size, sizeof(float));
            memcpy(&vignetteParams.padding.z, &aspect, sizeof(float));
            submitFullScreen(vignettePipeline, vignetteParams);
        };
        // Right before the HUD: save the clean picture, then fog (so bloom glows over the fogged scene), bloom, vignette.
        auto submitEffects = [&]() {
            takeSnapshot();
            if (fogScene) {
                submitFog();
                rt64_jfg_fog_drawn.fetch_add(1, std::memory_order_relaxed);
            }

            if (bloomPosition != UINT32_MAX) {
                submitBloom();
            }

            if (vignetteScene) {
                submitVignette();
            }
        };
        
        // JFG PORT: is this the first scene to draw into this colour buffer in this frame? Restoring the saved picture over the
        // buffer is only right then; later it would wipe what the earlier scenes of the frame (a backdrop, say) have drawn.
        bool firstSceneOnTarget = true;
        if (fbStorage->colorTarget != nullptr) {
            bool found = false;
            for (std::pair<const RenderTarget *, uint32_t> &touched : jfgTargetsTouched) {
                if (touched.first == fbStorage->colorTarget) {
                    firstSceneOnTarget = (touched.second != jfgFrame);
                    touched.second = jfgFrame;
                    found = true;
                    break;
                }
            }
            if (!found) {
                jfgTargetsTouched.emplace_back(fbStorage->colorTarget, jfgFrame);
            }
        }

        // JFG PORT: a scene without bloom (or fog / vignette) that draws into a colour buffer changes what the saved picture of that buffer was
        // made from (a fade to black, another screen), so the picture is dropped: restoring it later would bring back an old image.
        if ((effectsPosition == UINT32_MAX) && (sceneDrawCount > 0) && (fbStorage->colorTarget != nullptr)) {
            for (const std::unique_ptr<BloomSnapshot> &entry : bloomSnapshots) {
                if ((entry->target == fbStorage->colorTarget) && (entry->frame != jfgFrame)) {
                    entry->valid = false;
                }
            }
        }

        // JFG PORT: a scene that gets bloom (or fog / vignette) starts from the clean picture saved when the last one for this colour buffer was
        // made, so the parts the game does not draw again do not get the glow added to the old glow (see bloomSnapshots).
        if ((effectsPosition != UINT32_MAX) && firstSceneOnTarget) {
            RenderTarget *colorTarget = fbStorage->colorTarget;
            for (const std::unique_ptr<BloomSnapshot> &entry : bloomSnapshots) {
                if ((entry->target != colorTarget) || !entry->valid || (entry->texture == nullptr) || (entry->frame == jfgFrame) ||
                    (entry->targetTexture != colorTarget->texture.get()))
                {
                    continue;
                }

                if ((entry->width == colorTarget->width) && (entry->height == colorTarget->height) &&
                    (entry->format == colorTarget->format) && (entry->samples == colorTarget->multisampling.sampleCount))
                {
                    worker->commandList->barriers(RenderBarrierStage::COPY, {
                        RenderTextureBarrier(entry->texture.get(), RenderTextureLayout::COPY_SOURCE),
                        RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COPY_DEST)
                    });
                    worker->commandList->copyTexture(colorTarget->texture.get(), entry->texture.get());
                    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
                    worker->commandList->setFramebuffer(depthState ? fbStorage->colorWriteDepthRead.get() : fbStorage->colorDepthWrite.get());
                    switchToGraphicsPipeline();
                }

                entry->valid = false;
                break;
            }
        }

        if (sidesWide && (worldScene || jfgPaused) && ((jfgSidesClearedTarget != fbStorage->colorTarget) || (jfgSidesClearedFrame != jfgFrame))) {
            const RenderTarget *target = fbStorage->colorTarget;
            const int32_t fullWidth = int32_t(target->width);
            const int32_t sideWidth = (fullWidth - int32_t((uint64_t(target->height) * 4ULL) / 3ULL)) / 2;
            if (sideWidth > 0) {
                const RenderRect sides[2] = {
                    RenderRect(0, 0, sideWidth, int32_t(target->height)),
                    RenderRect(fullWidth - sideWidth, 0, fullWidth, int32_t(target->height))
                };
                worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 1.0f), sides, 2);
            }

            jfgSidesClearedTarget = target;
            jfgSidesClearedFrame = jfgFrame;
        }

        for (uint32_t i : drawOrder) {
            if (i == SsaoMarker) {
                submitSsao();
                continue;
            }
            else if (i == EffectsMarker) {
                switchToDepthRead();
                submitEffects();
                continue;
            }

            const InstanceDrawCall &drawCall = instanceDrawCallVector[i];
            switch (drawCall.type) {
            case InstanceDrawCall::Type::IndexedTriangles: 
            case InstanceDrawCall::Type::RawTriangles:
            case InstanceDrawCall::Type::RegularRect: {
                assert(fbStorage->colorTarget != nullptr);

                const bool typeDifferent = (drawCall.type != previousCallType);
                const bool testZDifferent = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) && (drawCall.triangles.vertexTestZ != previousVertexTestZ);
                if (typeDifferent || testZDifferent) {
                    switch (drawCall.type) {
                    case InstanceDrawCall::Type::IndexedTriangles:
                        worker->commandList->setVertexBuffers(0, indexedVertexViews.data(), uint32_t(indexedVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(drawCall.triangles.vertexTestZ ? &testZIndexBufferView : &indexBufferView);
                        previousVertexTestZ = drawCall.triangles.vertexTestZ;
                        break;
                    case InstanceDrawCall::Type::RawTriangles:
                    case InstanceDrawCall::Type::RegularRect:
                        worker->commandList->setVertexBuffers(0, rawVertexViews.data(), uint32_t(rawVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(nullptr);
                        break;
                    default:
                        assert(false && "Unknown draw call type.");
                        break;
                    };

                    previousCallType = drawCall.type;
                }

                const auto &triangles = drawCall.triangles;
                assert(triangles.pipeline != nullptr);

                // Draw calls can sometimes end up with empty scissors and cause validation errors. We just skip them.
                if (triangles.scissor.isEmpty()) {
                    continue;
                }

                // A new pass must be started if decals are required and something wrote to the depth buffer before this call.
                const interop::OtherMode otherMode = triangles.shaderDesc.otherMode;
                bool depthDecal = (otherMode.zMode() == ZMODE_DEC);
                bool depthWrite = otherMode.zUpd();
                if (depthDecal) {
                    switchToDepthRead();
                }
                else if (!depthDecal && depthWrite) {
                    switchToDepthWrite();
                }

                // JFG PORT: the pause screens are 4:3 pictures. Their 3D objects (the planets of the map) reach beyond the picture,
                // which in widescreen would show in the sides, so nothing is drawn there while paused.
                RenderRect drawScissor = triangles.scissor;
                if (jfgPaused) {
                    const int32_t pictureWidth = int32_t((uint64_t(fbStorage->colorTarget->height) * 4ULL) / 3ULL);
                    const int32_t sideWidth = (int32_t(fbStorage->colorTarget->width) - pictureWidth) / 2;
                    drawScissor.left = std::max(drawScissor.left, sideWidth);
                    drawScissor.right = std::min(drawScissor.right, sideWidth + pictureWidth);
                    if (drawScissor.isEmpty()) {
                        continue;
                    }
                }

                if (previousScissor != drawScissor) {
                    worker->commandList->setScissors(drawScissor);
                    previousScissor = drawScissor;
                }

                if (previousPipeline != triangles.pipeline) {
                    worker->commandList->setPipeline(triangles.pipeline);
                    previousPipeline = triangles.pipeline;
                }
                
                rasterParams.renderIndex = i;
                rasterParams.screenScale = triangles.screenScale;
                rasterParams.screenOffset = triangles.screenOffset;
                rasterParams.padding.x = ((drawCall.type == InstanceDrawCall::Type::IndexedTriangles) ? RASTER_JFG_INDEXED : 0) |
                    (jfgDebugWorld ? RASTER_JFG_DEBUG_WORLD : 0) | (triangles.jfgLevel ? RASTER_JFG_LEVEL : 0) |
                    (jfgDebugNormal ? RASTER_JFG_DEBUG_NORMAL : 0) |
                    (jfgDebugShadow ? RASTER_JFG_DEBUG_SHADOW : 0) | (jfgShadows << 16) | (jfgAmbient << 24); // JFG PORT: see RasterVS, RasterPS
                rasterParams.padding.y = jfgPixelLights ? jfgLightCount : 0;
                memcpy(&rasterParams.padding.z, &jfgFacing, sizeof(float));
                worker->commandList->setGraphicsPushConstants(0, &rasterParams);

                drawCallTriangles(drawCall);

                // Simulate dither noise.
                if (triangles.postBlendDitherNoise) {
                    if (triangles.postBlendDitherNoiseNegative) {
                        worker->commandList->setPipeline(postBlendDitherNoiseSubNegativePipeline);
                    }
                    else {
                        worker->commandList->setPipeline(postBlendDitherNoiseAddPipeline);
                        drawCallTriangles(drawCall);

                        worker->commandList->setPipeline(postBlendDitherNoiseSubPipeline);
                    }

                    drawCallTriangles(drawCall);
                    previousPipeline = nullptr;
                }

                break;
            };
            case InstanceDrawCall::Type::FillRect: {
                const auto &clearRect = drawCall.clearRect;
                RenderTarget *chosenTarget = (fbStorage->colorTarget != nullptr) ? fbStorage->colorTarget : fbStorage->depthTarget;
                bool rectCoversWholeTarget = (clearRect.rect.left == 0) && (clearRect.rect.top == 0) && (uint32_t(clearRect.rect.right) == chosenTarget->width) && (uint32_t(clearRect.rect.bottom) == chosenTarget->height);
                const RenderRect *clearRects = rectCoversWholeTarget ? nullptr : &clearRect.rect;
                uint32_t clearRectCount = rectCoversWholeTarget ? 0 : 1;
                if (fbStorage->colorTarget != nullptr) {
                    worker->commandList->clearColor(0, clearRect.color, clearRects, clearRectCount);
                }
                else {
                    worker->commandList->clearDepth(true, clearRect.depth, clearRects, clearRectCount);
                }

                break;
            };
            case InstanceDrawCall::Type::VertexTestZ: {
                assert(testZIndexBuffer != nullptr);
                assert(fbStorage->colorTarget != nullptr);

                const interop::RSPVertexTestZCB &testZCB = drawCall.vertexTestZ;
                switchToDepthRead();

                const bool useMSAA = (fbStorage->colorTarget->multisampling.sampleCount > 0);
                const auto &rspVertexTestZ = useMSAA ? shaderLibrary->rspVertexTestZMS : shaderLibrary->rspVertexTestZ;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::WRITE));
                worker->commandList->setPipeline(rspVertexTestZ.pipeline.get());
                worker->commandList->setComputePipelineLayout(rspVertexTestZ.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &testZCB);
                worker->commandList->setComputeDescriptorSet(vertexTestZSet->get(), 0);
                worker->commandList->setComputeDescriptorSet(descRealFbSet, 1);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::READ));

                switchToGraphicsPipeline();
                break;
            };
            default:
                // Do nothing.
                break;
            }
        }


        // Mark targets for resolve.
        if (fbStorage->colorTarget != nullptr) {
            fbStorage->colorTarget->markForResolve();
        }

        if (fbStorage->depthTarget != nullptr) {
            fbStorage->depthTarget->markForResolve();
        }
    }

    // JFG PORT: bloom buffers for one size (see submitRasterScene); with ssao, the ambient occlusion buffer (one texture).
    FramebufferRenderer::BloomBuffers *FramebufferRenderer::getBloomBuffers(RenderWorker *worker, uint32_t width, uint32_t height, bool ssao) {
        std::vector<std::unique_ptr<BloomBuffers>> &bloomBuffers = ssao ? this->ssaoBuffers : this->bloomBuffers;
        for (const std::unique_ptr<BloomBuffers> &buffers : bloomBuffers) {
            if ((buffers->width == width) && (buffers->height == height)) {
                return buffers.get();
            }
        }

        // Window resized many times: retire the old sizes; they are freed once this workload has finished.
        if (bloomBuffers.size() >= 4) {
            for (std::unique_ptr<BloomBuffers> &buffers : bloomBuffers) {
                bloomBuffersRetired.emplace_back(std::move(buffers));
            }

            bloomBuffers.clear();
        }

        std::unique_ptr<BloomBuffers> buffers = std::make_unique<BloomBuffers>();
        buffers->width = width;
        buffers->height = height;
        const RenderFormat format = RasterShaderUber::BloomBufferFormat;
        RenderClearValue clearValue = RenderClearValue::Color(RenderColor(), format);
        for (uint32_t i = 0; i < (ssao ? 1U : 2U); i++) {
            buffers->texture[i] = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, format, RenderMultisampling(), &clearValue));
            if (buffers->texture[i] == nullptr) {
                return nullptr;
            }

            buffers->texture[i]->setName(ssao ? std::string("JFG Ambient Occlusion") : ("JFG Bloom #" + std::to_string(i)));
            buffers->textureView[i] = buffers->texture[i]->createTextureView(RenderTextureViewDesc::Texture2D(format));
            const RenderTexture *colorTexture = buffers->texture[i].get();
            buffers->framebuffer[i] = worker->device->createFramebuffer(RenderFramebufferDesc(&colorTexture, 1));
        }

        bloomBuffers.emplace_back(std::move(buffers));
        return bloomBuffers.back().get();
    }

    RenderDescriptorSet *FramebufferRenderer::makeBloomDescriptorSet(RenderWorker *worker, RenderTexture *texture, RenderTextureView *textureView) {
        std::unique_ptr<RenderTargetCopyDescriptorSet> set = std::make_unique<RenderTargetCopyDescriptorSet>(worker->device);
        set->setTexture(set->gInput, texture, RenderTextureLayout::SHADER_READ, textureView);
        bloomDescriptorSets.emplace_back(std::move(set));
        return bloomDescriptorSets.back()->get();
    }

    void FramebufferRenderer::updateMultisampling() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
#   if RT_ENABLED
        if (rtResources != nullptr) {
            rtResources->updateMultisampling();
        }
#   endif
    }

#if RT_ENABLED
    void FramebufferRenderer::updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene) {
        auto &rtParams = rtResources->rtParams;
        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;

        // Only use jitter when an upscaler is active.
        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        bool jitterActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (jitterActive) {
            const int phaseCount = upscaler->getJitterPhaseCount(rtResources->textureWidth, rtScene.screenWidth);
            rtParams.pixelJitter = HaltonJitter(frameParams.frameCount, phaseCount);
        }
        else {
            rtParams.pixelJitter = { 0.0f, 0.0f };
        }

        rtParams.viewport.x = rtScene.viewport.x;
        rtParams.viewport.y = rtScene.viewport.y;
        rtParams.viewport.z = rtScene.viewport.width;
        rtParams.viewport.w = rtScene.viewport.height;

        const auto &preset = rtScene.presetScene;
        rtParams.ambientBaseColor = hlslpp::float4(preset.ambientBaseColor, 0.0f);
        rtParams.ambientNoGIColor = hlslpp::float4(preset.ambientNoGIColor, 0.0f);
        rtParams.eyeLightDiffuseColor = hlslpp::float4(preset.eyeLightDiffuseColor, 0.0f);
        rtParams.eyeLightSpecularColor = hlslpp::float4(preset.eyeLightSpecularColor, 0.0f);
        rtParams.giDiffuseStrength = preset.giDiffuseStrength;
        rtParams.giBackgroundStrength = preset.giBackgroundStrength;
        rtParams.tonemapExposure = preset.tonemapExposure;
        rtParams.tonemapWhite = preset.tonemapWhite;
        rtParams.tonemapBlack = preset.tonemapBlack;

        const auto &proj = rtScene.curProjMatrix;
        rtParams.fovRadians = fovFromProj(proj);
        rtParams.nearDist = nearPlaneFromProj(proj);
        rtParams.farDist = farPlaneFromProj(proj);

        if (isnan(rtParams.fovRadians)) {
            rtParams.fovRadians = 0.75f;
        }

        if (isnan(rtParams.nearDist)) {
            rtParams.nearDist = 1.0f;
        }

        if (isnan(rtParams.farDist)) {
            rtParams.farDist = 1000.0f;
        }

        rtParams.view = rtScene.curViewMatrix;
        rtParams.projection = rtScene.curProjMatrix;

        rtParams.viewI = hlslpp::inverse(rtParams.view);
        rtParams.projectionI = hlslpp::inverse(rtParams.projection);
        rtParams.viewProj = hlslpp::mul(rtParams.view, rtParams.projection);
        rtParams.prevViewProj = hlslpp::mul(rtScene.prevViewMatrix, rtScene.prevProjMatrix);

        // TODO: There's probably a way to compute this without calculating the FOV/Near/Far values.
        // Pinhole camera vectors to generate non-normalized ray direction.
        // TODO: Make a fake target and focal distance at the midpoint of the near/far planes
        // until the game sends that data in some way in the future.
        const float FocalDistance = (rtParams.nearDist + rtParams.farDist) / 2.0f;
        const hlslpp::float3 Up(0.0f, 1.0f, 0.0f);
        const hlslpp::float3 Pos = viewPositionFrom(rtParams.viewI);
        const hlslpp::float3 Target = Pos + viewDirectionFrom(rtParams.viewI) * FocalDistance;
        hlslpp::float3 cameraW = hlslpp::normalize(Target - Pos) * FocalDistance;
        hlslpp::float3 cameraU = hlslpp::normalize(hlslpp::cross(cameraW, Up));
        hlslpp::float3 cameraV = hlslpp::normalize(hlslpp::cross(cameraU, cameraW));
        const float ulen = FocalDistance * std::tan(rtParams.fovRadians * 0.5f);// * rtParams.aspectRatio;
        const float vlen = FocalDistance * std::tan(rtParams.fovRadians * 0.5f);
        cameraU = cameraU * ulen;
        cameraV = cameraV * vlen;
        rtParams.cameraU = hlslpp::float4(cameraU, 0.0f);
        rtParams.cameraV = hlslpp::float4(cameraV, 0.0f);
        rtParams.cameraW = hlslpp::float4(cameraW, 0.0f);

        // Enable light reprojection if denoising is enabled.
#   ifdef DI_REPROJECTION_SUPPORT
        globalParamsBufferData.diReproject = !rtResources->skipReprojection && denoiserEnabled && (globalParamsBufferData.diSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
#   else
        rtParams.diReproject = 0;
#   endif

        rtParams.giReproject = !rtResources->skipReprojection && rtResources->denoiserEnabled && (rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
        rtParams.binaryLockMask = (rtResources->upscalerMode != UpscaleMode::FSR);
        rtParams.interleavedRastersCount = interleavedRastersCount;
        
        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        RenderDescriptorSet *descRealDepthSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descriptorSets[] = { descCommonSet->get(), descTextureSet->get(), descTextureSet->get(), descRealDepthSet };
        rtResources->updateTopLevelASResources(worker, instanceDrawCallVector, rtScene.instanceIndices);
        rtResources->createShaderBindingTable(worker, rtState, descriptorSets, uint32_t(std::size(descriptorSets)), hitGroupVector);
        rtResources->updateLightsBuffer(worker, rtScene);
    }
    
    void FramebufferRenderer::submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene) {
        // Unbind any render targets.
        worker->commandList->setFramebuffer(nullptr);

        // Resolve the color target if necessary before using it as the RT scene background.
        colorTarget->resolveTarget(worker, shaderLibrary);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(colorTarget->getResolvedTexture(), RenderTextureLayout::SHADER_READ));

        if (rtResources->transitionOutputBuffers) {
            RenderTextureBarrier afterCreationBarriers[] = {
                RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCreationBarriers, uint32_t(std::size(afterCreationBarriers)));
            rtResources->transitionOutputBuffers = false;
        }

        // Make sure all these buffers are usable as UAVs.
        RenderTextureBarrier preDispatchBarriers[] = {
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[1].get(), RenderTextureLayout::GENERAL)
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, preDispatchBarriers, uint32_t(std::size(preDispatchBarriers)));
        
        // Bind pipeline and dispatch primary rays.
        RT64_LOG_PRINTF("Dispatching primary rays");
        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 0;
        worker->commandList->setPipeline(rtState->pipeline.get());
        worker->commandList->setRaytracingPipelineLayout(rtPipelineLayout);
        worker->commandList->setRaytracingDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 1);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 2);
        worker->commandList->setRaytracingDescriptorSet(descRealFbSet, 3);
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Barriers for shading buffers before dispatching secondary rays.
        RenderTextureBarrier shadingBarriers[] = {
            RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL),
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, shadingBarriers, uint32_t(std::size(shadingBarriers)));

        // Dispatch rays for direct light.
        RT64_LOG_PRINTF("Dispatching direct light rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 1;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Dispatch rays for indirect light.
        RT64_LOG_PRINTF("Dispatching indirect light rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 2;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Wait until indirect light is done before dispatching reflection or refraction rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL));

        // Dispatch rays for refraction.
        RT64_LOG_PRINTF("Dispatching refraction rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 4;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Wait until refraction is done before dispatching reflection rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL));

        // Reflection passes.
        int reflections = rtResources->maxReflections;
        while (reflections > 0) {
            // Dispatch rays for reflection.
            RT64_LOG_PRINTF("Dispatching reflection rays");
            rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 3;
            worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);
            reflections--;

            // Add a barrier to wait for the input UAVs to be finished if there's more passes left to be done.
            if (reflections > 0) {
                RenderTextureBarrier newInputBarriers[] = {
                    RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, newInputBarriers, uint32_t(std::size(newInputBarriers)));
            }
        }

        // Copy direct light raw buffer to the first direct filtered buffer.
        {
            RenderTexture *source = rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredDirectLightTexture[1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Copy indirect light raw buffer to the first indirect filtered buffer.
        bool denoiseGI = rtResources->denoiserEnabled && (rtResources->rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS);
        {
            RenderTexture *source = rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredIndirectLightTexture[denoiseGI ? 0 : 1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Apply a gaussian filter to the indirect light with a compute shader.
        if (denoiseGI) {
            for (int i = 0; i < 5; i++) {
                const uint32_t ThreadGroupWorkCount = 8;
                uint32_t dispatchX = (rtResources->textureWidth + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                uint32_t dispatchY = (rtResources->textureHeight + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                const ShaderRecord &gaussianFilter = shaderLibrary->gaussianFilterRGB3x3;
                interop::TextureCB textureCB;
                textureCB.TextureSize = { rtResources->textureWidth, rtResources->textureHeight };
                textureCB.TexelSize = { 1.0f / rtResources->textureWidth, 1.0f / rtResources->textureHeight };

                worker->commandList->setPipeline(gaussianFilter.pipeline.get());
                worker->commandList->setComputePipelineLayout(gaussianFilter.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &textureCB);
                worker->commandList->setComputeDescriptorSet(rtResources->indirectFilterSets[i % 2]->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);

                RenderTextureBarrier afterBlurBarriers[] = {
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 1 : 0].get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 0 : 1].get(), RenderTextureLayout::SHADER_READ)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterBlurBarriers, uint32_t(std::size(afterBlurBarriers)));
            }
        }

        // Compose the output buffer.
        RenderTexture *rtOutputCur = rtResources->outputTexture[rtResources->swapBuffers ? 1 : 0].get();

        // Barriers for shading buffers after rays are finished.
        RenderTextureBarrier afterDispatchBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterDispatchBarriers, uint32_t(std::size(afterDispatchBarriers)));

        // Set the output as the current render target.
        worker->commandList->setFramebuffer(rtResources->outputFramebuffer[rtResources->swapBuffers ? 1 : 0].get());

        // Apply the scissor and viewport to the size of the output texture.
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(rtResources->textureWidth), float(rtResources->textureHeight)));
        worker->commandList->setScissors(RenderRect(0, 0, rtResources->textureWidth, rtResources->textureHeight));

        // Draw the raytracing output.
        RT64_LOG_PRINTF("Composing the raytracing output");
        const ShaderRecord &composeShader = shaderLibrary->compose;
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->setPipeline(composeShader.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(composeShader.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->composeSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Switch resources to the correct states after composing the image
        RenderTextureBarrier afterComposeBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, afterComposeBarriers, uint32_t(std::size(afterComposeBarriers)));

        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;
        if (lumaActive) {
            const uint32_t ThreadGroupWorkRegionDim = 8;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the downscaling shader");
            {
                // Execute the compute shader for downscaling the image.
                const ShaderRecord &bicubicScaling = shaderLibrary->bicubicScaling;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::BicubicCB bicubicCB;
                bicubicCB.InputResolution = { rtResources->textureWidth, rtResources->textureHeight };
                bicubicCB.OutputResolution = { rtResources->textureWidth / 8, rtResources->textureHeight / 8 };
                worker->commandList->setPipeline(bicubicScaling.pipeline.get());
                worker->commandList->setComputePipelineLayout(bicubicScaling.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &bicubicCB);
                worker->commandList->setComputeDescriptorSet(rtResources->downscaleSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }


            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the luminance histogram shader");
            {
                // Execute the compute shader for the luminance histogram.
                const ShaderRecord &luminanceHistogram = shaderLibrary->luminanceHistogram;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::LuminanceHistogramCB histogramCB;
                histogramCB.inputWidth = rtResources->textureWidth / 8;
                histogramCB.inputHeight = rtResources->textureHeight / 8;
                histogramCB.minLuminance = rtScene.presetScene.minLuminance;
                histogramCB.oneOverLuminanceRange = 1.0f / rtScene.presetScene.luminanceRange;
                worker->commandList->setPipeline(luminanceHistogram.pipeline.get());
                worker->commandList->setComputePipelineLayout(luminanceHistogram.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &histogramCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));

            RT64_LOG_PRINTF("Do the luminance average shader");
            {
                // Execute the compute shader for the luminance histogram average.
                const ShaderRecord &histogramAverage = shaderLibrary->histogramAverage;
                interop::HistogramAverageCB averageCB;
                averageCB.pixelCount = (rtResources->textureWidth / 8) * (rtResources->textureHeight / 8);
                averageCB.minLuminance = rtScene.presetScene.minLuminance;
                averageCB.luminanceRange = rtScene.presetScene.luminanceRange;
                averageCB.timeDelta = rtScene.deltaTime;
                averageCB.tau = rtScene.presetScene.lumaUpdateTime;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramAverage.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramAverage.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &averageCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaAvgSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }

            RT64_LOG_PRINTF("Do the histogram clear shader");
            {
                // Execute the compute shader for clearing the luminance histogram.
                const ShaderRecord &histogramClear = shaderLibrary->histogramClear;
                worker->commandList->setPipeline(histogramClear.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramClear.pipelineLayout.get());
                worker->commandList->setComputeDescriptorSet(rtResources->lumaClearSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
            }
        }
        else {
            RT64_LOG_PRINTF("Do the histogram set shader");
            {
                // Execute the compute shader for setting the luminance value.
                const ShaderRecord &histogramSet = shaderLibrary->histogramSet;
                interop::HistogramSetCB setCB;
                setCB.luminanceValue = rtScene.presetScene.minLuminance;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramSet.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramSet.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &setCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSetSet->get(), 0);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }
        }

        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        const bool upscalerActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (upscalerActive) {
            thread_local std::vector<RenderTextureBarrier> beforeBarriers;
            thread_local std::vector<RenderTextureBarrier> afterBarriers;
            beforeBarriers.clear();
            afterBarriers.clear();

            beforeBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::GENERAL));
            afterBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));
            RenderTexture *rtDepthCur = rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get();
            if (upscaler->requiresNonShaderResourceInputs()) {
                for (RenderTexture *res : { rtOutputCur, rtResources->flowTexture.get(), rtResources->reactiveMaskTexture.get(), rtResources->lockMaskTexture.get(), rtDepthCur }) {
                    beforeBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                    afterBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                }
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, beforeBarriers);

            Upscaler::UpscaleParameters params;
            params.inRect = { 0, 0, static_cast<int>(rtResources->textureWidth), static_cast<int>(rtResources->textureHeight) };
            params.inDiffuseAlbedo = rtResources->diffuseTexture.get();
            params.inSpecularAlbedo = rtResources->shadingSpecularTexture.get();
            params.inNormalRoughness = rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get();
            params.inColor = rtOutputCur;
            params.inFlow = rtResources->flowTexture.get();
            params.inReactiveMask = rtResources->upscalerReactiveMask ? rtResources->reactiveMaskTexture.get() : nullptr;
            params.inLockMask = rtResources->upscalerLockMask ? rtResources->lockMaskTexture.get() : nullptr;
            params.inDepth = rtDepthCur;
            params.outColor = rtResources->upscaledOutputTexture.get();
            params.jitterX = -rtResources->rtParams.pixelJitter.x;
            params.jitterY = -rtResources->rtParams.pixelJitter.y;
            params.deltaTime = rtScene.deltaTime * 1000.0f;
            params.nearPlane = rtResources->rtParams.nearDist;
            params.farPlane = rtResources->rtParams.farDist;
            params.fovY = rtResources->rtParams.fovRadians;
            params.resetAccumulation = false; // TODO: Make this configurable via the API.
            upscaler->upscale(worker, params);

            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarriers);
        }

        // Set the final render target. Apply the same scissor and viewport that was determined for the raytracing step.
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(rtScene.viewport);
        worker->commandList->setScissors(rtScene.scissor);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);

        // Draw final output.
        const ShaderRecord &postProcess = shaderLibrary->postProcess;
        worker->commandList->setPipeline(postProcess.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(postProcess.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->postProcessSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Draw debug view on top.
        if (rtResources->rtParams.visualizationMode != interop::VisualizationMode::Final) {
            const ShaderRecord &debugShader = shaderLibrary->debug;
            worker->commandList->setPipeline(debugShader.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(debugShader.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }

        // Mark targets for resolve.
        colorTarget->markForResolve();
    }
#endif

    void FramebufferRenderer::submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers) {
        if (rspSmoothNormalVector.empty()) {
            return;
        }
        
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(outputBuffers->worldNormBuffer.buffer.get(), RenderBufferAccess::WRITE));

        const int ThreadGroupSize = 64;
        const auto &rspSmoothNormal = shaderLibrary->rspSmoothNormal;
        for (const RSPSmoothNormalGenerationCB &cb : rspSmoothNormalVector) {
            const uint32_t triangleCount = cb.indexCount / 3;
            const uint32_t dispatchCount = (triangleCount + ThreadGroupSize - 1) / ThreadGroupSize;
            worker->commandList->setPipeline(rspSmoothNormal.pipeline.get());
            worker->commandList->setComputePipelineLayout(rspSmoothNormal.pipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &cb);
            worker->commandList->setComputeDescriptorSet(smoothDescSet->get(), 0);
            worker->commandList->dispatch(dispatchCount, 1, 1);
        }
    }

    void FramebufferRenderer::recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor,
        VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled) {
        if (!dummyColorTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyColorTarget.get(), RenderTextureLayout::SHADER_READ));
            dummyColorTargetTransitioned = true;
        }

        if (!dummyDepthTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ));
            dummyDepthTargetTransitioned = true;
        }

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListBeforeBarriers(worker);
        }

        shaderUploader->commandListBeforeBarriers(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListCopyResources(worker);
        }

        shaderUploader->commandListCopyResources(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListAfterBarriers(worker);
        }

        if (rspProcessor != nullptr) {
            rspProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
        }

        if (vertexProcessor != nullptr) {
            vertexProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
        }

#   if RT_ENABLED
        if (rtEnabled) {
            assert(rtResources != nullptr);
            if (!rtResources->bottomLevelASVector.empty()) {
                if (!rspSmoothNormalVector.empty()) {
                    submitRSPSmoothNormalCompute(worker, outputBuffers);
                }

                rtResources->submitBottomLevelASCreation(worker);
                rtResources->submitTopLevelASCreation(worker);
            }
        }
#   endif

        shaderUploader->commandListAfterBarriers(worker);
        pendingUploaders = bufferUploaders;
        recordJfgShadows(worker); // JFG PORT: after the world positions and the uploads

        if (!dynamicTextureBarrierVector.empty()) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, dynamicTextureBarrierVector);
        }
    }

    void FramebufferRenderer::recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex) {
        // Submit all transition barriers first.
        thread_local std::vector<RenderTextureBarrier> startBarriers;
        startBarriers.clear();

        const Framebuffer &framebuffer = framebufferVector[framebufferIndex];
        for (RenderTarget *target : framebuffer.transitionRenderTargetSet) {
            startBarriers.emplace_back(RenderTextureBarrier(target->getResolvedTexture(), RenderTextureLayout::SHADER_READ));
        }
        
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        RenderTarget *colorTarget = targetDrawCall.fbStorage->colorTarget;
        RenderTarget *depthTarget = targetDrawCall.fbStorage->depthTarget;
        if (colorTarget != nullptr) {
            startBarriers.emplace_back(RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
        }

        startBarriers.emplace_back(RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, startBarriers);

        bool depthState = false;
        worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorDepthWrite.get());
        for (const auto &pair : targetDrawCall.sceneIndices) {
#       if RT_ENABLED
            if (pair.second) {
                const auto &rtScene = targetDrawCall.rtScenes[pair.first];

                // Draw all the interleaved rasterized buffers that will be used in the render target.
                thread_local std::vector<RenderTextureBarrier> interleavedBarriers;
                interleavedBarriers.clear();
                for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                    bool interleavedDepthState = false;
                    const uint32_t sceneIndex = rtScene.interleavedRasters[i].rasterSceneIndex;
                    RenderTarget *colorRenderTarget = rtResources->interleavedColorTargetVector[i].get();
                    RenderTarget *depthRenderTarget = rtResources->interleavedDepthTargetVector[i].get();
                    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                        RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
                        RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE)
                    });

                    RenderFramebufferStorage *fbStorage = rtResources->interleavedFramebufferStorageVector[i].get();
                    worker->commandList->setFramebuffer(fbStorage->colorDepthWrite.get());
                    worker->commandList->clearColor();
                    worker->commandList->clearDepth();

                    submitDepthAccess(worker, fbStorage, false, interleavedDepthState);
                    submitRasterScene(worker, framebuffer, fbStorage, targetDrawCall.rasterScenes[sceneIndex], interleavedDepthState);

                    // Resolve the interleaved scene.
                    // TODO: Depth textures need to be thrown into a separate view vector for multisampled textures.
                    fbStorage->colorTarget->resolveTarget(worker, shaderLibrary);

                    interleavedBarriers.emplace_back(RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::SHADER_READ));
                    interleavedBarriers.emplace_back(RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_READ));
                }

                if (!interleavedBarriers.empty()) {
                    worker->commandList->barriers(RenderBarrierStage::COMPUTE, interleavedBarriers);
                }

                submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
                submitRaytracingScene(worker, targetDrawCall.fbStorage->colorTarget, rtScene);
            }
            else
#       endif
            {
                const RasterScene &rasterScene = targetDrawCall.rasterScenes[pair.first];
                submitDepthAccess(worker, targetDrawCall.fbStorage, false, depthState);
                submitRasterScene(worker, framebuffer, targetDrawCall.fbStorage, rasterScene, depthState);
            }
        }
    }

    void FramebufferRenderer::waitForUploaders() {
        shaderUploader->wait();

        for (BufferUploader *uploader : pendingUploaders) {
            uploader->wait();
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        assert(rtResources != nullptr);
        rtResources->setRaytracingConfig(rtConfig, resolutionChanged);
    }
#endif

    void FramebufferRenderer::addFramebuffer(const DrawParams &p) {
        assert(p.fbStorage != nullptr);
        
        // Setup framebuffer pair data and descriptor set.
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        interop::FramebufferParams fbParams;
        fbParams.resolution = { p.targetWidth / p.resolutionScale.x, p.targetHeight / p.resolutionScale.y };
        fbParams.resolutionScale = p.resolutionScale;
        fbParams.horizontalMisalignment = p.horizontalMisalignment;
        framebufferCount++;

        while (framebufferCount > framebufferVector.size()) {
            framebufferVector.emplace_back();
            framebufferVector.back().paramsBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(256, RenderBufferFlag::CONSTANT));
            framebufferVector.back().descRealFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
            framebufferVector.back().descDummyFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
        }

        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        void *paramBufferBytes = framebuffer.paramsBuffer->map();
        memcpy(paramBufferBytes, &fbParams, sizeof(interop::FramebufferParams));
        framebuffer.paramsBuffer->unmap();

        RenderTexture *backgroundColorTexture = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTexture() : dummyColorTarget.get();
        RenderTextureView *backgroundColorTextureView = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTextureView() : dummyColorTargetView.get();
        framebuffer.descRealFbSet->setBuffer(framebuffer.descRealFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        framebuffer.descDummyFbSet->setBuffer(framebuffer.descDummyFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundDepth, dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ, dummyDepthTargetView.get());

        // Store ubershader and other effect pipelines.
        const RasterShaderUber *rasterShaderUber = p.rasterShaderCache->getGPUShaderUber();
        rendererPipelineLayout = rasterShaderUber->pipelineLayout.get();
        postBlendDitherNoiseAddPipeline = rasterShaderUber->postBlendDitherNoiseAddPipeline.get();
        postBlendDitherNoiseSubPipeline = rasterShaderUber->postBlendDitherNoiseSubPipeline.get();
        postBlendDitherNoiseSubNegativePipeline = rasterShaderUber->postBlendDitherNoiseSubNegativePipeline.get();
        ssaoPipeline = rasterShaderUber->ssaoPipeline.get();
        fogPipeline = rasterShaderUber->fogPipeline.get();
        vignettePipeline = rasterShaderUber->vignettePipeline.get();
        jfgShadowPipeline = rasterShaderUber->jfgShadowPipeline.get(); // JFG PORT
        bloomPipelineLayout = rasterShaderUber->bloomPipelineLayout.get();
        bloomExtractPipeline = rasterShaderUber->bloomExtractPipeline.get();
        bloomBlurPipeline = rasterShaderUber->bloomBlurPipeline.get();
        bloomCompositePipeline = rasterShaderUber->bloomCompositePipeline.get();
        ssaoBlurPipeline = rasterShaderUber->ssaoBlurPipeline.get();

        // Make a new render target draw call.
        RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        const DrawData &drawData = p.curWorkload->drawData;
        const DrawBuffers &drawBuffers = p.curWorkload->drawBuffers;
        const OutputBuffers &outputBuffers = p.curWorkload->outputBuffers;
        const RenderBuffer *screenPosRes = outputBuffers.screenPosBuffer.buffer.get();
        const RenderBuffer *tcRes = outputBuffers.genTexCoordBuffer.buffer.get();
        const RenderBuffer *indexRes = drawBuffers.faceIndicesBuffer.get();
        const RenderBuffer *shadedColRes = outputBuffers.shadedColBuffer.buffer.get();
        const RenderBuffer *worldPosRes = outputBuffers.worldPosBuffer.buffer.get();
        const RenderBuffer *worldNormRes = outputBuffers.worldNormBuffer.buffer.get();
        const RenderBuffer *worldVelRes = outputBuffers.worldVelBuffer.buffer.get();
        const RenderBuffer *triPosRes = drawBuffers.triPosBuffer.get();
        const RenderBuffer *triTcRes = drawBuffers.triTcBuffer.get();
        const RenderBuffer *triColRes = drawBuffers.triColorBuffer.get();
        const uint32_t PosStride = sizeof(float) * 4;
        const uint32_t ColStride = sizeof(float) * 4;
        const uint32_t WorldNormStride = sizeof(float) * 4;
        const uint32_t WorldVelStride = sizeof(float) * 4;
        const uint32_t TcStride = sizeof(float) * 2;
        const uint32_t IndexStride = sizeof(uint32_t);
        const uint32_t vertexCount = drawData.vertexCount();
        const uint32_t indexCount = uint32_t(drawData.faceIndices.size());
        const uint32_t rawTriVertexCount = drawData.rawTriVertexCount();
        vertexInputSlots[0] = RenderInputSlot(0, PosStride);
        vertexInputSlots[1] = RenderInputSlot(1, TcStride);
        vertexInputSlots[2] = RenderInputSlot(2, ColStride);
        indexedVertexViews[0] = RenderVertexBufferView(RenderBufferReference(screenPosRes), PosStride * vertexCount);
        indexedVertexViews[1] = RenderVertexBufferView(RenderBufferReference(tcRes), TcStride * vertexCount);
        indexedVertexViews[2] = RenderVertexBufferView(RenderBufferReference(shadedColRes), ColStride * vertexCount);
        indexBufferView = RenderIndexBufferView(RenderBufferReference(indexRes), IndexStride * indexCount, RenderFormat::R32_UINT);
        rawVertexViews[0] = RenderVertexBufferView(RenderBufferReference(triPosRes), PosStride * rawTriVertexCount);
        rawVertexViews[1] = RenderVertexBufferView(RenderBufferReference(triTcRes), TcStride * rawTriVertexCount);
        rawVertexViews[2] = RenderVertexBufferView(RenderBufferReference(triColRes), ColStride * rawTriVertexCount);
        testZIndexBuffer = outputBuffers.testZIndexBuffer.buffer.get();
        testZIndexBufferView = RenderIndexBufferView(testZIndexBuffer, uint32_t(outputBuffers.testZIndexBuffer.allocatedSize), RenderFormat::R32_UINT);

        RasterScene rasterScene;
        auto checkRasterScene = [&](RasterScene &rasterScene) {
            if (!rasterScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size());
                targetDrawCall.rasterScenes.push_back(rasterScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, false });
                rasterScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rasterScenes.clear();

#   if RT_ENABLED
        RaytracingScene rtScene;
        auto checkRtScene = [&](RaytracingScene &rtScene) {
            if (!rtScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rtScenes.size());
                targetDrawCall.rtScenes.push_back(rtScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, true });
                rtScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rtScenes.clear();
#   endif

        targetDrawCall.fbStorage = p.fbStorage;
        targetDrawCall.sceneIndices.clear();

        const float SimilarityPercentage = 0.1f; // TODO: Make more strict once VI ratios are in.
        const float scissorRatio = static_cast<float>(fbPair.scissorRect.width(false, true)) / static_cast<float>(fbPair.scissorRect.height(false, true));
        const bool adjustRatio = (abs((scissorRatio / p.aspectRatioSource) - 1.0f) < SimilarityPercentage);
        const float aspectRatioScale = adjustRatio ? (p.aspectRatioTarget / p.aspectRatioSource) : 1.0f;
        InstanceDrawCall instanceDrawCall;
        interop::RenderIndices renderIndices;
        uint32_t globalCallIndex = 0;
        const float wideWidth = p.fbWidth * p.resolutionScale.x;
        const float originalWidth = p.fbWidth * p.resolutionScale.y;
        const float commonHeight = float(p.targetHeight);
        framebuffer.viewport = RenderViewport(0.0f, 0.0f, wideWidth, commonHeight);
        
        const interop::float2 halfViewportSize = { framebuffer.viewport.width / 2.0f, framebuffer.viewport.height / 2.0f };
        const interop::float2 halfPixelOffset = { 1.0f / framebuffer.viewport.width, -1.0f / framebuffer.viewport.height };
        const float middleViewport = (wideWidth / 2.0f) - (originalWidth / 2.0f);
        const float extOriginPercentage = p.extAspectPercentage;
        uint32_t vertexTestZFaceIndicesStart = 0;
        int32_t vertexTestZCallIndex = -1;
        RenderViewport viewportClip;
        for (uint32_t pr = 0; (pr < fbPair.projectionCount) && (globalCallIndex < p.maxGameCall); pr++) {
            const Projection &proj = fbPair.projections[pr];
            if (proj.scissorRect.isNull()) {
                continue;
            }

#       if RT_ENABLED
            // TODO: Move heuristics of RT proj elsewhere?
            // TODO: Use detected scenes logic instead.
            const bool perspProj = (proj.type == Projection::Type::Perspective);
            bool rtProj = p.rtEnabled && perspProj && fbPair.depthWrite && targetDrawCall.rtScenes.empty(); // TODO: Remove the last condition once multiple heaps per RT scene are supported.

            // Make sure the matrices are compatible if we're switching to a new projection.
            bool rtProjCompatible = true;
            if (rtProj && (!rtScene.instanceIndices.empty())) {
                const float Threshold = 1e-6f;
                const float viewMatrixDiff = matrixDifference(drawData.modViewTransforms[proj.transformsIndex], rtScene.curViewMatrix);
                const float projMatrixDiff = matrixDifference(drawData.modProjTransforms[proj.transformsIndex], rtScene.curProjMatrix);
                rtProjCompatible = (viewMatrixDiff < Threshold) && (projMatrixDiff < Threshold);
            }

            // TODO: Remove this condition once multiple heaps per RT scene are supported.
            if (rtProj && !rtProjCompatible) {
                rtProj = false;
            }
#       endif
            
            auto &triangles = instanceDrawCall.triangles;
            triangles.screenScale = { 1.0f, 1.0f };
            triangles.screenOffset = halfPixelOffset;

            // JFG PORT: field of view and depth range of this projection, for the ambient occlusion pass. From a guPerspective
            // matrix (all elements times its scale): m[1][1] = s / tan(fovy / 2), m[2][3] = -s, m[2][2] = s (n + f) / (n - f),
            // m[3][2] = s 2nf / (n - f).
            float projTanHalfFovY = 0.0f, projNear = 0.0f, projFar = 0.0f;
            if ((proj.type == Projection::Type::Perspective) && (proj.transformsIndex < drawData.projTransforms.size())) {
                const interop::float4x4 &m = drawData.projTransforms[proj.transformsIndex];
                const float s = -m[2][3];
                if ((s > 1e-6f) && (fabsf(m[1][1]) > 1e-6f)) {
                    const float m22 = m[2][2] / s, m32 = m[3][2] / s;
                    const float n = m32 / (m22 - 1.0f), f = m32 / (m22 + 1.0f);
                    if ((n > 0.0f) && (f > n) && std::isfinite(f)) {
                        projTanHalfFovY = s / fabsf(m[1][1]);
                        projNear = n;
                        projFar = f;
                    }
                }
            }

            float projInvRatioScale = 1.0f / aspectRatioScale;
            const int16_t *viewportClipRatios = &drawData.viewportClipRatios[proj.transformsIndex * 4];
            const uint16_t viewportOrigin = drawData.viewportOrigins[proj.transformsIndex];
            if (proj.usesViewport()) {
                // The call's scissor spans the whole width of the framebuffer pair scissor. Custom origin must not be in use to be able to use the stretched viewport.
                const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                FixedRect intersectionRect = proj.scissorRect.intersection(viewport.rect(viewportClipRatios));
                bool coversWholeWidth = !intersectionRect.isEmpty() && (intersectionRect.ulx <= fbPair.scissorRect.ulx) && (intersectionRect.lrx + 4 >= fbPair.scissorRect.lrx); // JFG PORT: see rt64_projection_processor.cpp
                bool horizontalRatio = !intersectionRect.isEmpty() && (intersectionRect.width(true, true) > intersectionRect.height(true, true));
                bool useWideViewport = (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;
                // JFG PORT: a narrow projection whose shared matrix the projection processor already widened would be
                // squeezed twice (the Tribal counter's heads landed on their numbers). Its matrix already keeps 4:3.
                const bool sharedWideMatrix = (proj.transformsIndex < drawData.jfgProjAdjusted.size()) && (drawData.jfgProjAdjusted[proj.transformsIndex] != 0);
                if (useWideViewport) {
                    projInvRatioScale = 1.0f;
                }
                else if (!sharedWideMatrix) {
                    triangles.screenScale.x = originalWidth / wideWidth;

                    if (viewportOrigin < G_EX_ORIGIN_NONE) {
                        const float centerOffset = ((middleViewport * viewportOrigin) / G_EX_ORIGIN_CENTER) * extOriginPercentage + middleViewport * (1.0f - extOriginPercentage);
                        triangles.screenOffset.x = halfPixelOffset.x + ((centerOffset - middleViewport) / halfViewportSize.x);
                    }
                }

                viewportClip = convertViewportRect(viewport.rect(viewportClipRatios), p.resolutionScale, p.fbWidth, projInvRatioScale, extOriginPercentage, 0.0f, viewportOrigin, viewportOrigin);
            }

            for (uint32_t d = 0; (d < proj.gameCallCount) && (globalCallIndex < p.maxGameCall); d++) {
                const GameCall &call = proj.gameCalls[d];
                renderIndices.instanceIndex = call.callDesc.callIndex;
                renderIndices.faceIndicesStart = call.meshDesc.faceIndicesStart;
                renderIndices.rdpTileIndex = call.callDesc.tileIndex;
                renderIndices.rdpTileCount = call.callDesc.tileCount;
                renderIndices.highlightColor = call.debuggerDesc.highlightColor;
                renderIndicesVector.push_back(renderIndices);

                uint32_t cycleType = call.callDesc.otherMode.cycleType();
                if (cycleType == G_CYC_FILL) {
                    instanceDrawCall.type = InstanceDrawCall::Type::FillRect;

                    auto &clearRect = instanceDrawCall.clearRect;
                    if (call.debuggerDesc.highlightColor > 0) {
                        clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.debuggerDesc.highlightColor));
                    }
                    else {
                        if (p.fbStorage->colorTarget == nullptr) {
                            clearRect.depth = ColorConverter::D16::toF(call.callDesc.fillColor & 0xFFFF);
                        }
                        else if (fbPair.colorImage.siz == G_IM_SIZ_32b) {
                            clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.callDesc.fillColor));
                        }
                        else {
                            clearRect.color = toRenderColor(ColorConverter::RGBA16::toRGBCVGF(call.callDesc.fillColor & 0xFFFF, shaderLibrary->usesHDR));
                        }
                    }

                    float invRatioScale = 1.0f / aspectRatioScale;
                    int32_t horizontalMisalignment = 0;

                    // A rect that spans the whole width of the scissor.
                    // JFG PORT: one pixel short counts too (see rt64_projection_processor.cpp). The reflective floors' picture is cleared
                    // with a fill to 319 under a scissor to 320; kept at 4:3, its sides were never cleared and the floors showed a 4:3 box.
                    if ((call.callDesc.rect.ulx <= fbPair.scissorRect.ulx) && (call.callDesc.rect.lrx + 4 >= fbPair.scissorRect.lrx)) {
                        invRatioScale = 1.0f;
                    }
                    // A regular rectangle that should correct its misalignment.
                    else {
                        horizontalMisalignment = int32_t(p.horizontalMisalignment);
                    }

                    clearRect.rect = convertFixedRect(call.callDesc.rect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                }
                else if (call.callDesc.extendedType != DrawExtendedType::None) {
                    switch (call.callDesc.extendedType) {
                    case DrawExtendedType::VertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::VertexTestZ;
                        instanceDrawCall.vertexTestZ.vertexIndex = call.callDesc.extendedData.vertexTestZ.vertexIndex;
                        instanceDrawCall.vertexTestZ.resolutionScale = p.resolutionScale;
                        instanceDrawCall.vertexTestZ.srcIndexStart = call.meshDesc.faceIndicesStart + 3;
                        instanceDrawCall.vertexTestZ.dstIndexStart = vertexTestZFaceIndicesStart;
                        instanceDrawCall.vertexTestZ.indexCount = 0;
                        vertexTestZCallIndex = int32_t(instanceDrawCallVector.size());
                        break;
                    case DrawExtendedType::EndVertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::Unknown;
                        vertexTestZCallIndex = -1;
                        break;
                    default:
                        assert(false && "Unknown extended type.");
                        break;
                    }
                }
                else {
#               if RT_ENABLED
                    if (rtProj) {
                        instanceDrawCall.type = InstanceDrawCall::Type::Raytracing;

                        if (hitGroupVector.empty()) {
                            // TODO: Support specialized shaders.
                            //const RaytracingShaderPrograms &shaderPrograms = p.ubershadersOnly ? rtState->shaderProgramsMap.find(UberShaderHash)->second : rtState->getShaderPrograms(call.shaderDesc);
                            const RaytracingShaderPrograms &shaderPrograms = rtState->shaderProgramsMap.find(UberShaderHash)->second;
                            hitGroupVector.emplace_back(shaderPrograms.surface);
                            hitGroupVector.emplace_back(shaderPrograms.shadow);
                        }

                        auto &raytracing = instanceDrawCall.raytracing;
                        raytracing.hitGroupIndex = 0;
                        raytracing.cullDisable = !call.shaderDesc.flags.culling;

                        const interop::OtherMode &otherMode = call.callDesc.otherMode;
                        raytracing.queryMask = (otherMode.zCmp() || otherMode.zUpd()) ? DepthRayQueryMask : NoDepthRayQueryMask;
                        if (drawData.extraParams[call.callDesc.callIndex].shadowCatcherFactor > 0.0f) {
                            raytracing.queryMask |= ShadowCatcherRayQueryMask;
                        }

                        const RenderBottomLevelASMesh asMesh(indexRes->at(call.meshDesc.faceIndicesStart *IndexStride), worldPosRes->at(0), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, call.callDesc.triangleCount * 3, vertexCount, PosStride, false);
                        rtResources->addBottomLevelASMesh(asMesh);

                        if (false) { // TODO: call.shaderDesc.flags.smoothNormal
                            RSPSmoothNormalGenerationCB rspSmoothNormal;
                            rspSmoothNormal.indexStart = call.meshDesc.faceIndicesStart;
                            rspSmoothNormal.indexCount = call.callDesc.triangleCount * 3;
                            rspSmoothNormalVector.push_back(rspSmoothNormal);
                        }
                    }
                    else 
#               endif
                    {
                        triangles.shaderDesc = call.shaderDesc;

                        RasterShader *gpuShader = p.ubershadersOnly ? nullptr : p.rasterShaderCache->getGPUShader(call.shaderDesc);
                        if (gpuShader != nullptr) {
                            triangles.pipeline = gpuShader->pipeline.get();
                        }
                        else {
                            const bool copyMode = (call.shaderDesc.otherMode.cycleType() == G_CYC_COPY);
                            triangles.pipeline = rasterShaderUber->getPipeline(
                                !copyMode && call.shaderDesc.otherMode.zCmp() && (call.shaderDesc.otherMode.zMode() != ZMODE_DEC),
                                !copyMode && call.shaderDesc.otherMode.zUpd(),
                                (call.shaderDesc.otherMode.cvgDst() == CVG_DST_WRAP) || (call.shaderDesc.otherMode.cvgDst() == CVG_DST_SAVE));
                        }
                        
                        triangles.faceCount = call.callDesc.triangleCount;
                        triangles.vertexTestZ = (vertexTestZCallIndex >= 0);
                        triangles.postBlendDitherNoise = false;
                        triangles.jfgTanHalfFovY = projTanHalfFovY;
                        triangles.jfgNear = projNear;
                        triangles.jfgFar = projFar;
                        triangles.jfgLevel = call.callDesc.jfgLevel; // JFG PORT

                        float invRatioScale = 1.0f / aspectRatioScale;
                        float horizontalMisalignment = 0.0f;
                        switch (proj.type) {
                        case Projection::Type::Perspective:
                        case Projection::Type::Orthographic: {
                            instanceDrawCall.type = InstanceDrawCall::Type::IndexedTriangles;
                            triangles.indexStart = triangles.vertexTestZ ? vertexTestZFaceIndicesStart : call.meshDesc.faceIndicesStart;
                            invRatioScale = projInvRatioScale;

                            // JFG PORT: what may go into the lights' shadow maps: the level, and 3D draws that write
                            // depth (characters, objects; not glows, flames or smoke), kept if they turn out to be seen
                            // through the level's camera (the HUD's 3D parts have their own).
                            const uint32_t shadowStart = call.meshDesc.faceIndicesStart;
                            const uint32_t shadowEnd = shadowStart + call.callDesc.triangleCount * 3;
                            const bool shadowWorld = (proj.type == Projection::Type::Perspective) && (call.callDesc.jfgLevel || call.shaderDesc.otherMode.zUpd());
                            if (!triangles.vertexTestZ && shadowWorld && (shadowEnd > shadowStart) && (shadowEnd <= drawData.faceIndices.size())) {
                                jfgShadowCandidates.push_back({ shadowStart, shadowEnd, proj.transformsIndex, call.callDesc.jfgLevel });
                            }
                            break;
                        }
                        case Projection::Type::Rectangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RegularRect;
                            triangles.indexStart = call.meshDesc.rawVertexStart;

                            bool tileCopiesUsed = false;
                            for (uint32_t t = 0; (t < call.callDesc.tileCount) && !tileCopiesUsed; t++) {
                                tileCopiesUsed = drawData.callTiles[call.callDesc.tileIndex + t].tileCopyUsed;
                            }

                            // The call's scissor spans the whole width of the framebuffer pair scissor. The rect must not be using extended origins.
                            const bool regularOrigins = (call.callDesc.rectLeftOrigin == G_EX_ORIGIN_NONE) && (call.callDesc.rectRightOrigin == G_EX_ORIGIN_NONE);
                            const bool coversScissorWidth = regularOrigins && (call.callDesc.rect.ulx <= fbPair.scissorRect.ulx) && (call.callDesc.rect.lrx >= fbPair.scissorRect.lrx);
                            if ((tileCopiesUsed || coversScissorWidth || call.callDesc.rectAspect == G_EX_ASPECT_STRETCH) && (call.callDesc.rectAspect != G_EX_ASPECT_ADJUST)) {
                                invRatioScale = 1.0f;
                            }
                            else {
                                horizontalMisalignment = p.horizontalMisalignment;
                            }

                            RenderViewport viewportRect = convertViewportRect(call.callDesc.rect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                            triangles.screenScale = { viewportRect.width / framebuffer.viewport.width, viewportRect.height / framebuffer.viewport.height };
                            triangles.screenOffset.x = halfPixelOffset.x + ((viewportRect.x + viewportRect.width / 2.0f) - halfViewportSize.x) / halfViewportSize.x;
                            triangles.screenOffset.y = halfPixelOffset.y + (halfViewportSize.y - (viewportRect.y + viewportRect.height / 2.0f)) / halfViewportSize.y;

                            if (p.postBlendNoise) {
                                // Indicate if post blend dither noise should be applied.
                                bool rgbDitherNoise = (call.shaderDesc.otherMode.rgbDither() == G_CD_NOISE);
                                triangles.postBlendDitherNoise = rgbDitherNoise && !call.shaderDesc.otherMode.zCmp() && !call.shaderDesc.otherMode.zUpd();
                                triangles.postBlendDitherNoiseNegative = p.postBlendNoiseNegative;
                            }

                            break;
                        }
                        case Projection::Type::Triangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RawTriangles;
                            triangles.indexStart = call.meshDesc.rawVertexStart;
                            break;
                        }
                        case Projection::Type::None:
                        default:
                            break;
                        }

                        // Scissor aspect can override the scaling determined by the draw call.
                        float scissorInvRatioScale = invRatioScale;
                        float scissorHorizontalMisalignment = horizontalMisalignment;
                        if (call.callDesc.scissorAspect == G_EX_ASPECT_STRETCH) {
                            scissorInvRatioScale = 1.0f;
                            scissorHorizontalMisalignment = 0.0f;
                        }
                        else if (call.callDesc.scissorAspect == G_EX_ASPECT_ADJUST) {
                            scissorInvRatioScale = 1.0f / aspectRatioScale;
                            scissorHorizontalMisalignment = p.horizontalMisalignment;
                        }

                        triangles.scissor = convertFixedRect(call.callDesc.scissorRect, p.resolutionScale, p.fbWidth, scissorInvRatioScale, extOriginPercentage, int32_t(scissorHorizontalMisalignment), call.callDesc.scissorLeftOrigin, call.callDesc.scissorRightOrigin);

                        bool usesViewport = (proj.type == Projection::Type::Perspective) || (proj.type == Projection::Type::Orthographic);
                        if (usesViewport) {
                            triangles.scissor = viewportScissorIntersection(viewportClip, triangles.scissor);
                        }
                        
                        if (triangles.vertexTestZ && usesViewport) {
                            instanceDrawCallVector[vertexTestZCallIndex].vertexTestZ.indexCount += call.callDesc.triangleCount * 3;
                            vertexTestZFaceIndicesStart += call.callDesc.triangleCount * 3;
                        }
                    }
                }

                // Determine to use the draw call either in the RT scene or the raster scene.
                const uint32_t instanceIndex = static_cast<uint32_t>(instanceDrawCallVector.size());
#           if RT_ENABLED
                bool rtCall = instanceDrawCall.type == InstanceDrawCall::Type::Raytracing;
                if (rtCall) {
                    // If the current scene is not compatible, we submit it before the raster scene.
                    if (!rtProjCompatible) {
                        checkRtScene(rtScene);
                    }

                    const bool addedRasterScene = checkRasterScene(rasterScene);
                    if (rtScene.instanceIndices.empty()) {
                        float projRatioScale = 1.0f / aspectRatioScale;
                        float invRatioScale = 1.0f / aspectRatioScale;
                        const bool coversScissorWidth = (proj.scissorRect.ulx <= fbPair.scissorRect.ulx) && (proj.scissorRect.lrx >= fbPair.scissorRect.lrx);
                        if (coversScissorWidth) {
                            invRatioScale = 1.0f;
                        }
                        else {
                            projRatioScale = 1.0f;
                        }

                        rtScene.curViewMatrix = drawData.modViewTransforms[proj.transformsIndex];
                        rtScene.curProjMatrix = drawData.modProjTransforms[proj.transformsIndex];
                        rtScene.prevViewMatrix = drawData.prevViewTransforms[proj.transformsIndex];
                        rtScene.prevProjMatrix = drawData.prevProjTransforms[proj.transformsIndex];

                        const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                        rtScene.viewport = convertViewportRect(viewport.rect(viewportClipRatios), p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0.0f, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
                        rtScene.scissor = convertFixedRect(proj.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);

                        rtScene.presetScene = p.presetScene;
                        rtScene.screenWidth = lround(static_cast<float>(p.fbWidth) * p.resolutionScale.x);
                        rtScene.screenHeight = lround(static_cast<float>(p.fbHeight) * p.resolutionScale.y);

                        if (proj.pointLightCount > 0) {
                            rtScene.pointLights = proj.pointLights.data();
                            rtScene.lightCount = proj.pointLightCount;
                        }
                        else {
                            rtScene.pointLights = nullptr;
                            rtScene.lightCount = 0;
                        }
                    }
                    else if (rtProjCompatible && addedRasterScene) {
                        targetDrawCall.sceneIndices.pop_back();
                        const uint32_t rasterSceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size() - 1);
                        const auto &rasterScene = targetDrawCall.rasterScenes[rasterSceneIndex];
                        rtScene.interleavedRasters.push_back({ rasterSceneIndex, rasterScene.instanceIndices.back(), 0, 0 });
                    }

                    rtScene.instanceIndices.push_back(instanceIndex);
                }
                else 
#           endif
                {
                    rasterScene.instanceIndices.push_back(instanceIndex);
                }

                instanceDrawCallVector.push_back(instanceDrawCall);
                globalCallIndex++;
            }
        }

#   if RT_ENABLED
        checkRtScene(rtScene);
#   endif
        checkRasterScene(rasterScene);

        // JFG PORT: the shadow map triangles: the level, and the candidates with the projection of one of its draws.
        for (const JfgShadowCandidate &c : jfgShadowCandidates) {
            bool keep = c.level;
            for (size_t i = 0; !keep && (i < jfgShadowCandidates.size()); i++) {
                const JfgShadowCandidate &l = jfgShadowCandidates[i];
                keep = l.level && ((l.transformsIndex == c.transformsIndex) ||
                    (matrixDifference(drawData.projTransforms[l.transformsIndex], drawData.projTransforms[c.transformsIndex]) < 1e-4f));
            }

            if (keep) {
                jfgShadowIndices.insert(jfgShadowIndices.end(), drawData.faceIndices.begin() + c.start, drawData.faceIndices.begin() + c.end);
            }
        }

        jfgShadowCandidates.clear();
    }

    void FramebufferRenderer::setJfgLights(const std::vector<interop::PointLight> &lights, const std::vector<interop::PointLight> &points,
        const std::vector<interop::JfgObjectLight> &objects, float curFrameWeight)
    {
        jfgObjects = objects;
        if (jfgObjects.empty()) {
            jfgObjects.emplace_back(); // the upload needs one
        }

        // JFG PORT: the game's lights (State::fullSync), at most MaxLights (the port sends the nearest first). Linear
        // falloff from the inner radius (pointRadius) to the radius (attenuationRadius). On in-between frames a light is
        // placed back along how far it moved since the game frame before (direction, see rt64_jfg_frame_lights), as
        // RSPProcessCS does with vertices.
        const size_t MaxLights = 64;
        jfgLights.clear();
        for (size_t i = 0; (i < lights.size()) && (i < MaxLights); i++) {
            const interop::PointLight &l = lights[i];
            interop::JfgLight dst;
            const float back = 1.0f - curFrameWeight;
            dst.posRadius = { l.position.x - l.direction.x * back, l.position.y - l.direction.y * back, l.position.z - l.direction.z * back,
                l.attenuationRadius };
            dst.colorInner = { l.diffuseColor.x, l.diffuseColor.y, l.diffuseColor.z, l.pointRadius };
            dst.shadow = { 0.0f, 0.0f, 0.0f, (l.groupBits & 1U) ? 1.0f : 0.0f }; // w: casts no shadows
            jfgLights.emplace_back(dst);
        }

        // The camera ([0] of the frame's points), placed on in-between frames as the lights are.
        const float back = 1.0f - curFrameWeight;
        hlslpp::float3 camera = { 0.0f, 0.0f, 0.0f };
        if (!points.empty()) {
            const interop::PointLight &c = points[0];
            camera = { float(c.position.x - c.direction.x * back), float(c.position.y - c.direction.y * back), float(c.position.z - c.direction.z * back) };
        }

        // Shadow maps for the nearest few (sent nearest first), thrown from above the light (JFG_SHADOW_LIFT,
        // JfgShadow.hlsli) out to the farthest point it lights. So that a light's shadows don't pop when another one
        // takes its map, each fades out as it gets as far as the next light in line (and the new one fades in).
        const float ShadowLift = 40.0f;
        const size_t shadowed = (rt64_jfg_shadows.load(std::memory_order_relaxed) > 0.0f) ?
            size_t(std::clamp(rt64_jfg_shadow_lights.load(std::memory_order_relaxed), 0, int(JfgShadowRows))) : 0;
        auto distanceToCamera = [&](const interop::JfgLight &light) {
            const float dx = float(light.posRadius.x) - float(camera.x);
            const float dy = float(light.posRadius.y) - float(camera.y);
            const float dz = float(light.posRadius.z) - float(camera.z);
            return sqrtf(dx * dx + dy * dy + dz * dz);
        };

        jfgShadowLights.clear();
        float shadowDistance[JfgShadowRows] = {};
        float nextDistance = -1.0f;
        for (interop::JfgLight &light : jfgLights) {
            const bool noShadows = float(light.shadow.w) > 0.5f; // set by the lighting tool
            light.shadow = { 0.0f, 0.0f, 0.0f, 0.0f };
            if ((float(light.posRadius.w) <= 1.0f) || noShadows) {
                continue;
            }

            if (jfgShadowLights.size() < shadowed) {
                const float farPlane = float(light.posRadius.w) + ShadowLift + 8.0f;
                shadowDistance[jfgShadowLights.size()] = distanceToCamera(light);
                light.shadow = { float(jfgShadowLights.size() + 1), 1.0f, farPlane, 0.0f };
                jfgShadowLights.emplace_back(float(light.posRadius.x), float(light.posRadius.y) + ShadowLift, float(light.posRadius.z), farPlane);
            }
            else if (nextDistance < 0.0f) {
                nextDistance = distanceToCamera(light);
            }
        }

        float params[20];
        {
            std::lock_guard<std::mutex> lock(jfgParamsMutex);
            memcpy(params, jfgParams, sizeof(params));
        }

        interop::JfgFrame frame = {};
        frame.camera = { float(camera.x), float(camera.y), float(camera.z), 0.0f };
        hlslpp::float3 sunDir = { params[0], params[1], params[2] };
        const float sunLength = sqrtf(params[0] * params[0] + params[1] * params[1] + params[2] * params[2]);
        sunDir = (sunLength > 1e-4f) ? (sunDir / sunLength) : hlslpp::float3(0.0f, 1.0f, 0.0f);
        frame.sunDir = { float(sunDir.x), float(sunDir.y), float(sunDir.z), (params[3] > 0.5f) ? 1.0f : 0.0f };
        frame.sunColor = { params[4], params[5], params[6], params[7] };
        frame.material = { params[8], params[9], params[10], params[11] };
        float fade[JfgShadowRows];
        for (uint32_t r = 0; r < JfgShadowRows; r++) {
            const float transition = std::max(nextDistance * 0.2f, 40.0f);
            fade[r] = (nextDistance < 0.0f) ? 1.0f : std::clamp((nextDistance - shadowDistance[r]) / transition, 0.0f, 1.0f);
        }

        frame.shadowFade = { fade[0], fade[1], fade[2], fade[3] };
        frame.extra = { params[12], jfgSunRange, params[15], params[16] };

        // The directional light's shadow map: a square around where the camera looks ([1] of the frame's points),
        // moved only in whole texels across the light's view so its edges don't crawl as the camera moves.
        jfgSunShadow = { 0.0f, 0.0f, 0.0f, 0.0f };
        if ((frame.sunDir.w > 0.5f) && (params[13] > 0.5f) && (shadowed > 0 || rt64_jfg_shadows.load(std::memory_order_relaxed) > 0.0f) && (points.size() >= 2)) {
            const interop::PointLight &f = points[1];
            const hlslpp::float3 focus = { float(f.position.x - f.direction.x * back), float(f.position.y - f.direction.y * back), float(f.position.z - f.direction.z * back) };
            const float half = std::clamp(params[14], 250.0f, 8000.0f);
            const hlslpp::float3 forward = -sunDir;
            const hlslpp::float3 worldUp = (fabsf(float(forward.y)) > 0.99f) ? hlslpp::float3(0.0f, 0.0f, 1.0f) : hlslpp::float3(0.0f, 1.0f, 0.0f);
            const hlslpp::float3 right = hlslpp::normalize(hlslpp::cross(worldUp, forward));
            const hlslpp::float3 up = hlslpp::cross(forward, right);
            const float texel = std::max(half / float(std::max(jfgShadowTile, 1U)), 0.001f); // 2 x half over 2 tiles
            const float x = floorf(float(hlslpp::dot(focus, right)) / texel) * texel;
            const float y = floorf(float(hlslpp::dot(focus, up)) / texel) * texel;
            const float z = float(hlslpp::dot(focus, forward));
            const hlslpp::float3 centre = right * x + up * y + forward * z;
            jfgSunShadow = { float(centre.x), float(centre.y), float(centre.z), half };
        }

        frame.sunShadow = jfgSunShadow;
        frame.characters = { params[17], params[18], params[19], 0.0f };
        jfgFrameData.assign(1, frame);

        jfgLightCount = uint32_t(jfgLights.size());
        if (jfgLights.empty()) {
            jfgLights.emplace_back(); // the upload needs one
        }
    }

    void FramebufferRenderer::endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled) {
        if (jfgFrameData.empty()) {
            jfgFrameData.emplace_back(interop::JfgFrame{}); // JFG PORT: the upload needs one
        }

        uploadJfgMaterials(worker); // JFG PORT

        bool shaderViewRtEnabled = false;
        std::vector<BufferUploader::Upload> shaderUploads = {
            { renderIndicesVector.data(), { 0, renderIndicesVector.size() }, sizeof(interop::RenderIndices), RenderBufferFlag::STORAGE, { }, &renderIndicesBuffer},
            { &frameParams, { 0, 1 }, sizeof(interop::FrameParams), RenderBufferFlag::CONSTANT, { }, &frameParamsBuffer},
            { jfgLights.data(), { 0, jfgLights.size() }, sizeof(interop::JfgLight), RenderBufferFlag::STORAGE, { }, &jfgLightsBuffer }, // JFG PORT
            { jfgFrameData.data(), { 0, jfgFrameData.size() }, sizeof(interop::JfgFrame), RenderBufferFlag::STORAGE, { }, &jfgFrameBuffer },
            { jfgObjects.data(), { 0, jfgObjects.size() }, sizeof(interop::JfgObjectLight), RenderBufferFlag::STORAGE, { }, &jfgObjectsBuffer }
        };

        // JFG PORT: the shadow maps' triangles (never empty for the upload) and their atlas.
        if (jfgShadowIndices.empty()) {
            jfgShadowLights.clear();
            jfgShadowIndices.assign(3, 0);
        }

        shaderUploads.push_back({ jfgShadowIndices.data(), { 0, jfgShadowIndices.size() }, sizeof(uint32_t), RenderBufferFlag::INDEX, { }, &jfgShadowIndicesBuffer });
        updateJfgShadowAtlas(worker);

#   if RT_ENABLED
        // FIXME: Add support for multiple raytracing scenes.
        Framebuffer *chosenFramebuffer = nullptr;
        RaytracingScene *chosenRtScene = nullptr;
        if (rtEnabled) {
            rtResources->updateBottomLevelASResources(worker);

            for (uint32_t i = 0; i < framebufferCount; i++) {
                RenderTargetDrawCall &targetDrawCall = framebufferVector[i].renderTargetDrawCall;
                if (!targetDrawCall.rtScenes.empty()) {
                    chosenFramebuffer = &framebufferVector[i];
                    chosenRtScene = &targetDrawCall.rtScenes[0];
                }
            }
        }

        if (chosenRtScene != nullptr) {
            if (rtResources->updateOutputBuffers) {
                rtResources->createOutputBuffers(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight);
                rtResources->updateOutputBuffers = false;
            }

            const RenderTarget *framebufferTarget = chosenFramebuffer->renderTargetDrawCall.fbStorage->colorTarget;
            interleavedRastersCount = static_cast<uint32_t>(chosenRtScene->interleavedRasters.size());
            rtResources->updateInterleavedRenderTargets(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight, interleavedRastersCount, framebufferTarget->multisampling, framebufferTarget->usesHDR);

            for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                auto &intRaster = chosenRtScene->interleavedRasters[i];
                RenderTarget *colorTarget = rtResources->interleavedColorTargetVector[i].get();
                RenderTarget *depthTarget = rtResources->interleavedDepthTargetVector[i].get();
                intRaster.colorTextureIndex = getTextureIndex(colorTarget);
                intRaster.depthTextureIndex = getTextureIndex(depthTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(colorTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(depthTarget);
            }

            // Must have at least one element in the vector.
            if (chosenRtScene->interleavedRasters.empty()) {
                chosenRtScene->interleavedRasters.emplace_back();
            }

            shaderUploads.push_back({ &rtResources->rtParams, { 0, 1 }, sizeof(interop::RaytracingParams), RenderBufferFlag::CONSTANT, { }, &rtResources->rtParamsBuffer });
            shaderUploads.push_back({ chosenRtScene->interleavedRasters.data(), { 0, chosenRtScene->interleavedRasters.size() }, sizeof(interop::InterleavedRaster), RenderBufferFlag::STORAGE, { }, &interleavedRastersBuffer });

            updateRaytracingScene(worker, *chosenRtScene);
            shaderViewRtEnabled = true;
        }
#   endif

        shaderUploader->submit(worker, shaderUploads);
        jfgShadowIndexView = RenderIndexBufferView(RenderBufferReference(jfgShadowIndicesBuffer.get()), uint32_t(sizeof(uint32_t) * jfgShadowIndices.size()), RenderFormat::R32_UINT); // JFG PORT
        updateShaderViews(worker, drawBuffers, outputBuffers, shaderViewRtEnabled);
    }

    // JFG PORT: the material maps of a texture (by its hash), as GPUTile::jfgMaterial: <16 hex digits of the hash>_n.png
    // (normal map, tangent space with green up), _s.png (specular: red = how strong the shine, green = how sharp,
    // blue = how much the texture glows in its own colour) and _e.png (emission: the light the surface gives off, added
    // on top, black = none),
    // in the materials folder next to the game or anywhere in a texture pack (scanJfgMaterials). A texture's maps are
    // loaded the first time it is drawn (a pack's file is read here, under the texture cache's lock, then decoded on a
    // thread of its own); until they are uploaded the tile has none.
    static bool jfgMaterialName(const std::string &path, uint64_t &hash, int &kind) {
        const size_t slash = path.find_last_of("/\\");
        const std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
        if ((name.size() != 22) || (name[16] != '_')) {
            return false;
        }

        std::string ext = name.substr(18);
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(tolower(c)); });
        const char k = char(tolower((unsigned char)(name[17])));
        if ((ext != ".png") || ((k != 'n') && (k != 's') && (k != 'e'))) {
            return false;
        }

        char *parseEnd = nullptr;
        const std::string hex = name.substr(0, 16);
        hash = strtoull(hex.c_str(), &parseEnd, 16);
        if ((parseEnd == nullptr) || (*parseEnd != '\0')) {
            return false;
        }

        kind = (k == 'n') ? 0 : ((k == 's') ? 1 : 2);
        return true;
    }

    void FramebufferRenderer::scanJfgMaterials(const TextureCache *textureCache) {
        // The texture cache's map lock is held (updateTextureCache).
        const auto &packs = textureCache->textureMap.replacementMap.fileSystems;
        std::vector<const FileSystem *> current;
        for (const auto &fs : packs) {
            current.push_back(fs.get());
        }

        std::lock_guard<std::mutex> lock(jfgMaterialMutex);
        if (jfgMaterialsScanned && (current == jfgMaterialPacks)) {
            return;
        }

        jfgMaterialsScanned = true;
        jfgMaterialPacks = current;
        jfgMaterialFiles.clear();
        jfgMaterialSlots.clear(); // loaded maps keep their slots (the descriptor set may still use them); new ones load again
        jfgMaterialRequested.clear();
        uint64_t hash = 0;
        int kind = 0;
        std::error_code ec;
        const char *dir = std::getenv("RT64_JFG_MATERIALS");
        const std::filesystem::path folder = (dir != nullptr) ? std::filesystem::path(dir) : std::filesystem::path("materials");
        for (std::filesystem::directory_iterator it(folder, ec), end; !ec && (it != end); it.increment(ec)) {
            if (jfgMaterialName(it->path().filename().string(), hash, kind)) {
                jfgMaterialFiles[hash].path[kind] = it->path().string();
                jfgMaterialFiles[hash].fileSystem[kind] = -1;
            }
        }

        for (size_t i = 0; i < packs.size(); i++) {
            for (FileSystem::Iterator it = packs[i]->begin(); it != packs[i]->end(); ++it) {
                const std::string &path = *it;
                if (jfgMaterialName(path, hash, kind) && jfgMaterialFiles[hash].path[kind].empty()) {
                    jfgMaterialFiles[hash].path[kind] = path;
                    jfgMaterialFiles[hash].fileSystem[kind] = int(i);
                }
            }
        }

        rt64_jfg_material_files = int(jfgMaterialFiles.size());
        if (!jfgMaterialFiles.empty()) {
            fprintf(stderr, "[RT64 JFG] %zu textures have material maps (%s and %zu texture packs)\n", jfgMaterialFiles.size(), folder.string().c_str(), packs.size());
        }
    }

    uint32_t FramebufferRenderer::jfgMaterialFor(uint64_t hash, TextureCache *textureCache) {
        JfgMaterialFile file;
        {
            std::lock_guard<std::mutex> lock(jfgMaterialMutex);
            if (jfgMaterialFiles.empty()) {
                return 0;
            }

            const auto slots = jfgMaterialSlots.find(hash);
            if (slots != jfgMaterialSlots.end()) {
                return slots->second;
            }

            static std::unordered_set<uint64_t> logged;
            if ((getenv("RT64_JFG_MATERIAL_LOG") != nullptr) && logged.insert(hash).second) {
                fprintf(stderr, "[RT64 JFG] drawn %016llX\n", (unsigned long long)hash);
            }

            const auto files = jfgMaterialFiles.find(hash);
            if ((files == jfgMaterialFiles.end()) || !jfgMaterialRequested.insert(hash).second) {
                return 0;
            }

            file = files->second;
        }

        for (int kind = 0; kind < 3; kind++) {
            if (file.path[kind].empty()) {
                continue;
            }

            // A pack's file is read now, while the pack can't go away; the folder's on the decoding thread.
            std::vector<uint8_t> bytes;
            if ((file.fileSystem[kind] >= 0) && (textureCache != nullptr)) {
                const std::unique_lock<std::mutex> mapLock(textureCache->textureMapMutex);
                const auto &packs = textureCache->textureMap.replacementMap.fileSystems;
                if (size_t(file.fileSystem[kind]) < packs.size()) {
                    packs[file.fileSystem[kind]]->load(file.path[kind], bytes);
                }

                if (bytes.empty()) {
                    continue;
                }
            }

            const std::string path = file.path[kind];
            std::thread([hash, kind, path, bytes = std::move(bytes)]() mutable {
                if (bytes.empty()) {
                    std::ifstream stream(std::filesystem::path(path), std::ios::binary);
                    bytes.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
                }

                int width = 0, height = 0;
                stbi_uc *data = bytes.empty() ? nullptr : stbi_load_from_memory(bytes.data(), int(bytes.size()), &width, &height, nullptr, 4);
                if (data == nullptr) {
                    fprintf(stderr, "[RT64 JFG] material map %s could not be read\n", path.c_str());
                    return;
                }

                JfgMaterialImage image;
                image.hash = hash;
                image.kind = kind;
                image.width = uint32_t(width);
                image.height = uint32_t(height);
                image.rgba.assign(data, data + size_t(width) * size_t(height) * 4);
                stbi_image_free(data);
                std::lock_guard<std::mutex> imageLock(jfgMaterialImageMutex);
                jfgMaterialImages.emplace_back(std::move(image));
            }).detach();
        }

        return 0;
    }

    // JFG PORT: the material maps decoded since the last frame go into textures of their own (recorded on this frame's
    // command list, before anything is drawn) and get their slots; the descriptor set takes them in
    // updateShaderDescriptorSet.
    void FramebufferRenderer::uploadJfgMaterials(RenderWorker *worker) {
        std::vector<JfgMaterialImage> images;
        {
            std::lock_guard<std::mutex> imageLock(jfgMaterialImageMutex);
            images.swap(jfgMaterialImages);
        }

        if (images.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(jfgMaterialMutex);
        for (const JfgMaterialImage &image : images) {
            const uint32_t slot = uint32_t(jfgMaterialTextures.size());
            if (slot >= JFG_MATERIAL_SLOTS) {
                fprintf(stderr, "[RT64 JFG] more than %d material maps in use: the rest are left out\n", JFG_MATERIAL_SLOTS);
                break;
            }

            std::unique_ptr<Texture> texture = std::make_unique<Texture>();
            std::unique_ptr<RenderBuffer> upload;
            TextureCache::setRGBA32(texture.get(), worker->device, worker->commandList.get(), image.rgba.data(), image.rgba.size(), image.width, image.height, image.width * 4, upload);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(texture->texture.get(), RenderTextureLayout::SHADER_READ));
            jfgMaterialTextures.emplace_back(std::move(texture));
            jfgMaterialUploads.emplace_back(std::move(upload));
            jfgMaterialPending.push_back(slot);
            uint32_t &packed = jfgMaterialSlots[image.hash];
            packed |= (slot + 1) << (10 * image.kind);
            if (getenv("RT64_JFG_MATERIAL_LOG") != nullptr) {
                fprintf(stderr, "[RT64 JFG] material map %016llX kind %d in slot %u\n", (unsigned long long)image.hash, image.kind, slot);
            }
        }

        rt64_jfg_material_loaded = int(jfgMaterialTextures.size());
    }

    // JFG PORT: the shadow atlas at the size set (six faces a row, JfgShadowRows rows). A replaced one is kept until
    // the next workload, as the command list being recorded may still use it.
    void FramebufferRenderer::updateJfgShadowAtlas(RenderWorker *worker) {
        const uint32_t tile = uint32_t(std::clamp(rt64_jfg_shadow_size.load(std::memory_order_relaxed), 128, 1024));
        if ((jfgShadowAtlas != nullptr) && (jfgShadowTile == tile)) {
            return;
        }

        if (jfgShadowAtlas != nullptr) {
            jfgShadowRetired.emplace_back(std::move(jfgShadowAtlas));
            jfgShadowRetiredViews.emplace_back(std::move(jfgShadowAtlasView));
            jfgShadowRetiredFramebuffers.emplace_back(std::move(jfgShadowFramebuffer));
        }

        RenderClearValue clearValue = RenderClearValue::Depth(RenderDepth(), RenderFormat::D32_FLOAT);
        // Rows of cubes, then the directional light's map (two tiles square) below them.
        jfgShadowAtlas = worker->device->createTexture(RenderTextureDesc::DepthTarget(tile * 6, tile * (JfgShadowRows + 2), RenderFormat::D32_FLOAT, RenderMultisampling(), &clearValue));
        jfgShadowAtlas->setName("JFG Shadow Maps");
        jfgShadowAtlasView = jfgShadowAtlas->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::D32_FLOAT));
        jfgShadowFramebuffer = worker->device->createFramebuffer(RenderFramebufferDesc(nullptr, 0, jfgShadowAtlas.get()));
        jfgShadowTile = tile;
        jfgShadowAtlasReadable = false;
    }

    // JFG PORT: the faces of each shadowed light's cube, after the RSP has written the frame's world positions: the
    // frame's shadow triangles drawn once per face into its tile (JfgShadowVS).
    void FramebufferRenderer::recordJfgShadows(RenderWorker *worker) {
        if (jfgShadowAtlas == nullptr) {
            return;
        }

        const bool sun = float(jfgSunShadow.w) > 0.0f;
        const bool draw = (jfgShadowPipeline != nullptr) && (rendererPipelineLayout != nullptr) && (!jfgShadowLights.empty() || sun);
        if (!draw) {
            if (!jfgShadowAtlasReadable) {
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(jfgShadowAtlas.get(), RenderTextureLayout::DEPTH_READ));
                jfgShadowAtlasReadable = true;
            }

            return;
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(jfgShadowAtlas.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(jfgShadowFramebuffer.get());
        const RenderRect fullRect(0, 0, int32_t(jfgShadowTile * 6), int32_t(jfgShadowTile * (JfgShadowRows + 2)));
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(fullRect.right), float(fullRect.bottom)));
        worker->commandList->setScissors(fullRect);
        worker->commandList->clearDepth();
        worker->commandList->setPipeline(jfgShadowPipeline);
        worker->commandList->setGraphicsPipelineLayout(rendererPipelineLayout);
        worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->setIndexBuffer(&jfgShadowIndexView);
        const uint32_t indexCount = uint32_t(jfgShadowIndices.size());
        for (uint32_t row = 0; row < uint32_t(jfgShadowLights.size()); row++) {
            const hlslpp::float4 &light = jfgShadowLights[row];
            interop::RasterParams params = {};
            const float pos[3] = { float(light.x), float(light.y), float(light.z) };
            memcpy(&params.padding, pos, sizeof(pos));
            params.screenScale.x = 1.0f;
            params.screenScale.y = float(light.w);
            for (uint32_t face = 0; face < 6; face++) {
                const int32_t x = int32_t(face * jfgShadowTile);
                const int32_t y = int32_t(row * jfgShadowTile);
                worker->commandList->setViewports(RenderViewport(float(x), float(y), float(jfgShadowTile), float(jfgShadowTile)));
                worker->commandList->setScissors(RenderRect(x, y, x + int32_t(jfgShadowTile), y + int32_t(jfgShadowTile)));
                params.renderIndex = face;
                worker->commandList->setGraphicsPushConstants(0, &params);
                worker->commandList->drawIndexedInstanced(indexCount, 1, 0, 0, 0);
            }
        }

        if (sun) {
            interop::RasterParams params = {};
            const float centre[3] = { float(jfgSunShadow.x), float(jfgSunShadow.y), float(jfgSunShadow.z) };
            memcpy(&params.padding, centre, sizeof(centre));
            params.renderIndex = 6; // JFG_SUN_FACE
            params.screenScale.x = float(jfgSunShadow.w);
            params.screenScale.y = jfgSunRange;
            const int32_t y = int32_t(JfgShadowRows * jfgShadowTile), size = int32_t(jfgShadowTile * 2);
            worker->commandList->setViewports(RenderViewport(0.0f, float(y), float(size), float(size)));
            worker->commandList->setScissors(RenderRect(0, y, size, y + size));
            worker->commandList->setGraphicsPushConstants(0, &params);
            worker->commandList->drawIndexedInstanced(indexCount, 1, 0, 0, 0);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(jfgShadowAtlas.get(), RenderTextureLayout::DEPTH_READ));
        jfgShadowAtlasReadable = true;
    }

    void FramebufferRenderer::advanceFrame(bool rtEnabled) {
        frameParams.frameCount++;

#   if RT_ENABLED
        if (rtEnabled) {
            rtResources->swapBuffers = !rtResources->swapBuffers;
            rtResources->skipReprojection = false;
        }
#   endif
    }
};

/*
void RT64::View::renderIm3d() {
    if (Im3d::GetDrawListCount() > 0) {
        commandList->SetGraphicsRootSignature(worker->device->getIm3dRootSignature());

        commandList->SetDescriptorHeaps(1, &descriptorHeap);
        commandList->SetGraphicsRootDescriptorTable(0, descriptorHeap->GetGPUDescriptorHandleForHeapStart());
        commandList->RSSetViewports(1, &rtViewport);
        commandList->RSSetScissorRects(1, &rtScissor);

        unsigned int totalVertexCount = 0;
        for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
            auto &drawList = Im3d::GetDrawLists()[i];
            totalVertexCount += drawList.m_vertexCount;
        }

        if (totalVertexCount > 0) {
            // Release the previous vertex buffer if it should be bigger.
            if (!im3dVertexBuffer.IsNull() && (totalVertexCount > im3dVertexCount)) {
                im3dVertexBuffer.Release();
            }

            // Create the vertex buffer if it's empty.
            const UINT vertexBufferSize = totalVertexCount * sizeof(Im3d::VertexData);
            if (im3dVertexBuffer.IsNull()) {
                CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
                im3dVertexBuffer = worker->device->allocateResource(D3D12_HEAP_TYPE_UPLOAD, &uploadBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr);
                im3dVertexCount = totalVertexCount;
                im3dVertexBufferView.BufferLocation = im3dVertexBuffer.Get()->GetGPUVirtualAddress();
                im3dVertexBufferView.StrideInBytes = sizeof(Im3d::VertexData);
                im3dVertexBufferView.SizeInBytes = vertexBufferSize;
            }

            // Copy data to vertex buffer.
            UINT8 *pDataBegin;
            CD3DX12_RANGE readRange(0, 0);
            D3D12_CHECK(im3dVertexBuffer.Get()->Map(0, &readRange, reinterpret_cast<void **>(&pDataBegin)));
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                size_t copySize = sizeof(Im3d::VertexData) * drawList.m_vertexCount;
                memcpy(pDataBegin, drawList.m_vertexData, copySize);
                pDataBegin += copySize;
            }

            im3dVertexBuffer.Get()->Unmap(0, nullptr);

            unsigned int vertexOffset = 0;
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                commandList->IASetVertexBuffers(0, 1, &im3dVertexBufferView);
                switch (drawList.m_primType) {
                case Im3d::DrawPrimitive_Points:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStatePoint());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                    break;
                case Im3d::DrawPrimitive_Lines:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateLine());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
                    break;
                case Im3d::DrawPrimitive_Triangles:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateTriangle());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    break;
                default:
                    break;
                }

                commandList->DrawInstanced(drawList.m_vertexCount, 1, vertexOffset, 0);
                vertexOffset += drawList.m_vertexCount;
            }
        }
    }
}
*/

/*
namespace RT64 {
    class View {
    private:
        // Im3D
        AllocatedResource im3dVertexBuffer;
        D3D12_VERTEX_BUFFER_VIEW im3dVertexBufferView;
        unsigned int im3dVertexCount;
    };
};
*/