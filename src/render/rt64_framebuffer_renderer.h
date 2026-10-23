//
// RT64
//

#pragma once

#include <stdint.h>

#include "common/rt64_emulator_configuration.h"
#include "common/rt64_user_configuration.h"
#include "hle/rt64_framebuffer_manager.h"
#include "hle/rt64_workload.h"
#include "preset/rt64_preset_scene.h"
#include "shared/rt64_frame_params.h"
#include "shared/rt64_gpu_tile.h"
#include "shared/rt64_jfg_light.h"
#include "shared/rt64_render_indices.h"
#include "shared/rt64_render_params.h"

#include "rt64_buffer_uploader.h"
#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_raster_shader_cache.h"
#include "rt64_render_target.h"
#include "rt64_rsp_processor.h"
#include "rt64_vertex_processor.h"

#if RT_ENABLED
#   include "rt64_raytracing_resources.h"
#   include "rt64_raytracing_shader_cache.h"
#endif

namespace RT64 {
    struct DynamicTextureView {
        const RenderTexture *texture = nullptr;
        uint32_t dstIndex = 0;
        const RenderTextureView *textureView = nullptr;
    };

    struct RasterScene {
        std::vector<uint32_t> instanceIndices;

        RasterScene();
    };

    struct RenderTargetDrawCall {
        typedef std::pair<uint32_t, bool> SceneIndexPair;

        RenderFramebufferStorage *fbStorage = nullptr;
        std::vector<RasterScene> rasterScenes;
        std::vector<SceneIndexPair> sceneIndices;
#   if RT_ENABLED
        std::vector<RaytracingScene> rtScenes;
#   endif
    };

    struct RSPSmoothNormalGenerationCB {
        uint32_t indexStart;
        uint32_t indexCount;
    };

    struct FramebufferRenderer {
        std::vector<uint32_t> textureCacheVersions;
        std::vector<Texture *> textureCacheTextures;
        std::vector<Texture *> textureCacheTextureReplacements;
        std::vector<uint32_t> textureCacheFreeSpaces;
        uint32_t textureCacheSize = 0;
        uint32_t textureCacheGlobalVersion = 0;
        bool textureCacheReplacementMapEnabled = false;
        std::vector<InstanceDrawCall> instanceDrawCallVector;
        std::vector<RenderPipelineProgram> hitGroupVector;
        std::vector<interop::RenderIndices> renderIndicesVector;
        std::vector<DynamicTextureView> dynamicTextureViewVector;
        std::vector<RenderTextureBarrier> dynamicTextureBarrierVector;
        std::unique_ptr<BufferUploader> shaderUploader;
        std::vector<RSPSmoothNormalGenerationCB> rspSmoothNormalVector;
        std::array<RenderInputSlot, 3> vertexInputSlots;
        std::array<RenderVertexBufferView, 3> indexedVertexViews;
        std::array<RenderVertexBufferView, 3> rawVertexViews;
        RenderIndexBufferView indexBufferView;
        RenderBuffer *testZIndexBuffer = nullptr;
        RenderIndexBufferView testZIndexBufferView;
        BufferPair renderIndicesBuffer;
        BufferPair interleavedRastersBuffer;
        uint32_t interleavedRastersCount = 0;
        // JFG PORT: the frame's lights for the per-pixel level lighting (setJfgLights, RasterPS); never empty for the
        // upload, jfgLightCount is how many are real.
        std::vector<interop::JfgLight> jfgLights;
        uint32_t jfgLightCount = 0;
        BufferPair jfgLightsBuffer;
        std::vector<hlslpp::float4> jfgCasters; // xyz = sphere centre, w = radius; never empty, jfgCasterCount are real
        std::vector<interop::JfgObjectLight> jfgObjects; // by slot; never empty
        BufferPair jfgObjectsBuffer;
        uint32_t jfgCasterCount = 0;
        BufferPair jfgCastersBuffer;
        BufferPair frameParamsBuffer;
        RenderPipelineLayout *rendererPipelineLayout = nullptr;
        RenderPipeline *postBlendDitherNoiseAddPipeline = nullptr;
        RenderPipeline *postBlendDitherNoiseSubPipeline = nullptr;
        RenderPipeline *postBlendDitherNoiseSubNegativePipeline = nullptr;
        RenderPipeline *ssaoPipeline = nullptr; // JFG PORT
        RenderPipeline *ssaoBlurPipeline = nullptr; // JFG PORT: bloom layout
        RenderPipeline *fogPipeline = nullptr; // JFG PORT
        RenderPipeline *vignettePipeline = nullptr; // JFG PORT
        // JFG PORT: shadow maps for the nearest lights (recordJfgShadows, JfgShadow.hlsli). The level's and the
        // characters' triangles of the frame (from the draws' ranges of the shared index list), the lights given a
        // cube (lifted position, far; one atlas row each) and the atlas: six tiles a row, JfgShadowRows rows.
        static const uint32_t JfgShadowRows = 4;
        RenderPipeline *jfgShadowPipeline = nullptr;
        std::vector<uint32_t> jfgShadowIndices;
        struct JfgShadowCandidate {
            uint32_t start, end, transformsIndex;
            bool level;
        };
        std::vector<JfgShadowCandidate> jfgShadowCandidates; // one framebuffer's (addFramebuffer)
        BufferPair jfgShadowIndicesBuffer;
        RenderIndexBufferView jfgShadowIndexView;
        std::vector<hlslpp::float4> jfgShadowLights; // xyz = where the shadows are thrown from, w = far
        uint32_t jfgShadowTile = 0;
        bool jfgShadowAtlasReadable = false;
        std::unique_ptr<RenderTexture> jfgShadowAtlas;
        std::unique_ptr<RenderTextureView> jfgShadowAtlasView;
        std::unique_ptr<RenderFramebuffer> jfgShadowFramebuffer;
        std::vector<std::unique_ptr<RenderTexture>> jfgShadowRetired;
        std::vector<std::unique_ptr<RenderTextureView>> jfgShadowRetiredViews;
        std::vector<std::unique_ptr<RenderFramebuffer>> jfgShadowRetiredFramebuffers;
        // JFG PORT: bloom. Two small buffers per bloom-buffer size (kept: sizes rarely change, and a buffer may still
        // be referenced by the command list being recorded), and the descriptor sets made while recording one workload.
        struct BloomBuffers {
            uint32_t width = 0;
            uint32_t height = 0;
            std::unique_ptr<RenderTexture> texture[2];
            std::unique_ptr<RenderTextureView> textureView[2];
            std::unique_ptr<RenderFramebuffer> framebuffer[2];
        };
        std::vector<std::unique_ptr<BloomBuffers>> bloomBuffers;
        std::vector<std::unique_ptr<BloomBuffers>> bloomBuffersRetired;
        std::vector<std::unique_ptr<BloomBuffers>> ssaoBuffers; // JFG PORT: ambient occlusion buffer (texture[0] only), full size
        std::vector<std::unique_ptr<RenderTargetCopyDescriptorSet>> bloomDescriptorSets;
        // JFG PORT: the scene as it was just before the glow was added, kept so that the next frame can start from it. Without
        // this, the parts of the screen the game does not draw again (the sides in widescreen, what the camera sees beyond the
        // level) keep last frame's glow and get it added again, every frame, until they are white. The game draws into
        // two (or more) colour buffers in turn, so each one has its own copy.
        struct BloomSnapshot {
            std::unique_ptr<RenderTexture> texture;
            const RenderTarget *target = nullptr;
            const RenderTexture *targetTexture = nullptr; // a target that was recreated is a different picture
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t samples = 0;
            RenderFormat format = RenderFormat::UNKNOWN;
            uint32_t frame = 0;
            bool valid = false;
        };
        std::vector<std::unique_ptr<BloomSnapshot>> bloomSnapshots;
        std::vector<std::unique_ptr<RenderTexture>> bloomSnapshotRetired;
        uint32_t jfgFrame = 0;
        const RenderTarget *jfgSidesClearedTarget = nullptr; // the colour buffer whose widescreen sides were cleared in jfgSidesClearedFrame
        uint32_t jfgSidesClearedFrame = 0;
        std::vector<std::pair<const RenderTarget *, uint32_t>> jfgTargetsTouched; // colour buffer and the frame it was last drawn into
        RenderPipelineLayout *bloomPipelineLayout = nullptr;
        RenderPipeline *bloomExtractPipeline = nullptr;
        RenderPipeline *bloomBlurPipeline = nullptr;
        RenderPipeline *bloomCompositePipeline = nullptr;
        BloomBuffers *getBloomBuffers(RenderWorker *worker, uint32_t width, uint32_t height, bool ssao = false);
        void updateJfgShadowAtlas(RenderWorker *worker); // JFG PORT
        void recordJfgShadows(RenderWorker *worker); // JFG PORT
        RenderDescriptorSet *makeBloomDescriptorSet(RenderWorker *worker, RenderTexture *texture, RenderTextureView *textureView);
        std::unique_ptr<FramebufferRendererDescriptorCommonSet> descCommonSet;
        std::unique_ptr<FramebufferRendererDescriptorTextureSet> descTextureSet;
        std::unique_ptr<RenderTexture> dummyColorTarget;
        std::unique_ptr<RenderTexture> dummyDepthTarget;
        std::unique_ptr<RenderTextureView> dummyColorTargetView;
        std::unique_ptr<RenderTextureView> dummyDepthTargetView;
        bool dummyColorTargetTransitioned = false;
        bool dummyDepthTargetTransitioned = false;
        std::vector<uint32_t> descriptorTextureVersions;
        uint32_t descriptorTextureGlobalVersion = 0;
        bool descriptorTextureReplacementMapEnabled = false;
        std::unique_ptr<RSPSmoothNormalDescriptorSet> smoothDescSet;
        std::unique_ptr<RSPVertexTestZDescriptorSet> vertexTestZSet;
        interop::FrameParams frameParams;
        const ShaderLibrary *shaderLibrary = nullptr;

#   if RT_ENABLED
        const RenderTexture *blueNoiseTexture = nullptr;
        const RaytracingState *rtState = nullptr;
        const RenderPipelineLayout *rtPipelineLayout = nullptr;
        std::unique_ptr<RaytracingResources> rtResources;
        bool rtSupport = false;
#   endif

        struct Framebuffer {
            std::unique_ptr<RenderBuffer> paramsBuffer;
            std::unique_ptr<FramebufferRendererDescriptorFramebufferSet> descRealFbSet;
            std::unique_ptr<FramebufferRendererDescriptorFramebufferSet> descDummyFbSet;
            std::set<RenderTarget *> transitionRenderTargetSet;
            RenderTargetDrawCall renderTargetDrawCall;
            RenderViewport viewport;
        };

        std::vector<Framebuffer> framebufferVector;
        std::vector<BufferUploader *> pendingUploaders;
        uint32_t framebufferCount = 0;

        struct DrawParams {
            RenderWorker *worker;
            RenderFramebufferStorage *fbStorage;
            const Workload *curWorkload;
            uint32_t fbPairIndex;
            uint32_t fbWidth;
            uint32_t fbHeight;
            uint32_t targetWidth;
            uint32_t targetHeight;
            RasterShaderCache *rasterShaderCache;
            hlslpp::float2 resolutionScale;
            float aspectRatioSource;
            float aspectRatioTarget;
            float extAspectPercentage;
            float horizontalMisalignment;
            PresetScene presetScene;
            bool rtEnabled;
            uint64_t submissionFrame;
            float deltaTimeMs;
            bool ubershadersOnly;
            bool postBlendNoise;
            bool postBlendNoiseNegative;
            uint32_t maxGameCall;
        };

        FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary);
        ~FramebufferRenderer();
        void resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling);
        void updateTextureCache(TextureCache *textureCache);
        void createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, TextureCache *textureCache, uint64_t submissionFrame);
        uint32_t getDestinationIndex();
        uint32_t getTextureIndex(RenderTarget *renderTarget);
        uint32_t getTextureIndex(const FramebufferManager::TileCopy &tileCopy);
        void updateMultisampling();
        void updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool raytracingEnabled);
        void updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers);
        void updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers);
        void updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool raytracingEnabled);
        void submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers);
        bool submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState);
        void submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState);
        void addFramebuffer(const DrawParams &p);
        void endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled);
        void setJfgLights(const std::vector<interop::PointLight> &lights, const std::vector<interop::PointLight> &casters,
            const std::vector<interop::JfgObjectLight> &objects, float curFrameWeight); // JFG PORT: before endFramebuffers
        void recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor, VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled);
        void recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex);
        void waitForUploaders();
        void advanceFrame(bool rtEnabled);

#   if RT_ENABLED
        void resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture);
        void updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene);
        void submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene);
        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
#   endif
    };
};