//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

// JFG PORT: authored material maps (normal, specular, emission) live at the top of the texture array, from this index
// on (FramebufferRenderer::jfgMaterialFor); GPUTile::jfgMaterial = normal slot + 1 | (specular slot + 1) << 10 |
// (emission slot + 1) << 20 (0 = none).
#define JFG_MATERIAL_BASE 7680
#define JFG_MATERIAL_SLOTS 512

#ifdef HLSL_CPU
namespace interop {
#endif
#ifdef HLSL_CPU
    union GPUTileFlags {
        struct {
            uint alphaIsCvg : 1;
            uint highRes : 1;
            uint fromCopy : 1;
            uint rawTMEM : 1;
            uint hasMipmaps : 1;
            uint shiftedByHalf : 1;
            uint jfgSoft : 1; // JFG PORT: a copy of a small framebuffer, sampled with a soft edge (TextureSampler.hlsli)
            uint jfgGlow : 8; // JFG PORT: how much the texture glows (0-255, rt64_jfg_glow_for)
        };

        uint value;
    };
#else
    // SPIR-V code generation does not seem to like bitfields at the moment, so we work around it by querying the flags manually.
    typedef uint GPUTileFlags;

    bool gpuTileFlagAlphaIsCvg(GPUTileFlags flags) {
        return flags & 0x1;
    }

    bool gpuTileFlagHighRes(GPUTileFlags flags) {
        return flags & 0x2;
    }

    bool gpuTileFlagFromCopy(GPUTileFlags flags) {
        return flags & 0x4;
    }

    bool gpuTileFlagRawTMEM(GPUTileFlags flags) {
        return flags & 0x8;
    }

    bool gpuTileFlagHasMipmaps(GPUTileFlags flags) {
        return flags & 0x10;
    }

    bool gpuTileFlagShiftedByHalf(GPUTileFlags flags) {
        return flags & 0x20;
    }

    bool gpuTileFlagJfgSoft(GPUTileFlags flags) {
        return flags & 0x40;
    }

    float gpuTileFlagJfgGlow(GPUTileFlags flags) {
        return float((flags >> 7) & 0xFF) / 255.0f;
    }
#endif
    struct GPUTile {
        float2 ulScale;
        float2 tcScale;
        uint2 texelShift;
        uint2 texelMask;
        uint textureIndex;
        float3 textureDimensions;
        GPUTileFlags flags;
        uint jfgMaterial; // JFG PORT: see JFG_MATERIAL_BASE (0 = none)
    };
#ifdef HLSL_CPU
};
#endif