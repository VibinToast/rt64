//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

// JFG PORT: for the game's raster draws, padding.x holds RASTER_JFG_* bits (the full-screen passes, SsaoPS, FogPS,
// VignettePS, use the padding for their own parameters). INDEXED: an indexed triangle draw, whose SV_VertexID indexes
// the RSP's per-vertex output buffers (the world position in RasterVS). DEBUG_WORLD: show the world position as colour
// (RT64_JFG_DEBUG_WORLD).
#define RASTER_JFG_INDEXED      0x1
#define RASTER_JFG_DEBUG_WORLD  0x2

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RasterParams {
        uint renderIndex;
        uint3 padding;
        float2 screenScale;
        float2 screenOffset;
    };
#ifdef HLSL_CPU
};
#endif
