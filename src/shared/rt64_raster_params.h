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
#define RASTER_JFG_LEVEL        0x4 // the level's own geometry: light it per pixel with padding.y of the frame's lights
#define RASTER_JFG_DEBUG_NORMAL 0x8 // show the surface direction as colour
#define RASTER_JFG_DEBUG_SHADOW 0x10 // paint where the casters block the frame's lights red
// padding.x bits 16-23: how dark the shadows of characters and objects are (0-255).
// padding.x bits 24-31: the level's ambient, a scale on its own colours (128 = as the game, 0-255 = 0 to ~2).
// padding.y: the frame's lights (low 16 bits) and shadow casters (high 16 bits) for the level lighting.
// padding.z (float bits): how much the level lighting depends on which way a surface faces its light (0-1).

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
