//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

// JFG PORT: one of the game's dynamic lights (its track lights) as the raster shaders get them for per-pixel level
// lighting (RasterPS). Full strength within the inner radius, fading linearly to nothing at the radius, as the game's
// own per-vertex lighting (trackLightAsm). Position in game world units, colour 0-1.
#ifdef HLSL_CPU
namespace interop {
#endif
    struct JfgLight {
        float4 posRadius;   // xyz = position, w = radius
        float4 colorInner;  // rgb = colour, w = inner radius
    };

    // JFG PORT: how the game lights one character or object this frame (its ObjectLighting, read by the port at
    // lightObject): brightness = ambient + the sum over its directions of range x how much the surface faces them, as
    // the game's lightSingleLight / lightTwoLights / lightMultiLight do per vertex, here per pixel with the model's
    // normals. 0-1 units (the game's 0-255 / 255).
    struct JfgObjectLight {
        float4 dirs[4];     // xyz = direction towards the light (world), w = range (max - min)
        float4 colors[4];   // rgb = that light's colour (its colour table in the game; white for the first)
        float4 params;      // x = ambient (min), y = directions used, z = how much the frame's lights add (0-1)
    };
#ifdef HLSL_CPU
};
#endif
