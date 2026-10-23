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
#ifdef HLSL_CPU
};
#endif
