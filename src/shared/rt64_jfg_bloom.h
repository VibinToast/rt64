//
// RT64 - JFG PORT: parameters of the bloom passes (BloomPS.hlsl, FramebufferRenderer::submitRasterScene).
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct BloomCB {
        float threshold;    // extract: brightness where bloom starts (0..1)
        float knee;         // extract: width of the soft start above the threshold
        float intensity;    // composite: amount added back
        float sigma;        // blur: gaussian sigma in bloom-buffer pixels
        uint scale;         // extract: full-resolution pixels per bloom-buffer pixel; composite: same
        uint directionX;    // blur: 1 for the horizontal pass
        uint directionY;    // blur: 1 for the vertical pass
        uint padding;
    };
#ifdef HLSL_CPU
};
#endif
