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
        float4 shadow;      // x = its tile row in the shadow map atlas + 1 (0 = casts no shadows), y = near, z = far
    };

    // JFG PORT: how the game lights one character or object this frame (its ObjectLighting, read by the port at
    // lightObject): brightness = ambient + the sum over its directions of range x how much the surface faces them, as
    // the game's lightSingleLight / lightTwoLights / lightMultiLight do per vertex, here per pixel with the model's
    // normals. 0-1 units (the game's 0-255 / 255).
    struct JfgObjectLight {
        float4 dirs[4];      // xyz = direction towards the light (world), w = range (max - min)
        float4 colors[4];    // rgb = that light's colour (its colour table in the game; white for the first)
        float4 positions[4]; // xyz = where that light is, w = 1 when known (for its shadows)
        float4 params;       // x = ambient (min), y = directions used, z = how much the frame's lights add (0-1)
    };

    // JFG PORT: what the lighting needs once per frame (FramebufferRenderer::setJfgLights, binding 70).
    struct JfgFrame {
        float4 camera;      // xyz = the camera's position (world), placed on in-between frames as the lights are
        float4 sunDir;      // xyz = direction towards the directional light (normalised), w = 1 when it is on
        float4 sunColor;    // rgb = its colour x intensity, w = how much it lights characters (0-1)
        float4 material;    // x = surface detail height (world units), y = shine strength, z = shine sharpness,
                            // w = how much a shadow takes from the level's painted glow (0-1)
        float4 shadowFade;  // how much each shadow map row's shadows count (rows 1-4), so lights fade in and out
        float4 extra;       // x = share of the surface detail on characters and objects (0-1), y = the directional
                            // light's shadow map depth half range, z = how much authored material maps count (0 = off)
        float4 sunShadow;   // xyz = centre of the directional light's shadow map, w = its half size (0 = none)
    };
#ifdef HLSL_CPU
};
#endif
