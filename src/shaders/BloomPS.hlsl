//
// RT64 - JFG PORT: bloom for Jet Force Gemini.
//
// Three full-screen passes (FullScreenVS) run when the game finishes its 3D scene and starts drawing the HUD, so the
// HUD does not glow:
//   BLOOM_EXTRACT    scene colour -> small bloom buffer: box-averaged down to about 270 lines, keeping only what is
//                    brighter than the threshold (soft knee).
//   BLOOM_BLUR       bloom buffer -> bloom buffer: one direction of a separable gaussian.
//   BLOOM_COMPOSITE  bloom buffer -> scene colour: bilinear upscale, added with an additive blend.
// SSAO_BLUR is the second pass of the ambient occlusion (see SsaoPS.hlsl), here for the bloom passes' layout.
//

#include "shared/rt64_jfg_bloom.h"

[[vk::push_constant]] ConstantBuffer<BloomCB> gConstants : register(b0);

#if defined(BLOOM_EXTRACT) && defined(MULTISAMPLING)
Texture2DMS<float4> gInput : register(t1);
float3 loadInput(int2 p) { return gInput.Load(p, 0).rgb; }
void inputSize(out uint w, out uint h) { uint s; gInput.GetDimensions(w, h, s); }
#else
Texture2D<float4> gInput : register(t1);
float3 loadInput(int2 p) { return gInput.Load(int3(p, 0)).rgb; }
void inputSize(out uint w, out uint h) { gInput.GetDimensions(w, h); }
#endif

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    uint width, height;
    inputSize(width, height);
    const int2 maxP = int2(width - 1, height - 1);
    const int2 p = int2(pos.xy);

#if defined(BLOOM_EXTRACT)
    // Average a 4x4 grid spread over the block of source pixels this bloom pixel covers.
    const int scale = int(max(gConstants.scale, 1u));
    const int2 base = p * scale;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (int y = 0; y < 4; y++) {
        [unroll]
        for (int x = 0; x < 4; x++) {
            const int2 o = int2(((x * 2 + 1) * scale) / 8, ((y * 2 + 1) * scale) / 8);
            sum += loadInput(min(base + o, maxP));
        }
    }

    const float3 color = max(sum / 16.0f, 0.0f);
    const float brightness = max(color.r, max(color.g, color.b));
    const float knee = max(gConstants.knee, 1e-4f);
    float soft = clamp(brightness - gConstants.threshold + knee, 0.0f, 2.0f * knee);
    soft = (soft * soft) / (4.0f * knee);
    const float contribution = max(soft, brightness - gConstants.threshold) / max(brightness, 1e-4f);
    return float4(color * contribution, 1.0f);
#elif defined(BLOOM_BLUR)
    const float sigma = max(gConstants.sigma, 0.5f);
    const int taps = int(ceil(sigma * 3.0f));
    const int2 dir = int2(gConstants.directionX, gConstants.directionY);
    const float k = -0.5f / (sigma * sigma);
    float3 sum = loadInput(p);
    float weightSum = 1.0f;
    for (int i = 1; i <= taps; i++) {
        const float w = exp(float(i * i) * k);
        sum += (loadInput(clamp(p + dir * i, int2(0, 0), maxP)) + loadInput(clamp(p - dir * i, int2(0, 0), maxP))) * w;
        weightSum += 2.0f * w;
    }

    return float4(sum / weightSum, 1.0f);
#elif defined(SSAO_BLUR)
    // Ambient occlusion buffer (SsaoPS.hlsl: r occlusion, g distance, 0 for nothing drawn) -> multiplied into the scene.
    // A 7x7 gaussian that leaves out neighbours more than a few percent nearer or farther, so the occlusion of one
    // surface does not bleed onto another across an edge.
    const float2 centre = gInput.Load(int3(p, 0)).rg;
    if (centre.y <= 0.0f) {
        return float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    const float tolerance = 0.04f * centre.y;
    float sum = 0.0f;
    float weightSum = 0.0f;
    [unroll]
    for (int y = -3; y <= 3; y++) {
        [unroll]
        for (int x = -3; x <= 3; x++) {
            const float2 s = gInput.Load(int3(clamp(p + int2(x, y), int2(0, 0), maxP), 0)).rg;
            const float spatial = exp(-float(x * x + y * y) / 8.0f);
            const float dz = abs(s.y - centre.y);
            const float w = spatial * saturate(1.0f - dz / tolerance) * ((s.y > 0.0f) ? 1.0f : 0.0f);
            sum += s.x * w;
            weightSum += w;
        }
    }

    const float ao = (weightSum > 1e-4f) ? (sum / weightSum) : centre.x;
    return float4(ao, ao, ao, 1.0f);
#elif defined(BLOOM_COMPOSITE)
    // Bilinear sample of the bloom buffer at this full-resolution pixel.
    const float scale = float(max(gConstants.scale, 1u));
    const float2 src = (pos.xy / scale) - 0.5f;
    const int2 i0 = int2(floor(src));
    const float2 f = src - float2(i0);
    const float3 c00 = loadInput(clamp(i0, int2(0, 0), maxP));
    const float3 c10 = loadInput(clamp(i0 + int2(1, 0), int2(0, 0), maxP));
    const float3 c01 = loadInput(clamp(i0 + int2(0, 1), int2(0, 0), maxP));
    const float3 c11 = loadInput(clamp(i0 + int2(1, 1), int2(0, 0), maxP));
    const float3 bloom = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
    return float4(bloom * gConstants.intensity, 0.0f);
#else
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
}
