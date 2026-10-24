//
// RT64 - JFG PORT: the indirect light's irradiance probes (JfgProbeCS.hlsl updates them; RasterPS reads them). Needs a
// buffer of JfgProbe named jfgProbes (read-only or read-write) declared before it.
//
// The grid: JFG_PROBES_X x Y x Z probes, JfgFrame.gi.y units apart, from the cell JfgFrame.giOrigin.xyz (the camera's
// cell minus half the grid). A cell's probe is in slot (cell mod size), so moving the grid keeps the probes it still has.
//

#pragma once

uint jfgProbeSlot(int3 cell) {
    const int3 size = int3(JFG_PROBES_X, JFG_PROBES_Y, JFG_PROBES_Z);
    const int3 m = ((cell % size) + size) % size;
    return uint(m.x + size.x * (m.y + size.y * m.z));
}

// The cell a slot holds this frame: the one inside the grid that wraps onto it.
int3 jfgProbeCellOfSlot(uint slot, JfgFrame f) {
    const int3 size = int3(JFG_PROBES_X, JFG_PROBES_Y, JFG_PROBES_Z);
    const int3 m = int3(slot % uint(size.x), (slot / uint(size.x)) % uint(size.y), slot / uint(size.x * size.y));
    const int3 origin = int3(f.giOrigin.xyz);
    return origin + ((m - origin % size + size) % size);
}

// A probe's light coming from around n (its ambient cube).
float3 jfgProbeCube(JfgProbe probe, float3 n) {
    const float3 n2 = n * n;
    return n2.x * ((n.x >= 0.0f) ? probe.cube[0].rgb : probe.cube[1].rgb) +
        n2.y * ((n.y >= 0.0f) ? probe.cube[2].rgb : probe.cube[3].rgb) +
        n2.z * ((n.z >= 0.0f) ? probe.cube[4].rgb : probe.cube[5].rgb);
}

// The indirect light reaching a surface at p facing n: the eight probes around p, by distance (trilinear), less those
// behind the surface (so a probe under a floor or inside a wall doesn't bring its darkness or light through), and
// only those that hold their cell this frame.
float3 jfgProbeIrradiance(float3 p, float3 n, JfgFrame f) {
    if (f.gi.w < 0.5f) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    const float spacing = f.gi.y;
    const float3 g = p / spacing - 0.5f;
    const int3 base = int3(floor(g));
    const float3 t = g - float3(base);
    const int3 origin = int3(f.giOrigin.xyz);
    const int3 size = int3(JFG_PROBES_X, JFG_PROBES_Y, JFG_PROBES_Z);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (uint k = 0; k < 8; k++) {
        const int3 offset = int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        const int3 cell = base + offset;
        if (any(cell < origin) || any(cell >= origin + size)) {
            continue;
        }

        const JfgProbe probe = jfgProbes[jfgProbeSlot(cell)];
        if (any(probe.cell.xyz != cell) || (probe.cell.w == 0)) {
            continue;
        }

        const float3 trilinear = lerp(1.0f - t, t, float3(offset));
        const float3 toProbe = (float3(cell) + 0.5f) * spacing - p;
        const float side = dot(normalize(toProbe + n * 1e-3f), n);
        const float facing = saturate(side * 0.5f + 0.5f);
        const float weight = trilinear.x * trilinear.y * trilinear.z * (facing * facing + 0.02f);
        sum += jfgProbeCube(probe, n) * weight;
        weightSum += weight;
    }

    return (weightSum > 1e-5f) ? (sum / weightSum) : float3(0.0f, 0.0f, 0.0f);
}
