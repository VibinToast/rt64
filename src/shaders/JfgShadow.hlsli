//
// RT64 - JFG PORT: shadow maps for the game's lights (FramebufferRenderer::recordJfgShadows, JfgShadowVS, RasterPS).
//
// Each shadowed light gets a cube of depth maps around it, stored as one row of six square tiles in a depth atlas
// (row = its slot, column = cube face). A face looks down one axis (face / 2: x, y, z) in one direction (face & 1:
// +, -) with a 90 degree field of view, so the other two axes divided by the distance along it are the tile's
// coordinates. Shadows are thrown from JFG_SHADOW_LIFT units above each light: the game puts a torch's light at its
// bowl, about waist high, which makes every shadow a thin streak along the floor.
//

#define JFG_SHADOW_LIFT 40.0f
#define JFG_SUN_FACE 6 // JfgShadowVS: the directional light's map, below the cubes (two tiles square)

// The directional light's view: looking along -towardsSun, x right, y up (world up unless it looks straight down).
void jfgSunBasis(float3 towardsSun, out float3 right, out float3 up, out float3 forward) {
    forward = -towardsSun;
    const float3 worldUp = (abs(forward.y) > 0.99f) ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
    right = normalize(cross(worldUp, forward));
    up = cross(forward, right);
}

// The light-space vector v as (tile x, tile y, distance along the face's axis) for one face.
float3 jfgShadowFace(float3 v, uint face) {
    const uint axis = face >> 1;
    const float sign = ((face & 1) != 0) ? -1.0f : 1.0f;
    if (axis == 0) {
        return float3(v.z, v.y, v.x * sign);
    }
    else if (axis == 1) {
        return float3(v.x, v.z, v.y * sign);
    }
    else {
        return float3(v.x, v.y, v.z * sign);
    }
}

// The face that sees v: its dominant axis and direction.
uint jfgShadowFaceOf(float3 v) {
    const float3 a = abs(v);
    if ((a.x >= a.y) && (a.x >= a.z)) {
        return (v.x >= 0.0f) ? 0 : 1;
    }
    else if (a.y >= a.z) {
        return (v.y >= 0.0f) ? 2 : 3;
    }
    else {
        return (v.z >= 0.0f) ? 4 : 5;
    }
}

// Depth as the depth buffer stores it, 0 at near to 1 at far, for a distance d along the face's axis.
float jfgShadowDepth(float d, float n, float f) {
    return (f / (f - n)) - ((f * n) / ((f - n) * d));
}

// ... and back to the distance.
float jfgShadowDistance(float depth, float n, float f) {
    return (f * n) / (f - depth * (f - n));
}
