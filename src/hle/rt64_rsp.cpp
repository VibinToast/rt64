//
// RT64
//

#include "rt64_rsp.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "../include/rt64_extended_gbi.h"
#include "common/rt64_common.h"
#include "common/rt64_math.h"
#include "gbi/rt64_f3d.h"
#include "shared/rt64_rsp_fog.h"

#include "rt64_interpreter.h"
#include "rt64_state.h"

// JFG PORT: the game tags the model matrices it writes (src/jfg/jfg_interpolation.cpp): the matrix's RDRAM address, an
// ID for the object (0 = none), a check value of the 64 bytes written, and the float model matrix the game multiplied
// with the camera to make them. The matrix heap is reused every other frame and not every matrix is tagged, so a tag
// only counts while the matrix at that address still has those bytes.
namespace {
    struct JfgMatrixTag {
        std::atomic<uint32_t> address{ 0 };
        std::atomic<uint32_t> id{ 0 };
        std::atomic<uint32_t> check{ 0 };
        std::atomic<uint32_t> sequence{ 0 }; // odd while the tag is being written
        float model[16] = {};
    };

    JfgMatrixTag jfgMatrixTags[1 << 14];

    JfgMatrixTag &jfgMatrixTagFor(uint32_t address) {
        return jfgMatrixTags[(address >> 6) & ((1 << 14) - 1)];
    }
}

uint32_t rt64_jfg_matrix_check(const uint8_t *rdram, uint32_t address) {
    uint32_t hash = 2166136261u; // FNV-1a over the fixed-point matrix (64 bytes)
    for (uint32_t i = 0; i < 64; i++) {
        hash = (hash ^ rdram[(address + i) & 0x00FFFFFF]) * 16777619u;
    }

    return hash;
}

void rt64_jfg_tag_matrix(uint32_t address, uint32_t id, uint32_t check, const float *model) {
    JfgMatrixTag &tag = jfgMatrixTagFor(address);
    tag.sequence.fetch_add(1, std::memory_order_acq_rel);
    tag.address.store(0, std::memory_order_relaxed);
    tag.id.store(id, std::memory_order_relaxed);
    tag.check.store(check, std::memory_order_relaxed);
    std::memcpy(tag.model, model, sizeof(tag.model));
    tag.address.store(address, std::memory_order_relaxed);
    tag.sequence.fetch_add(1, std::memory_order_release);
}

// The tag's model matrix, if the tag is for this address and was not being rewritten while it was read.
static bool jfgMatrixModel(uint32_t address, hlslpp::float4x4 &model) {
    const JfgMatrixTag &tag = jfgMatrixTagFor(address);
    const uint32_t before = tag.sequence.load(std::memory_order_acquire);
    if ((before & 1) || (tag.address.load(std::memory_order_relaxed) != address)) {
        return false;
    }

    float m[16];
    std::memcpy(m, tag.model, sizeof(m));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (tag.sequence.load(std::memory_order_relaxed) != before) {
        return false;
    }

    model = hlslpp::float4x4(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
    return true;
}

// The tag's ID, and its model matrix in model (hasModel), for a slot load from this address.
// JFG PORT: the level's own vertex arrays (its track segments, both copies when the game's lighting double-buffers
// them), sent by the game every frame (src/jfg/jfg_lighting.cpp) as physical [start, end) pairs. Draws whose vertices
// come from there get the per-pixel level lighting; characters and objects are lit by the game on the CPU instead.
namespace {
    std::mutex jfgLevelMutex;
    std::vector<uint32_t> jfgLevelRanges;
}

void rt64_jfg_level_vertices(const uint32_t *ranges, uint32_t count) {
    std::lock_guard<std::mutex> lock(jfgLevelMutex);
    jfgLevelRanges.assign(ranges, ranges + count * 2);
}

static bool jfgIsLevelVertex(uint32_t address) {
    std::lock_guard<std::mutex> lock(jfgLevelMutex);
    for (size_t i = 0; (i + 1) < jfgLevelRanges.size(); i += 2) {
        if ((address >= jfgLevelRanges[i]) && (address < jfgLevelRanges[i + 1])) {
            return true;
        }
    }

    return false;
}

// JFG PORT: the bone buffers of the object being drawn, tagged by the game (src/jfg/jfg_interpolation.cpp) with the
// object's ID. A model's bones are multiplied from the buffer the game's G_DMAOFFSETS points at.
namespace {
    struct JfgBoneBufferTag {
        std::atomic<uint32_t> address{ 0 };
        std::atomic<uint32_t> id{ 0 };
    };

    JfgBoneBufferTag jfgBoneBufferTags[1 << 12];

    JfgBoneBufferTag &jfgBoneBufferTagFor(uint32_t address) {
        return jfgBoneBufferTags[(address >> 6) & ((1 << 12) - 1)];
    }
}

void rt64_jfg_tag_bone_buffer(uint32_t address, uint32_t id) {
    JfgBoneBufferTag &tag = jfgBoneBufferTagFor(address);
    tag.address.store(0, std::memory_order_relaxed);
    tag.id.store(id, std::memory_order_relaxed);
    tag.address.store(address, std::memory_order_release);
}

// The object whose bone buffer holds this address (up to 128 bones in: a model part can start inside its buffer),
// and the address's distance from the buffer's start.
static uint32_t jfgBoneBufferId(uint32_t address, uint32_t &offsetInBuffer) {
    for (uint32_t back = 0; back <= 0x2000; back += 0x40) {
        if (back > address) {
            break;
        }

        const JfgBoneBufferTag &tag = jfgBoneBufferTagFor(address - back);
        if (tag.address.load(std::memory_order_acquire) == (address - back)) {
            offsetInBuffer = back;
            return tag.id.load(std::memory_order_relaxed);
        }
    }

    return 0;
}

static uint32_t jfgMatrixId(const uint8_t *rdram, uint32_t address, hlslpp::float4x4 &model, bool &hasModel) {
    hasModel = false;
    // Debug aid: RT64_INTERP_TRACE=1 also counts how slot loads found their tag.
    static const bool trace = std::getenv("RT64_INTERP_TRACE") != nullptr;
    static uint32_t loads = 0, untagged = 0, changed = 0, tagged = 0, notObject = 0;
    if (trace && (++loads % 2000) == 0) {
        fprintf(stderr, "[INTERP] matrix loads: tag with an ID %u, tagged 0 %u, no tag %u, bytes changed %u\n", tagged, notObject, untagged, changed);
        tagged = notObject = untagged = changed = 0;
    }

    const JfgMatrixTag &tag = jfgMatrixTagFor(address);
    if (tag.address.load(std::memory_order_acquire) != address) {
        untagged++;
        return 0;
    }

    const uint32_t id = tag.id.load(std::memory_order_relaxed);
    const uint32_t check = tag.check.load(std::memory_order_relaxed);
    if (check != rt64_jfg_matrix_check(rdram, address)) {
        changed++;
        return 0;
    }

    (id != 0) ? tagged++ : notObject++;
    hasModel = jfgMatrixModel(address, model);
    return id;
}

//#define LOG_SPECIAL_MATRIX_OPERATIONS

namespace RT64 {
    // RSP

    constexpr float DepthRange = 1024.0f;

    RSP::RSP(State *state) {
        this->state = state;

        NoN = false;
        cullBothMask = 0;
        cullFrontMask = 0;
        projMask = 0;
        loadMask = 0;
        pushMask = 0;
        shadingSmoothMask = 0;

        segments.fill(0);
        reset();
    }

    void RSP::reset() {
        modelMatrixStackSize = 1;
        projectionMatrixStackSize = 1;
        viewportStackSize = 1;
        geometryModeStackSize = 1;
        otherModeStackSize = 1;
        modelMatrixStack.fill(hlslpp::float4x4(0.0f));
        modelMatrixSegmentedAddressStack.fill(0);
        modelMatrixPhysicalAddressStack.fill(0);
        viewMatrixStack[0] = hlslpp::float4x4(0.0f);
        projMatrixStack[0] = hlslpp::float4x4(0.0f);
        viewProjMatrixStack[0] = hlslpp::float4x4(0.0f);
        invViewProjMatrixStack[0] = hlslpp::float4x4(0.0f);
        vertices.fill({});
        indices.fill(0);
        used.reset();
        lights.fill({});
        segments.fill(0);
        viewportStack[0] = {};
        clipRatios[0] = 1;
        clipRatios[1] = 1;
        clipRatios[2] = -1;
        clipRatios[3] = -1;
        textureState = {};
        curViewProjIndex = 0;
        curTransformIndex = 0;
        curFogIndex = 0;
        curLightIndex = 0;
        curLightCount = 0;
        curLookAtIndex = 0;
        projectionIndex = -1;
        projectionMatrixChanged = false;
        projectionMatrixInversed = false;
        viewportChanged = false;
        modelViewProjMatrix = hlslpp::float4x4(0.0f);
        modelViewProjChanged = false;
        modelViewProjInserted = false;
        lightCount = 0;
        lightsChanged = false;
        vertexFogIndex = 0;
        vertexLightIndex = 0;
        vertexLightCount = 0;
        vertexLookAtIndex = 0;
        vertexColorPDAddress = 0;
        fog.mul = 0.0f;
        fog.offset = 0.0f;
        lookAt.x = { 0.0f, 0.0f, 0.0f };
        lookAt.y = { 0.0f, 0.0f, 0.0f };
        otherModeStack[0].L = 0x0;
        otherModeStack[0].H = 0x080CFF;
        geometryModeStack[0] = G_CLIPPING;
        fogChanged = true;
        lookAtChanged = true;
        DKR.matrices.fill(hlslpp::float4x4::identity());
        DKR.segmentedAddresses.fill(0);
        DKR.physicalAddresses.fill(0);
        DKR.matrixOffset = 0;
        DKR.vertexOffset = 0;
        DKR.textureOffset = 0;
        DKR.textureShift = 0;
        DKR.textureCount = 0;
        DKR.loadedVertices.reset();
        DKR.activeMatrix = 0;
        DKR.vertexCursor = 0;
        DKR.billboard = false;
        DKR.jfgSplit = false; // JFG PORT
        DKR.jfgLevel = false;
        DKR.jfgIds.fill(0);
        DKR.jfgHasModel.fill(false);
        DKR.jfgObjectId = 0;
        DKR.jfgBoneOffset = 0;
        DKR.force8MBAddressMask = false;
        DKR.jfg = false;

        clearExtended();
    }

    constexpr uint32_t ExtendedMask = 0x80000000U;

    // Masks addresses as the RSP DMA hardware would.
    template<uint32_t mask> uint32_t RSP::maskPhysicalAddress(uint32_t address) {
        if (DKR.force8MBAddressMask) {
            return address & mask & 0x007FFFFFU;
        }
        else if (state->extended.extendRDRAM && ((address & ExtendedMask) == ExtendedMask)) {
            return address - ExtendedMask;
        }
        else {
            return address & mask;
        }
    }

    // Performs a lookup in the segment table to convert the given address.
    uint32_t RSP::fromSegmented(uint32_t segAddress) {
        if (!DKR.force8MBAddressMask && state->extended.extendRDRAM && ((segAddress & ExtendedMask) == ExtendedMask)) {
            return segAddress;
        }
        else {
            return segments[((segAddress) >> 24) & 0x0F] + ((segAddress) & 0x00FFFFFF);
        }
    }

    // Converts the given segmented address and then applies the RSP DMA physical address mask.
    // Used in cases where the RSP performs a DMA with a segmented address as the input. 
    uint32_t RSP::fromSegmentedMasked(uint32_t segAddress) {
        return maskPhysicalAddress<0x00FFFFF8>(fromSegmented(segAddress));
    }

    uint32_t RSP::fromSegmentedMaskedPD(uint32_t segAddress) {
        return maskPhysicalAddress<0x00FFFFFC>(fromSegmented(segAddress));
    }

    void RSP::setSegment(uint32_t seg, uint32_t address) {
        assert(seg < RSP_MAX_SEGMENTS);
        segments[seg] = address;
    }

    void RSP::matrixCommon(const hlslpp::float4x4 &floatMatrix, uint32_t address, uint8_t params) {
        // Projection matrix.
        hlslpp::float4x4 &viewMatrix = viewMatrixStack[projectionMatrixStackSize - 1];
        hlslpp::float4x4 &projMatrix = projMatrixStack[projectionMatrixStackSize - 1];
        hlslpp::float4x4 &viewProjMatrix = viewProjMatrixStack[projectionMatrixStackSize - 1];
        uint32_t &projectionMatrixSegmentedAddress = projectionMatrixSegmentedAddressStack[projectionMatrixStackSize - 1];
        uint32_t &projectionMatrixPhysicalAddress = projectionMatrixPhysicalAddressStack[projectionMatrixStackSize - 1];
        if (params & projMask) {
            if (params & loadMask) {
                viewProjMatrix = floatMatrix;

                if (isMatrixViewProj(floatMatrix)) {
                    matrixDecomposeViewProj(floatMatrix, viewMatrix, projMatrix);
                }
                else {
                    projMatrix = floatMatrix;
                    viewMatrix = hlslpp::float4x4::identity();
                }
            }
            else {
                viewProjMatrix = hlslpp::mul(floatMatrix, viewProjMatrix);

                if (isMatrixAffine(floatMatrix) && !isMatrixIdentity(floatMatrix)) {
                    viewMatrix = hlslpp::mul(floatMatrix, viewMatrix);
                }
                else {
                    projMatrix = hlslpp::mul(floatMatrix, projMatrix);
                }
            }

            projectionMatrixSegmentedAddress = address;
            projectionMatrixPhysicalAddress = fromSegmentedMasked(address);
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }
        // Modelview matrix.
        else {
            if ((params & pushMask) && (modelMatrixStackSize < RSP_MATRIX_STACK_SIZE)) {
                modelMatrixStackSize++;
                modelMatrixStack[modelMatrixStackSize - 1] = modelMatrixStack[modelMatrixStackSize - 2];
            }

            if (params & loadMask) {
                modelMatrixStack[modelMatrixStackSize - 1] = floatMatrix;
            }
            else {
                modelMatrixStack[modelMatrixStackSize - 1] = hlslpp::mul(floatMatrix, modelMatrixStack[modelMatrixStackSize - 1]);
            }

            modelMatrixSegmentedAddressStack[modelMatrixStackSize - 1] = address;
            modelMatrixPhysicalAddressStack[modelMatrixStackSize - 1] = fromSegmentedMasked(address);
        }

        modelViewProjChanged = true;
    }

    void RSP::matrix(uint32_t address, uint8_t params) {
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const FixedMatrix *fixedMatrix = reinterpret_cast<FixedMatrix *>(state->fromRDRAM(rdramAddress));
        const hlslpp::float4x4 floatMatrix = fixedMatrix->toMatrix4x4();
        matrixCommon(floatMatrix, address, params);
    }

    void RSP::matrixFloat(uint32_t address, uint8_t params) {
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const float *floats = reinterpret_cast<float *>(state->fromRDRAM(rdramAddress));
        const hlslpp::float4x4 floatMatrix(
            floats[0], floats[1], floats[2], floats[3],
            floats[4], floats[5], floats[6], floats[7],
            floats[8], floats[9], floats[10], floats[11],
            floats[12], floats[13], floats[14], floats[15]
        );

        matrixCommon(floatMatrix, address, params);
    }

    // JFG PORT: exact comparison, so an unchanged camera does not start a new projection.
    static bool matricesEqual(const hlslpp::float4x4 &a, const hlslpp::float4x4 &b) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                if (a[r][c] != b[r][c]) {
                    return false;
                }
            }
        }

        return true;
    }

    void RSP::matrixDKR(uint32_t address, uint8_t index) {
        if (index >= DKR.matrices.size()) {
            assert(false && "F3DDKR matrix index is invalid.");
            return;
        }

        const uint32_t rdramAddress = maskPhysicalAddress<0x00FFFFF8>(DKR.matrixOffset + fromSegmented(address));
        const FixedMatrix *fixedMatrix = reinterpret_cast<FixedMatrix *>(state->fromRDRAM(rdramAddress));
        DKR.matrices[index] = fixedMatrix->toMatrix4x4();
        DKR.segmentedAddresses[index] = address;
        DKR.physicalAddresses[index] = rdramAddress;
        selectMatrixDKR(index);
    }

    // JFG PORT: Jet Force Gemini's G_MTX (microcode handler at IMEM 0x13A8). The parameter byte is
    //   bit 7     : 0 = load the matrix into the destination slot, 1 = multiply
    //   bits 3..5 : source slot (multiply only)
    //   bits 0..2 : destination slot, which also becomes the active matrix
    // Multiply computes destination = new matrix * source slot (the standard Fast3D routine: the new matrix is
    // applied first).
    void RSP::matrixJFG(uint32_t address, uint8_t param) {
        const uint8_t dst = param & 0x7;
        const uint8_t src = (param >> 3) & 0x7;
        const uint32_t rdramAddress = maskPhysicalAddress<0x00FFFFF8>(DKR.matrixOffset + fromSegmented(address));
        const FixedMatrix *fixedMatrix = reinterpret_cast<FixedMatrix *>(state->fromRDRAM(rdramAddress));
        const hlslpp::float4x4 loaded = fixedMatrix->toMatrix4x4();
        DKR.matrices[dst] = (param & 0x80) ? hlslpp::mul(loaded, DKR.matrices[src]) : loaded;
        DKR.segmentedAddresses[dst] = address;
        DKR.physicalAddresses[dst] = rdramAddress;
        // The ID that lets RT64 match this matrix to the same one in the frame before: an object's model matrix as
        // the game tagged it, or a bone multiplied onto one (the parent's ID with the bone's place in its buffer).
        // The exact model matrix comes with the tag too (a bone: multiplied onto its parent's), so the model matrix
        // does not have to be worked out from the fixed-point result, which differs a little every frame.
        if (dst == 0) {
            DKR.jfgIds[0] = 0;
            DKR.jfgHasModel[0] = false;
            endObjectJFG(); // the camera of a new viewport (or frame): no object is being drawn
        }
        else if (param & 0x80) {
            // A model's bones (the game's animation matrices) are world-space model matrices multiplied onto the camera
            // in slot 0, so the loaded matrix is the bone's exact model matrix. Its ID: the object whose bone buffer
            // G_DMAOFFSETS points at (setDMAOffsetsDKR) with the bone's place in that buffer.
            const bool ontoCamera = (src == 0);
            const uint32_t parentId = ontoCamera ? DKR.jfgObjectId : DKR.jfgIds[src];
            const uint32_t bonePlace = DKR.jfgBoneOffset + (address & 0x00FFFFFFu);
            const uint32_t boneId = parentId ? ((parentId * 2654435761u) ^ (bonePlace * 40503u)) : 0;
            DKR.jfgIds[dst] = ((boneId == G_EX_ID_IGNORE) || (boneId == G_EX_ID_AUTO)) ? 0 : boneId;
            DKR.jfgHasModel[dst] = ontoCamera || DKR.jfgHasModel[src];
            if (ontoCamera) {
                DKR.jfgModels[dst] = loaded;
            }
            else if (DKR.jfgHasModel[dst]) {
                DKR.jfgModels[dst] = hlslpp::mul(loaded, DKR.jfgModels[src]);
            }
        }
        else {
            bool hasModel = false;
            DKR.jfgIds[dst] = jfgMatrixId(state->fromRDRAM(0), rdramAddress, DKR.jfgModels[dst], hasModel);
            DKR.jfgHasModel[dst] = hasModel;
        }

        // Debug aid (RT64_INTERP_TRACE): the kinds of matrix loads.
        static const bool loadTrace = std::getenv("RT64_INTERP_TRACE") != nullptr;
        if (loadTrace) {
            static uint32_t kinds[6] = {}, total = 0;
            const int kind = (dst == 0) ? 0 : (param & 0x80) ? ((DKR.jfgIds[dst] != 0) ? 1 : 2) : ((DKR.jfgIds[dst] != 0) ? 3 : 4);
            kinds[kind]++;
            kinds[5] += (dst != 0 && (param & 0x80) && src != 1) ? 1 : 0;
            static uint32_t pairs[8][8] = {};
            if (param & 0x80) pairs[src][dst]++;
            if (((total + 1) % 3000) == 0) {
                for (int s = 0; s < 8; s++) for (int t = 0; t < 8; t++) if (pairs[s][t]) fprintf(stderr, "[INTERP] multiply %d -> %d: %u\n", s, t, pairs[s][t]);
                memset(pairs, 0, sizeof(pairs));
            }
            if ((++total % 3000) == 0) {
                fprintf(stderr, "[INTERP] slot loads: camera %u, bone with ID %u, bone without %u, load with ID %u, load without %u (bones not onto slot 1: %u)\n",
                    kinds[0], kinds[1], kinds[2], kinds[3], kinds[4], kinds[5]);
                memset(kinds, 0, sizeof(kinds));
            }
        }

        if (dst == 0) {
            setCameraJFG(DKR.matrices[0]);
        }

        selectMatrixDKR(dst);
    }

    // JFG PORT: bones multiplied from now on belong to no object until G_DMAOFFSETS points into a tagged bone buffer.
    void RSP::endObjectJFG() {
        DKR.jfgObjectId = 0;
        DKR.jfgBoneOffset = 0;
    }

    // JFG PORT: slot 0 was loaded. A perspective camera has a w column that depends on position (the game's
    // guPerspectiveF); the HUD's orthographic matrix has (0, 0, 0, k) there and cannot be inverted, so it is not split.
    // RT64's isMatrixViewProj() is not used: it accepts that orthographic matrix too (its [3][3] is 160).
    void RSP::setCameraJFG(const hlslpp::float4x4 &viewProj) {
        static const bool splitDisabled = (std::getenv("RT64_JFG_NO_SPLIT") != nullptr); // debug aid: the old behaviour
        const float wDepends = std::abs(viewProj[0][3]) + std::abs(viewProj[1][3]) + std::abs(viewProj[2][3]);
        DKR.jfgSplit = false;
        if (splitDisabled || (wDepends < 1e-6f)) {
            return;
        }

        const hlslpp::float4x4 inverse = hlslpp::inverse(viewProj);
        if (matrixIsNaN(inverse)) {
            return;
        }

        // The camera is view x projection with a rigid view and the projection of guPerspectiveF (x and y scale, a z
        // mapping, w = -z), so (row vectors, VP = V * P): VP column 0 = V column 0 * sx, column 1 = V column 1 * sy,
        // column 3 = V column 2 * -k, and column 2 = V column 2 * a + (0, 0, 0, b). This is solved from whole columns;
        // RT64's matrixDecomposeViewProj divides by single elements, which are 0 when the camera looks along an axis.
        auto length3 = [](float x, float y, float z) { return std::sqrt(x * x + y * y + z * z); };
        const float sx = length3(viewProj[0][0], viewProj[1][0], viewProj[2][0]);
        const float sy = length3(viewProj[0][1], viewProj[1][1], viewProj[2][1]);
        const float k = length3(viewProj[0][3], viewProj[1][3], viewProj[2][3]);
        if ((sx < 1e-6f) || (sy < 1e-6f) || (k < 1e-6f)) {
            return;
        }

        hlslpp::float4x4 view = hlslpp::float4x4::identity();
        hlslpp::float4x4 proj(0.0f);
        for (int r = 0; r < 4; r++) {
            view[r][0] = viewProj[r][0] / sx;
            view[r][1] = viewProj[r][1] / sy;
            view[r][2] = -viewProj[r][3] / k;
            view[r][3] = (r == 3) ? 1.0f : 0.0f;
        }

        const float a = viewProj[0][2] * view[0][2] + viewProj[1][2] * view[1][2] + viewProj[2][2] * view[2][2];
        proj[0][0] = sx;
        proj[1][1] = sy;
        proj[2][2] = a;
        proj[2][3] = -k;
        proj[3][2] = viewProj[3][2] - view[3][2] * a;
        if (matrixIsNaN(view) || matrixIsNaN(proj)) {
            return;
        }

        DKR.jfgViewProj = viewProj;
        DKR.jfgInvViewProj = inverse;
        DKR.jfgView = view;
        DKR.jfgProj = proj;
        DKR.jfgSplit = true;
    }

    void RSP::selectMatrixDKR(uint8_t index) {
        if (index >= DKR.matrices.size()) {
            assert(false && "F3DDKR matrix index is invalid.");
            return;
        }

        DKR.activeMatrix = index;
        modelMatrixStackSize = 1;
        const bool jfgSplit = DKR.jfg && DKR.jfgSplit; // JFG PORT: see setCameraJFG
        if (jfgSplit && (index != 0) && DKR.jfgHasModel[index]) {
            modelMatrixStack[0] = DKR.jfgModels[index];

            // Debug aid (RT64_INTERP_TRACE): how far the model matrix worked out from the fixed-point result is off.
            static const bool trace = std::getenv("RT64_INTERP_TRACE") != nullptr;
            if (trace) {
                static float worst = 0.0f;
                static uint32_t count = 0;
                const hlslpp::float4x4 worked = hlslpp::mul(DKR.matrices[index], DKR.jfgInvViewProj);
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        worst = std::max(worst, std::abs(worked[r][c] - DKR.jfgModels[index][r][c]));
                    }
                }

                if ((++count % 5000) == 0) {
                    fprintf(stderr, "[INTERP] model worked out from the fixed-point matrix: largest difference %g\n", worst);
                    worst = 0.0f;
                }
            }
        }
        else if (jfgSplit) {
            modelMatrixStack[0] = (index == 0) ? hlslpp::float4x4::identity() : hlslpp::mul(DKR.matrices[index], DKR.jfgInvViewProj);

            // A model matrix has no perspective part: what the inverse leaves in column 3 is rounding. Exact zeros and
            // one there keep RT64 from treating the matrix as perspective (decomposing it for interpolation fails).
            hlslpp::float4x4 &model = modelMatrixStack[0];
            if ((std::abs(model[0][3]) < 1e-3f) && (std::abs(model[1][3]) < 1e-3f) && (std::abs(model[2][3]) < 1e-3f) && (std::abs(model[3][3] - 1.0f) < 1e-3f)) {
                model[0][3] = 0.0f;
                model[1][3] = 0.0f;
                model[2][3] = 0.0f;
                model[3][3] = 1.0f;
            }
        }
        else {
            modelMatrixStack[0] = DKR.matrices[index];
        }
        modelMatrixSegmentedAddressStack[0] = DKR.segmentedAddresses[index];
        modelMatrixPhysicalAddressStack[0] = DKR.physicalAddresses[index];

        // F3DDKR supplies complete model-view-projection matrices from a heap
        // whose slots are reused for different racer parts between frames.
        // Cross-frame transform matching therefore stretches body geometry
        // between unrelated matrices. Keep these transforms current-frame-only;
        // the game still advances them at its native update rate.
        TransformGroup &group = extended.modelMatrixIdStack[0];
        group.matrixId = G_EX_ID_IGNORE;
        group.decompose = false;
        group.positionInterpolation = G_EX_COMPONENT_SKIP;
        group.rotationInterpolation = G_EX_COMPONENT_SKIP;
        group.scaleInterpolation = G_EX_COMPONENT_SKIP;
        group.skewInterpolation = G_EX_COMPONENT_SKIP;
        group.perspectiveInterpolation = G_EX_COMPONENT_SKIP;
        // Also skip tile interpolation: with the default AUTO, every pair of
        // hash-identical draw calls (hundreds per frame in DKR's wave
        // renderer) enters the frame-matcher's quadratic tile-matching loop,
        // which stalls the workload thread for the entire frame.
        group.tileInterpolation = G_EX_COMPONENT_SKIP;
        group.ordering = G_EX_ORDER_LINEAR;

        // JFG PORT: with the camera split off, a matrix the game tagged with its object (setCameraJFG, matrixJFG) is
        // matched to the same object's matrix in the frame before by that ID, in the order they were drawn, so the
        // heap reuse described above cannot pair unrelated objects.
        static const bool jfgNoIds = (std::getenv("RT64_JFG_NO_IDS") != nullptr); // debug aid: the camera split without IDs
        const uint32_t jfgId = (jfgSplit && !jfgNoIds) ? DKR.jfgIds[index] : 0;
        if (jfgId != 0) {
            // Blended as matrices (no decomposition): between two 60 fps frames an object moves and turns very little.
            group.matrixId = jfgId;
            group.decompose = false;
            group.positionInterpolation = G_EX_COMPONENT_INTERPOLATE;
            group.rotationInterpolation = G_EX_COMPONENT_INTERPOLATE;
            group.scaleInterpolation = G_EX_COMPONENT_INTERPOLATE;
            group.skewInterpolation = G_EX_COMPONENT_INTERPOLATE;
            group.perspectiveInterpolation = G_EX_COMPONENT_SKIP;
        }

        extended.modelMatrixIdStackChanged = true;

        // F3DDKR matrices are complete MVP matrices. Treat them as the model
        // transform and keep RT64's view/projection side at identity. Do not
        // invalidate an already-identity projection on every matrix switch:
        // racer meshes intentionally draw triangles across a base load under
        // matrix 1 and an append load under matrix 2. Creating a new projection
        // for the append makes those vertices use a different view-projection
        // index, and only the draw-bearing later projection is aspect-adjusted.
        // That stretches the mixed seam triangles in widescreen mode.
        const uint32_t projectionStackIndex = projectionMatrixStackSize - 1;
        if (jfgSplit) {
            // JFG PORT: the camera is the projection for the whole viewport; it only changes when slot 0 is loaded again,
            // so the seam triangles described above keep one view-projection index here too.
            const bool cameraChanged =
                !matricesEqual(viewProjMatrixStack[projectionStackIndex], DKR.jfgViewProj) ||
                !matricesEqual(viewMatrixStack[projectionStackIndex], DKR.jfgView) ||
                !matricesEqual(projMatrixStack[projectionStackIndex], DKR.jfgProj);
            if (cameraChanged) {
                viewMatrixStack[projectionStackIndex] = DKR.jfgView;
                projMatrixStack[projectionStackIndex] = DKR.jfgProj;
                viewProjMatrixStack[projectionStackIndex] = DKR.jfgViewProj;
                projectionMatrixChanged = true;
                projectionMatrixInversed = false;
            }

            modelViewProjChanged = true;
            return;
        }

        const bool resetProjectionToIdentity =
            !isMatrixIdentity(viewMatrixStack[projectionStackIndex]) ||
            !isMatrixIdentity(projMatrixStack[projectionStackIndex]) ||
            !isMatrixIdentity(viewProjMatrixStack[projectionStackIndex]);
        if (resetProjectionToIdentity) {
            viewMatrixStack[projectionStackIndex] = hlslpp::float4x4::identity();
            projMatrixStack[projectionStackIndex] = hlslpp::float4x4::identity();
            viewProjMatrixStack[projectionStackIndex] = hlslpp::float4x4::identity();
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }

        modelViewProjChanged = true;
    }

    void RSP::setBillboardDKR(bool enabled) {
        DKR.billboard = enabled;
    }

    void RSP::setDMAOffsetsDKR(uint32_t matrixOffset, uint32_t vertexOffset) {
        // The microcode's offset moveword only updates the segment table in
        // DMEM; the vertex cursor (0x14C) and the transformed vertex slots
        // are untouched.
        DKR.matrixOffset = matrixOffset;

        // JFG PORT: the object whose bones follow is the one whose bone buffer the offset points into (tagged by the
        // game); 0 (the game resets it when the bones are drawn) or an untagged buffer is no object.
        if (DKR.jfg) {
            uint32_t offsetInBuffer = 0;
            const uint32_t id = (matrixOffset != 0) ? jfgBoneBufferId(matrixOffset & 0x00FFFFFFu, offsetInBuffer) : 0;
            DKR.jfgObjectId = id;
            DKR.jfgBoneOffset = id ? offsetInBuffer : 0;
        }
        DKR.vertexOffset = vertexOffset;
    }

    void RSP::setDMATextureOffsetDKR(uint32_t address) {
        DKR.textureOffset = maskPhysicalAddress<0x00FFFFF8>(fromSegmented(address));
        DKR.textureShift = 0;
        DKR.textureCount = 0;
    }

    uint32_t RSP::textureImageAddressDKR(uint8_t fmt, uint32_t address) {
        uint32_t physicalAddress = maskPhysicalAddress<0x00FFFFFF>(fromSegmented(address));
        if (DKR.jfg) {
            // JFG PORT: Jet Force Gemini's microcode (handler at IMEM 0x1328) is simpler than DKR's. The texture
            // offset command copies 32 16-bit offsets into DMEM and sets a flag; from then on EVERY G_SETTIMG,
            // whatever its format, adds the next offset to its address. Only G_DMAOFFSETS clears the flag.
            // Animated textures (force fields, weapon reticles, ...) depend on this to select their frame.
            if ((DKR.textureOffset != 0) && (DKR.textureCount < 32)) {
                const uint32_t shiftAddress = DKR.textureOffset + DKR.textureCount * sizeof(uint16_t);
                DKR.textureShift = *reinterpret_cast<const uint16_t *>(state->fromRDRAM(shiftAddress ^ 0x2));
                DKR.textureCount++;
                physicalAddress += DKR.textureShift;
            }

            return physicalAddress;
        }

        if (DKR.textureOffset != 0) {
            if (fmt == G_IM_FMT_RGBA) {
                const uint32_t shiftAddress = DKR.textureOffset + DKR.textureCount * sizeof(uint16_t);
                DKR.textureShift = *reinterpret_cast<const uint16_t *>(state->fromRDRAM(shiftAddress ^ 0x2));
                physicalAddress += DKR.textureShift;
            }
            else {
                DKR.textureOffset = 0;
                DKR.textureShift = 0;
                DKR.textureCount = 0;
            }
        }

        return physicalAddress;
    }

    void RSP::loadBlockDKR(uint16_t lrs) {
        if (DKR.jfg || (DKR.textureOffset == 0)) {
            return;
        }

        const uint32_t blockBytes = (((lrs >> 2) + 1) << 3);
        if ((blockBytes != 0) && ((DKR.textureShift % blockBytes) != 0)) {
            state->rdp->texture.address -= DKR.textureShift;
            DKR.textureOffset = 0;
            DKR.textureShift = 0;
            DKR.textureCount = 0;
        }
        else {
            DKR.textureCount++;
        }
    }

    void RSP::popMatrix(uint32_t count) {
        while (count--) {
            if (modelMatrixStackSize > 1) {
                modelMatrixStackSize--;
                modelViewProjChanged = true;
            }
        }
    }

    void RSP::pushProjectionMatrix() {
        if (projectionMatrixStackSize < RSP_EXTENDED_STACK_SIZE) {
            viewMatrixStack[projectionMatrixStackSize] = viewMatrixStack[projectionMatrixStackSize - 1];
            projMatrixStack[projectionMatrixStackSize] = projMatrixStack[projectionMatrixStackSize - 1];
            viewProjMatrixStack[projectionMatrixStackSize] = viewProjMatrixStack[projectionMatrixStackSize - 1];
            invViewProjMatrixStack[projectionMatrixStackSize] = invViewProjMatrixStack[projectionMatrixStackSize - 1];
            projectionMatrixSegmentedAddressStack[projectionMatrixStackSize] = projectionMatrixSegmentedAddressStack[projectionMatrixStackSize - 1];
            projectionMatrixPhysicalAddressStack[projectionMatrixStackSize] = projectionMatrixPhysicalAddressStack[projectionMatrixStackSize - 1];
            projectionMatrixStackSize++;
        }
    }

    void RSP::popProjectionMatrix() {
        if (projectionMatrixStackSize > 1) {
            projectionMatrixStackSize--;
            modelViewProjChanged = true;
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }
    }
    
    void RSP::insertMatrix(uint32_t address, uint32_t value) {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::insertMatrix(dst %u, value %u)", dst, value);
#   endif

        // We assume unaligned addresses or overlapping is impossible until we have verification
        // from the microcode itself if this is allowed or not due to how much more complex
        // the implementation could get.
        assert(((address & 0x3) == 0) && "Unaligned addresses for insert matrix are not currently supported.");

        // Copies a 32-bit value into the location occupied by the fixed point matrices.
        const uint16_t MatrixSize = 0x40;
        const uint16_t FractionalMatrixAddress = MatrixSize / 2;
        const uint16_t ModelAddress = 0x0;
        const uint16_t ViewProjAddress = ModelAddress + MatrixSize;
        const uint16_t ModelViewProjAddress = ViewProjAddress + MatrixSize;

        // According to the microcode, the address requires this kind of wrapping
        // to access the real destination.
        uint32_t dstAddr = (address + ModelViewProjAddress) & 0xFFFFU;

        // Figure out which matrix should be modified and compute a relative address to it.
        uint32_t relAddr = 0;
        hlslpp::float4x4 *dstMat = nullptr;
        hlslpp::float4x4 &viewProjMatrix = viewProjMatrixStack[projectionMatrixStackSize - 1];
        if (dstAddr >= (ModelViewProjAddress + MatrixSize)) {
            assert(false && "Undefined behavior due to destination address extending outside of the allowed bounds.");
            return;
        }
        if (dstAddr >= ModelViewProjAddress) {
            dstMat = &modelViewProjMatrix;
            relAddr = dstAddr - ModelViewProjAddress;
            modelViewProjInserted = true;
        }
        else if (dstAddr >= ViewProjAddress) {
            dstMat = &viewProjMatrix;
            relAddr = dstAddr - ViewProjAddress;
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }
        else if (dstAddr >= ModelAddress) {
            dstMat = &modelMatrixStack[modelMatrixStackSize - 1];
            relAddr = dstAddr - ModelAddress;
        }

        // Modify two fractional parts or two integer parts.
        const bool modifyFractional = relAddr >= FractionalMatrixAddress;
        if (modifyFractional) {
            relAddr -= FractionalMatrixAddress;
        }

        const uint32_t index = relAddr / 2;
        const uint32_t row = index / 4;
        const uint32_t column = index % 4;
        if (modifyFractional) {
            FixedMatrix::modifyMatrix4x4Fraction(*dstMat, row, column, uint16_t((value >> 16U) & 0xFFFFU));
            FixedMatrix::modifyMatrix4x4Fraction(*dstMat, row, column + 1, uint16_t(value & 0xFFFFU));
        }
        else {
            FixedMatrix::modifyMatrix4x4Integer(*dstMat, row, column, int16_t((value >> 16) & 0xFFFF));
            FixedMatrix::modifyMatrix4x4Integer(*dstMat, row, column + 1, int16_t(value & 0xFFFF));
        }
    }

    void RSP::forceMatrix(uint32_t address) {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::forceMatrix(0x%08X)", address);
#   endif

        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const FixedMatrix *fixedMatrix = reinterpret_cast<FixedMatrix *>(state->fromRDRAM(rdramAddress));
        modelViewProjMatrix = fixedMatrix->toMatrix4x4();
        modelViewProjInserted = true;
        modelViewProjChanged = false;
    }

    static void setExtendedMatrixFloat(RSP &rsp, uint32_t address, hlslpp::float4x4 &matrix, hlslpp::float4x4 &invMatrix) {
        const uint32_t rdramAddress = rsp.fromSegmentedMasked(address);
        const float *floatMatrix = reinterpret_cast<float *>(rsp.state->fromRDRAM(rdramAddress));
        for (uint32_t j = 0; j < 4; j++) {
            for (uint32_t i = 0; i < 4; i++) {
                matrix[j][i] = floatMatrix[j * 4 + i];
            }
        }

        invMatrix = hlslpp::inverse(matrix);

        rsp.extended.viewProjMatrix = hlslpp::mul(rsp.extended.viewMatrix, rsp.extended.projMatrix);
        rsp.extended.viewProjRotationMatrix = hlslpp::float3x3(rsp.extended.viewProjMatrix);
        rsp.extended.invViewProjMatrix = hlslpp::inverse(rsp.extended.viewProjMatrix);
        rsp.projectionMatrixChanged = true;
        rsp.modelViewProjChanged = true;
        rsp.lightsChanged = true;
        rsp.lookAtChanged = true;
    }

    void RSP::setProjectionMatrixFloat(uint32_t address) {
        setExtendedMatrixFloat(*this, address, extended.projMatrix, extended.invProjMatrix);
    }

    void RSP::setViewMatrixFloat(uint32_t address) {
        setExtendedMatrixFloat(*this, address, extended.viewMatrix, extended.invViewMatrix);
    }

    void RSP::computeModelViewProj() {
        const hlslpp::float4x4 &viewProjMatrix = viewProjMatrixStack[projectionMatrixStackSize - 1];
        modelViewProjMatrix = hlslpp::mul(modelMatrixStack[modelMatrixStackSize - 1], viewProjMatrix);
        modelViewProjInserted = false;
        modelViewProjChanged = false;
    }

    void RSP::specialComputeModelViewProj() {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::specialComputeModelViewProj()");
#   endif

        computeModelViewProj();
    }

    void RSP::setModelViewProjChanged(bool changed) {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::setModelViewProjChanged(%d)", changed);
#   endif

        modelViewProjChanged = changed;
    }

    void RSP::setVertex(uint32_t address, uint32_t vtxCount, uint32_t dstIndex) {
        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "Vertex indices are not valid. DL is possibly corrupted.");
            return;
        }

        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const Vertex *dlVerts = reinterpret_cast<const Vertex *>(state->fromRDRAM(rdramAddress));
        memcpy(&vertices[dstIndex], dlVerts, sizeof(Vertex) * vtxCount);
        setVertexCommon<true, sizeof(Vertex)>(rdramAddress, dstIndex, dstIndex + vtxCount);
    }

    void RSP::setVertexDKR(uint32_t address, uint32_t vtxCount, uint32_t sourceOffset, bool append, int32_t dstIndexJFG) {
        // Microcode-verified placement rule (DMEM 0x14C cursor): a non-append
        // load always writes to slot 0 and sets the cursor to its count; an
        // append load writes to the cursor and leaves it unchanged. This is
        // how racer models stitch body (matrix 1) and head/tail (matrix 2)
        // vertices contiguously, and how sprites reuse slots 1..n across
        // triangle batches while slot 0 keeps the anchor vertex.
        uint32_t dstIndex;
        if (dstIndexJFG >= 0) {
            // JFG PORT: Jet Force Gemini's revision has no vertex cursor; the command names its destination slot.
            // The billboard anchor (slot 0) is added whenever billboarding is on, so treat any load that does not
            // start at slot 0 like DKR's append case below.
            dstIndex = uint32_t(dstIndexJFG);
            append = (dstIndex != 0);
        }
        else if (!append) {
            dstIndex = 0;
            DKR.vertexCursor = uint8_t(vtxCount);
        }
        else {
            dstIndex = DKR.vertexCursor;
        }

        // DMEM vertex slots persist across batches on hardware, and DKR's
        // larger models rely on it: triangle batches reference seam vertices
        // loaded by a previous batch that were never re-uploaded. The
        // validity bits therefore only accumulate; they guard against slots
        // that were never written since the task started.

        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "F3DDKR vertex indices are not valid.");
            return;
        }

        const uint32_t rdramAddress = maskPhysicalAddress<0x00FFFFF8>(DKR.vertexOffset + fromSegmented(address)) + sourceOffset;
        if (DKR.jfg) { // JFG PORT: level geometry or not (see rt64_jfg_level_vertices)
            const bool level = jfgIsLevelVertex(rdramAddress);
            if (level != DKR.jfgLevel) {
                DKR.jfgLevel = level;
                state->updateDrawStatusAttribute(DrawAttribute::ExtendedFlags);
            }
        }
        for (uint32_t i = 0; i < vtxCount; i++) {
            const uint32_t src = rdramAddress + i * 10;
            Vertex &dst = vertices[dstIndex + i];
            dst.x = *reinterpret_cast<const int16_t *>(state->fromRDRAM(src ^ 0x2));
            dst.y = *reinterpret_cast<const int16_t *>(state->fromRDRAM((src + 2) ^ 0x2));
            dst.z = *reinterpret_cast<const int16_t *>(state->fromRDRAM((src + 4) ^ 0x2));
            dst.flag = 0;
            dst.s = 0;
            dst.t = 0;
            dst.color.r = *state->fromRDRAM((src + 6) ^ 0x3);
            dst.color.g = *state->fromRDRAM((src + 7) ^ 0x3);
            dst.color.b = *state->fromRDRAM((src + 8) ^ 0x3);
            dst.color.a = *state->fromRDRAM((src + 9) ^ 0x3);
        }

        // F3DDKR billboards transform vertex zero with the regular MVP, then
        // add its clip-space position to every appended billboard vertex. Add
        // that anchor to the translation row so both RT64's CPU bookkeeping
        // and the GPU world transform reproduce the microcode operation.
        const hlslpp::float4x4 unanchoredMatrix = modelMatrixStack[0];
        bool anchoredBillboard = false;

        // F3DDKR intentionally permits a triangle to span vertices loaded
        // under matrix 1 and vertices appended under matrix 2. RT64 already
        // records a world-transform index per vertex, so keeping each load in
        // its original coordinate space reproduces the microcode directly.
        // Transforming the appended vertices on the CPU and quantizing them
        // back to int16 corrupts exactly the mixed head/tail seam triangles
        // used by close-range racer models.

        if (append && DKR.billboard) {
            const int workloadCursor = state->ext.workloadQueue->writeCursor;
            Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
            const uint32_t anchorIndex = indices[0];
            if (anchorIndex < workload.drawData.posTransformed.size()) {
                // JFG PORT: with the camera split off (setCameraJFG), the model matrix is slot x inverse(camera), so the
                // clip-space anchor goes onto the slot's own matrix and the result is moved into model space again.
                const bool jfgSplit = DKR.jfg && DKR.jfgSplit;
                if (jfgSplit) {
                    modelMatrixStack[0] = DKR.matrices[DKR.activeMatrix];
                }

                modelMatrixStack[0][3] += workload.drawData.posTransformed[anchorIndex];
                // Keep the anchor's clip-space W for RT64's matrix-based
                // billboard emulation. Adding the local W term here creates
                // huge screen-space primitives and pathological overdraw.
                modelMatrixStack[0][3].w = workload.drawData.posTransformed[anchorIndex].w;
                if (jfgSplit) {
                    modelMatrixStack[0] = hlslpp::mul(modelMatrixStack[0], DKR.jfgInvViewProj);
                }

                modelViewProjChanged = true;
                anchoredBillboard = true;
            }
        }

        setVertexCommon<true, 10>(rdramAddress, dstIndex, dstIndex + vtxCount);

        for (uint32_t i = dstIndex; i < (dstIndex + vtxCount); i++) {
            DKR.loadedVertices.set(i);
        }

        if (anchoredBillboard) {
            modelMatrixStack[0] = unanchoredMatrix;
            modelMatrixSegmentedAddressStack[0] = DKR.segmentedAddresses[DKR.activeMatrix];
            modelMatrixPhysicalAddressStack[0] = DKR.physicalAddresses[DKR.activeMatrix];
            modelViewProjChanged = true;
        }
    }
    
    void RSP::setVertexPD(uint32_t address, uint32_t vtxCount, uint32_t dstIndex) {
        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "Vertex indices are not valid. DL is possibly corrupted.");
            return;
        }

        const uint32_t rdramAddress = fromSegmentedMaskedPD(address);
        const VertexPD *dlVerts = reinterpret_cast<const VertexPD *>(state->fromRDRAM(rdramAddress));
        for (uint32_t i = 0; i < vtxCount; i++) {
            Vertex &dst = vertices[dstIndex + i];
            const VertexPD &src = dlVerts[i];
            const uint8_t *col = state->fromRDRAM(vertexColorPDAddress + (src.ci & 0xFFU));
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.s = src.s;
            dst.t = src.t;
            dst.color.r = col[3];
            dst.color.g = col[2];
            dst.color.b = col[1];
            dst.color.a = col[0];
        }

        setVertexCommon<true, sizeof(VertexPD)>(rdramAddress, dstIndex, dstIndex + vtxCount);
    }

    void RSP::setVertexEXV1(uint32_t address, uint32_t vtxCount, uint32_t dstIndex) {
        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "Vertex indices are not valid. DL is possibly corrupted.");
            return;
        }

        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const VertexEXV1 *dlVerts = reinterpret_cast<const VertexEXV1 *>(state->fromRDRAM(rdramAddress));
        auto &velFloats = workload.drawData.velFloats;
        auto &tcVelFloats = workload.drawData.tcVelFloats;
        for (uint32_t i = 0; i < vtxCount; i++) {
            const VertexEXV1 &src = dlVerts[i];
            velFloats.emplace_back(float(src.v.x - src.xp));
            velFloats.emplace_back(float(src.v.y - src.yp));
            velFloats.emplace_back(float(src.v.z - src.zp));
            tcVelFloats.emplace_back(0.0f);
            tcVelFloats.emplace_back(0.0f);
            vertices[dstIndex + i] = src.v;
        }

        setVertexCommon<false, sizeof(VertexEXV1)>(rdramAddress, dstIndex, dstIndex + vtxCount);
    }

    void RSP::setVertexColorPD(uint32_t address) {
        vertexColorPDAddress = fromSegmentedMasked(address);
    }

    void RSP::setVertexSegmentV1(bool isEnabled, uint32_t vertexElement, uint32_t vertexAddress, uint32_t baseSegmentAddress) {
        extended.vertexAddresses[vertexElement] = vertexAddress;
        extended.baseSegmentAddresses[vertexElement] = baseSegmentAddress;
        extended.vertexSegmentEnabled[vertexElement] = isEnabled;
    }

    Projection::Type RSP::getCurrentProjectionType() const {
        const hlslpp::float4x4 &projMatrix = projMatrixStack[projectionMatrixStackSize - 1];
        const bool perspProj = (projMatrix[3][3] == 0.0f) && (abs(projMatrix[1][1]) > 1e-6f);
        if (perspProj) {
            return Projection::Type::Perspective;
        }
        else {
            return Projection::Type::Orthographic;
        }
    }
    
    void RSP::addCurrentProjection(Projection::Type type) {
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        if (extended.viewProjMatrixIdStackChanged) {
            extended.curViewProjMatrixIdGroupIndex = int(workload.drawData.transformGroups.size());
            workload.drawData.transformGroups.emplace_back(extended.viewProjMatrixIdStack[extended.viewProjMatrixIdStackSize - 1]);
            extended.viewProjMatrixIdStackChanged = false;
        }

        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        if (projectionMatrixChanged || viewportChanged) {
            DrawData &drawData = workload.drawData;
            uint32_t transformsIndex = uint32_t(drawData.viewTransforms.size());
            curViewProjIndex = transformsIndex;

            uint32_t physicalAddress = projectionMatrixPhysicalAddressStack[projectionMatrixStackSize - 1];
            workload.physicalAddressTransformMap.emplace(physicalAddress, uint32_t(drawData.viewProjTransformGroups.size()));
            drawData.viewTransforms.emplace_back(hlslpp::mul(extended.invViewMatrix, viewMatrixStack[projectionMatrixStackSize - 1]));
            drawData.projTransforms.emplace_back(hlslpp::mul(extended.invProjMatrix, projMatrixStack[projectionMatrixStackSize - 1]));
            drawData.viewProjTransforms.emplace_back(hlslpp::mul(extended.invViewProjMatrix, viewProjMatrixStack[projectionMatrixStackSize - 1]));
            drawData.viewProjTransformGroups.emplace_back(extended.curViewProjMatrixIdGroupIndex);
            drawData.rspViewports.emplace_back(viewportStack[viewportStackSize - 1]);
            drawData.viewportOrigins.emplace_back(extended.viewportOriginStack[viewportStackSize - 1]);

            for (uint32_t j = 0; j < 4; j++) {
                drawData.viewportClipRatios.emplace_back(clipRatios[j]);
            }

            projectionMatrixChanged = false;
            viewportChanged = false;
        }

        projectionIndex = fbPair.changeProjection(curViewProjIndex, type);
    }

    template<uint32_t floatCount, uint32_t vertexSize>
    void RSP::readExtendedVertexSegment(uint32_t rdramAddress, uint32_t dstIndex, uint32_t dstMax, uint32_t globalIndex, uint32_t vertexElement, std::vector<float> &floatsVector) {
        if (!extended.vertexSegmentEnabled[vertexElement]) {
            return;
        }

        uint32_t vertexRdramAddress = fromSegmentedMasked(extended.vertexAddresses[vertexElement]);
        uint32_t baseRdramAddress = fromSegmentedMasked(extended.baseSegmentAddresses[vertexElement]);
        float *vertexFloats = (float *)(state->fromRDRAM(vertexRdramAddress));
        uint32_t j = globalIndex * floatCount;
        uint32_t k = ((rdramAddress - baseRdramAddress) / vertexSize) * floatCount;
        for (uint32_t i = dstIndex; i < dstMax; i++) {
            for (uint32_t f = 0; f < floatCount; f++) {
                floatsVector[j++] = vertexFloats[k++];
            }
        }
    }

    template<bool addEmptyVelocity, uint32_t vertexSize>
    void RSP::setVertexCommon(uint32_t rdramAddress, uint32_t dstIndex, uint32_t dstMax) {
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];

        if (extended.modelMatrixIdStackChanged) {
            const int stackIndex = extended.modelMatrixIdStackSize - 1;
            extended.curModelMatrixIdGroupIndex = int(workload.drawData.transformGroups.size());
            workload.drawData.transformGroups.emplace_back(extended.modelMatrixIdStack[stackIndex]);
            extended.modelMatrixIdStackChanged = false;
        }
        
        // ModelViewProj is only updated when a vertex is processed and the flag is enabled.
        auto &worldTransforms = workload.drawData.worldTransforms;
        auto &worldTransformGroups = workload.drawData.worldTransformGroups;
        auto &worldTransformSegmentedAddresses = workload.drawData.worldTransformSegmentedAddresses;
        auto &worldTransformPhysicalAddresses = workload.drawData.worldTransformPhysicalAddresses;
        auto &worldTransformVertexIndices = workload.drawData.worldTransformVertexIndices;
        bool addWorldTransform = (modelViewProjChanged || modelViewProjInserted);
        if (modelViewProjChanged) {
            computeModelViewProj();
            curTransformIndex = static_cast<uint16_t>(worldTransforms.size());
            worldTransforms.emplace_back(hlslpp::mul(modelMatrixStack[modelMatrixStackSize - 1], extended.viewProjMatrix));
        }
        else if (modelViewProjInserted) {
#       ifdef LOG_SPECIAL_MATRIX_OPERATIONS
            RT64_LOG_PRINTF("RSP::setVertexCommon(dstIndex %u, dstMax %u): An MVP was inserted and must be inversed.", dstIndex, dstMax);
#       endif

            modelViewProjInserted = false;

            if (!projectionMatrixInversed) {
                invViewProjMatrixStack[projectionMatrixStackSize - 1] = hlslpp::inverse(viewProjMatrixStack[projectionMatrixStackSize - 1]);
                projectionMatrixInversed = true;
            }

            curTransformIndex = static_cast<uint16_t>(worldTransforms.size());
            worldTransforms.emplace_back(hlslpp::mul(hlslpp::mul(modelViewProjMatrix, invViewProjMatrixStack[projectionMatrixStackSize - 1]), extended.viewProjMatrix));
        }

        if (addWorldTransform) {
            uint32_t physicalAddress = modelMatrixPhysicalAddressStack[modelMatrixStackSize - 1];
            workload.physicalAddressTransformMap.emplace(physicalAddress, uint32_t(worldTransformGroups.size()));
            worldTransformGroups.emplace_back(extended.curModelMatrixIdGroupIndex);
            worldTransformSegmentedAddresses.emplace_back(modelMatrixSegmentedAddressStack[modelMatrixStackSize - 1]);
            worldTransformPhysicalAddresses.emplace_back(physicalAddress);
            worldTransformVertexIndices.emplace_back(workload.drawData.vertexCount());
        }

        // Push a new projection if it's changed.
        if ((projectionIndex < 0) || projectionMatrixChanged || viewportChanged) {
            state->flush();
            addCurrentProjection(getCurrentProjectionType());
        }

        // We push a new set of lights if a vertex actually uses it.
        const GBI *curGBI = state->ext.interpreter->hleGBI;
        uint32_t &geometryMode = geometryModeStack[geometryModeStackSize - 1];
        const bool usesLighting = (geometryMode & G_LIGHTING);
        const bool usesPointLighting = curGBI->flags.pointLighting && (geometryMode & G_POINT_LIGHTING);
        if (usesLighting) {
            if (lightsChanged) {
                auto &rspLights = workload.drawData.rspLights;
                vertexLightIndex = static_cast<uint32_t>(rspLights.size());
                vertexLightCount = lightCount + 1;
                for (int l = 0; l < lightCount + 1; l++) {
                    const Light &light = lights[l];

                    // Decode light into easier to use parameters.
                    interop::RSPLight rspLight;
                    rspLight.col = {
                        light.dir.colr / 255.0f,
                        light.dir.colg / 255.0f,
                        light.dir.colb / 255.0f
                    };

                    rspLight.colc = {
                        light.dir.colcr / 255.0f,
                        light.dir.colcg / 255.0f,
                        light.dir.colcb / 255.0f
                    };

                    if (usesPointLighting && (light.pos.kc > 0)) {
                        rspLight.posDir = {
                            static_cast<float>(light.pos.posx),
                            static_cast<float>(light.pos.posy),
                            static_cast<float>(light.pos.posz)
                        };

                        rspLight.kc = light.pos.kc;
                        rspLight.kl = light.pos.kl;
                        rspLight.kq = light.pos.kq;
                    }
                    else {
                        rspLight.posDir = {
                             static_cast<float>(light.dir.dirx),
                             static_cast<float>(light.dir.diry),
                             static_cast<float>(light.dir.dirz)
                        };

                        rspLight.posDir = hlslpp::mul(rspLight.posDir, extended.viewProjRotationMatrix);
                        rspLight.kc = 0;
                        rspLight.kl = 0;
                        rspLight.kq = 0;
                    }

                    rspLights.push_back(rspLight);
                }

                state->updateDrawStatusAttribute(DrawAttribute::Lights);
                lightsChanged = false;
            }

            curLightIndex = vertexLightIndex;
            curLightCount = vertexLightCount;
        }
        else {
            curLightIndex = 0;
            curLightCount = 0;
        }

        const bool usesFog = (geometryMode & G_FOG);
        if (usesFog) {
            if (fogChanged) {
                auto &rspFogVector = workload.drawData.rspFog;
                vertexFogIndex = static_cast<uint32_t>(rspFogVector.size());
                rspFogVector.emplace_back(fog);
                fogChanged = false;
            }

            // Fog index is encoded as the real index + 1 to use 0 as the "fog disabled" index.
            curFogIndex = vertexFogIndex + 1;
        }
        else {
            curFogIndex = 0;
        }

        const uint32_t textureGenMask = G_LIGHTING | G_TEXTURE_GEN;
        const bool usesTextureGen = (geometryMode & textureGenMask) == textureGenMask;
        if (usesTextureGen) {
            if (lookAtChanged) {
                auto &rspLookAtVector = workload.drawData.rspLookAt;
                interop::RSPLookAt vertexLookAt = lookAt;
                vertexLookAtIndex = static_cast<uint32_t>(rspLookAtVector.size());
                vertexLookAt.x = hlslpp::mul(vertexLookAt.x, extended.viewProjRotationMatrix);
                vertexLookAt.y = hlslpp::mul(vertexLookAt.y, extended.viewProjRotationMatrix);
                rspLookAtVector.emplace_back(vertexLookAt);
                lookAtChanged = false;
            }

            // Look at index is encoded with one bit to determine if texture gen is enabled, another bit 
            // to determine if it's linear texture gen or not, and the rest of the bits to hold the real index.
            curLookAtIndex = RSP_LOOKAT_INDEX_ENABLED;
            curLookAtIndex |= (geometryMode & G_TEXTURE_GEN_LINEAR) ? RSP_LOOKAT_INDEX_LINEAR : 0x0;
            curLookAtIndex |= (vertexLookAtIndex << RSP_LOOKAT_INDEX_SHIFT);
        }
        else {
            curLookAtIndex = 0;
        }

        auto &posFloats = workload.drawData.posFloats;
        auto &velFloats = workload.drawData.velFloats;
        auto &tcFloats = workload.drawData.tcFloats;
        auto &tcVelFloats = workload.drawData.tcVelFloats;
        auto &normColBytes = workload.drawData.normColBytes;
        auto &viewProjIndices = workload.drawData.viewProjIndices;
        auto &worldIndices = workload.drawData.worldIndices;
        auto &fogIndices = workload.drawData.fogIndices;
        auto &lightIndices = workload.drawData.lightIndices;
        auto &lightCounts = workload.drawData.lightCounts;
        auto &lookAtIndices = workload.drawData.lookAtIndices;
        auto &posTransformed = workload.drawData.posTransformed;
        auto &posScreen = workload.drawData.posScreen;
        const auto &mvp = modelViewProjMatrix;
        const uint32_t globalIndex = workload.drawData.vertexCount();
        for (uint32_t i = dstIndex; i < dstMax; i++) {
            auto &v = vertices[i];
            posFloats.emplace_back(v.x);
            posFloats.emplace_back(v.y);
            posFloats.emplace_back(v.z);
            normColBytes.emplace_back(v.color.r);
            normColBytes.emplace_back(v.color.g);
            normColBytes.emplace_back(v.color.b);
            normColBytes.emplace_back(v.color.a);
            viewProjIndices.emplace_back(curViewProjIndex);
            worldIndices.emplace_back(curTransformIndex);
            fogIndices.emplace_back(curFogIndex);
            lightIndices.emplace_back(curLightIndex);
            lightCounts.emplace_back(curLightCount);
            lookAtIndices.emplace_back(curLookAtIndex);
            indices[i] = uint32_t(globalIndex) + (i - dstIndex);
            used[i] = false;

            if constexpr (addEmptyVelocity) {
                velFloats.emplace_back(0.0f);
                velFloats.emplace_back(0.0f);
                velFloats.emplace_back(0.0f);
                tcVelFloats.emplace_back(0.0f);
                tcVelFloats.emplace_back(0.0f);
            }
        }

        readExtendedVertexSegment<3, vertexSize>(rdramAddress, dstIndex, dstMax, globalIndex, G_EX_VERTEX_POSITION, posFloats);
        readExtendedVertexSegment<3, vertexSize>(rdramAddress, dstIndex, dstMax, globalIndex, G_EX_VERTEX_VELOCITY, velFloats);

        uint32_t floatIndex = globalIndex * 3;
        for (uint32_t i = dstIndex; i < dstMax; i++) {
            const hlslpp::float4 tfPos = hlslpp::mul(hlslpp::float4(posFloats[floatIndex + 0], posFloats[floatIndex + 1], posFloats[floatIndex + 2], 1.0f), mvp);
            const interop::RSPViewport &viewport = viewportStack[viewportStackSize - 1];
            posTransformed.emplace_back(tfPos);
            posScreen.emplace_back((tfPos.xyz / hlslpp::float3(tfPos.w, -tfPos.w, tfPos.w)) * viewport.scale + viewport.translate);
            floatIndex += 3;
        }

        if (usesTextureGen) {
            const float TextureSc = static_cast<float>(textureState.sc);
            const float TextureTc = static_cast<float>(textureState.tc);
            for (uint32_t i = dstIndex; i < dstMax; i++) {
                tcFloats.emplace_back(TextureSc);
                tcFloats.emplace_back(TextureTc);
            }
        }
        else {
            const int32_t TextureSc = (int32_t)(textureState.sc);
            const int32_t TextureTc = (int32_t)(textureState.tc);
            const double Divisor = 65536.0f * 32.0f;
            for (uint32_t i = dstIndex; i < dstMax; i++) {
                tcFloats.emplace_back((float)((double)((vertices[i].s) * TextureSc) / Divisor));
                tcFloats.emplace_back((float)((double)((vertices[i].t) * TextureTc) / Divisor));
            }
        }
    }

    void RSP::modifyVertex(uint16_t dstIndex, uint16_t dstAttribute, uint32_t value) {
        if (dstIndex >= RSP_MAX_VERTICES) {
            assert(false && "Vertex index is not valid. DL is possibly corrupted.");
            return;
        }

        // If the vertex was used already in the frame, then we create a new copy instead.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        auto &normColBytes = workload.drawData.normColBytes;
        auto &tcFloats = workload.drawData.tcFloats;
        auto &tcVelFloats = workload.drawData.tcVelFloats;
        auto &fogIndices = workload.drawData.fogIndices;
        auto &lightIndices = workload.drawData.lightIndices;
        auto &lightCounts = workload.drawData.lightCounts;
        auto &lookAtIndices = workload.drawData.lookAtIndices;
        auto &modifyPosUints = workload.drawData.modifyPosUints;
        auto &posScreen = workload.drawData.posScreen;
        uint32_t globalIndex = indices[dstIndex];
        if (used[dstIndex]) {
            auto &posFloats = workload.drawData.posFloats;
            auto &velFloats = workload.drawData.velFloats;
            auto &viewProjIndices = workload.drawData.viewProjIndices;
            auto &worldIndices = workload.drawData.worldIndices;
            auto &posTransformed = workload.drawData.posTransformed;
            const uint32_t newIndex = workload.drawData.vertexCount();

            // Copy every attribute into locals before appending: passing a
            // reference to a vector's own element to emplace_back is undefined
            // behavior when the append reallocates, and this path runs
            // thousands of times per frame for F3DDKR's per-triangle ST
            // rewrites. The dangling reads produced deterministic garbage
            // positions on models with many shared vertices.
            const float px = posFloats[globalIndex * 3 + 0];
            const float py = posFloats[globalIndex * 3 + 1];
            const float pz = posFloats[globalIndex * 3 + 2];
            const float vx = velFloats[globalIndex * 3 + 0];
            const float vy = velFloats[globalIndex * 3 + 1];
            const float vz = velFloats[globalIndex * 3 + 2];
            const uint8_t nc0 = normColBytes[globalIndex * 4 + 0];
            const uint8_t nc1 = normColBytes[globalIndex * 4 + 1];
            const uint8_t nc2 = normColBytes[globalIndex * 4 + 2];
            const uint8_t nc3 = normColBytes[globalIndex * 4 + 3];
            const float tcS = tcFloats[globalIndex * 2 + 0];
            const float tcT = tcFloats[globalIndex * 2 + 1];
            const float tcVelS = tcVelFloats[globalIndex * 2 + 0];
            const float tcVelT = tcVelFloats[globalIndex * 2 + 1];
            const auto viewProjIndex = viewProjIndices[globalIndex];
            const auto worldIndex = worldIndices[globalIndex];
            const auto fogIndex = fogIndices[globalIndex];
            const auto lightIndex = lightIndices[globalIndex];
            const auto lightCount = lightCounts[globalIndex];
            const auto lookAtIndex = lookAtIndices[globalIndex];
            const auto transformedPos = posTransformed[globalIndex];
            const auto screenPos = posScreen[globalIndex];

            posFloats.emplace_back(px);
            posFloats.emplace_back(py);
            posFloats.emplace_back(pz);
            velFloats.emplace_back(vx);
            velFloats.emplace_back(vy);
            velFloats.emplace_back(vz);
            normColBytes.emplace_back(nc0);
            normColBytes.emplace_back(nc1);
            normColBytes.emplace_back(nc2);
            normColBytes.emplace_back(nc3);
            tcFloats.emplace_back(tcS);
            tcFloats.emplace_back(tcT);
            tcVelFloats.emplace_back(tcVelS);
            tcVelFloats.emplace_back(tcVelT);
            viewProjIndices.emplace_back(viewProjIndex);
            worldIndices.emplace_back(worldIndex);
            fogIndices.emplace_back(fogIndex);
            lightIndices.emplace_back(lightIndex);
            lightCounts.emplace_back(lightCount);
            lookAtIndices.emplace_back(lookAtIndex);
            posTransformed.emplace_back(transformedPos);
            posScreen.emplace_back(screenPos);
            indices[dstIndex] = newIndex;
            used[dstIndex] = false;
            globalIndex = indices[dstIndex];
        }

        // Modify the attributes.
        switch (dstAttribute) {
        case G_MWO_POINT_RGBA: {
            normColBytes[globalIndex * 4 + 0] = (value >> 24) & 0xFF;
            normColBytes[globalIndex * 4 + 1] = (value >> 16) & 0xFF;
            normColBytes[globalIndex * 4 + 2] = (value >> 8) & 0xFF;
            normColBytes[globalIndex * 4 + 3] = value & 0xFF;
            fogIndices[globalIndex] = 0;
            lightIndices[globalIndex] = 0;
            lightCounts[globalIndex] = 0;
            break;
        }
        case G_MWO_POINT_ST: {
            const float s = int16_t((value >> 16) & 0xFFFF) / 32.0f;
            const float t = int16_t(value & 0xFFFF) / 32.0f;
            tcFloats[globalIndex * 2 + 0] = s;
            tcFloats[globalIndex * 2 + 1] = t;
            lookAtIndices[globalIndex] = 0;
            break;
        }
        case G_MWO_POINT_XYSCREEN: {
            // First bit being 0 indicates it should modify only XY.
            modifyPosUints.emplace_back(globalIndex << 1);
            modifyPosUints.emplace_back(value);

            // We decode on the CPU anyway for draw area tracking and the debugger.
            const uint16_t extX = (value >> 16) & 0xFFFF;
            const uint16_t extY = value & 0xFFFF;
            posScreen[globalIndex][0] = int16_t(extX) / 4.0f;
            posScreen[globalIndex][1] = int16_t(extY) / 4.0f;
            break;
        }
        case G_MWO_POINT_ZSCREEN: {
            // First bit being 1 indicates it should modify only Z.
            modifyPosUints.emplace_back((globalIndex << 1) | 0x1);
            modifyPosUints.emplace_back(value);

            // We decode on the CPU anyway for depth tracking, branchZ and the debugger.
            posScreen[globalIndex][2] = value / 65536.0f;
            break;
        }
        default:
            assert(false && "Unsupported modify vertex");
            break;
        }
    }
    
    void RSP::branchZ(uint32_t branchDl, uint16_t vtxIndex, uint32_t zValue, DisplayList **dl) {
        const bool forceBranch = state->ext.enhancementConfig->f3dex.forceBranch || extended.forceBranch;
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        const Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const uint32_t globalIndex = indices[vtxIndex];
        const float screenZ = workload.drawData.posScreen[globalIndex][2] * DepthRange;
        const float zValueFloat = zValue / 65536.0f;
        if (forceBranch || (screenZ < zValueFloat)) {
            const uint32_t rdramAddress = fromSegmentedMasked(branchDl);
            *dl = reinterpret_cast<DisplayList *>(state->fromRDRAM(rdramAddress)) - 1;
        }
    }

    void RSP::branchW(uint32_t branchDl, uint16_t vtxIndex, uint32_t wValue, DisplayList **dl) {
        const bool forceBranch = state->ext.enhancementConfig->f3dex.forceBranch || extended.forceBranch;
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        const Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const uint32_t globalIndex = indices[vtxIndex];
        const float posW = workload.drawData.posTransformed[globalIndex][3];
        if (forceBranch || (posW < static_cast<float>(wValue))) {
            const uint32_t rdramAddress = fromSegmentedMasked(branchDl);
            *dl = reinterpret_cast<DisplayList *>(state->fromRDRAM(rdramAddress)) - 1;
        }
    }

    void RSP::setGeometryMode(uint32_t mask) {
        geometryModeStack[geometryModeStackSize - 1] |= mask;
        state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
    }

    void RSP::pushGeometryMode() {
        if (geometryModeStackSize < RSP_EXTENDED_STACK_SIZE) {
            geometryModeStack[geometryModeStackSize] = geometryModeStack[geometryModeStackSize - 1];
            geometryModeStackSize++;
        }
    }

    void RSP::popGeometryMode() {
        if (geometryModeStackSize > 1) {
            geometryModeStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
        }
    }

    void RSP::clearGeometryMode(uint32_t mask) {
        geometryModeStack[geometryModeStackSize - 1] &= ~mask;
        state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
    }

    void RSP::modifyGeometryMode(uint32_t offMask, uint32_t onMask) {
        uint32_t &geometryMode = geometryModeStack[geometryModeStackSize - 1];
        const uint32_t updatedGeometryMode = (geometryMode & offMask) | onMask;
        if (updatedGeometryMode != geometryMode) {
            geometryMode = updatedGeometryMode;
            state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
        }
    }

    void RSP::setObjRenderMode(uint32_t value) {
        objRenderMode = value;
        state->updateDrawStatusAttribute(DrawAttribute::ObjRenderMode);
    }

    void RSP::setViewport(uint32_t address) {
        setViewport(address, extended.global.viewportOrigin, extended.global.viewportOffsetX, extended.global.viewportOffsetY);
    }
    
    void RSP::setViewport(uint32_t address, uint16_t ori, int16_t offx, int16_t offy) {
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const Vp_t *vp = reinterpret_cast<const Vp_t *>(state->fromRDRAM(rdramAddress));
        interop::RSPViewport &viewport = viewportStack[viewportStackSize - 1];
        viewport.scale.x = float(vp->vscale[1]) / 4.0f;
        viewport.scale.y = float(vp->vscale[0]) / 4.0f;
        viewport.scale.z = float(vp->vscale[3]) / DepthRange;
        viewport.translate.x = float(state->rdp->movedFromOrigin(vp->vtrans[1], ori) + offx) / 4.0f;
        viewport.translate.y = float(vp->vtrans[0] + offy) / 4.0f;
        viewport.translate.z = float(vp->vtrans[3]) / DepthRange;
        extended.viewportOriginStack[viewportStackSize - 1] = ori;
        static const bool vpTrace = std::getenv("RT64_VI_TRACE") != nullptr;
        if (vpTrace) {
            static float last[4] = {};
            const float now[4] = { viewport.scale.x, viewport.scale.y, viewport.translate.x, viewport.translate.y };
            if (memcmp(last, now, sizeof(now)) != 0) {
                memcpy(last, now, sizeof(now));
                fprintf(stderr, "[VP] scale %g %g translate %g %g (ori %u off %d %d)\n", now[0], now[1], now[2], now[3], ori, offx, offy);
            }
        }
        viewportChanged = true;
    }

    void RSP::pushViewport() {
        if (viewportStackSize < RSP_EXTENDED_STACK_SIZE) {
            viewportStack[viewportStackSize] = viewportStack[viewportStackSize - 1];
            extended.viewportOriginStack[viewportStackSize] = extended.viewportOriginStack[viewportStackSize - 1];
            viewportStackSize++;
        }
    }

    void RSP::popViewport() {
        if (viewportStackSize > 1) {
            viewportStackSize--;
            viewportChanged = true;
        }
    }
    
    void RSP::setLight(uint8_t index, uint32_t address) {
        assert((index >= 0) && (index <= RSP_MAX_LIGHTS));
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const uint8_t *data = reinterpret_cast<const uint8_t *>(state->fromRDRAM(rdramAddress));
        memcpy(&lights[index], data, sizeof(Light));
        lightsChanged = true;
    }

    void RSP::setLightColor(uint8_t index, uint32_t value) {
        assert((index >= 0) && (index <= RSP_MAX_LIGHTS));
        auto &rawLight = lights[index].raw;
        rawLight.words[0] = value;
        rawLight.words[1] = value;
        lightsChanged = true;
    }

    void RSP::setLightCount(uint8_t count) {
        lightCount = count;
        lightsChanged = true;
    }

    void RSP::setClipRatioEdge(uint8_t index, int16_t value) {
        assert(index < clipRatios.size());
        clipRatios[index] = value;
        viewportChanged = true;
    }

    void RSP::setClipRatioAll(int16_t value) {
        setClipRatioEdge(0, value);
        setClipRatioEdge(1, value);
        setClipRatioEdge(2, -value);
        setClipRatioEdge(3, -value);
    }

    void RSP::setPerspNorm(uint32_t perspNorm) {
        // TODO
    }

    void RSP::setLookAt(uint8_t index, uint32_t address) {
        assert(index < 2);
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const DirLight *dirLight = reinterpret_cast<const DirLight *>(state->fromRDRAM(rdramAddress));
        auto &dstLookAt = (index == 1) ? lookAt.y : lookAt.x;
        if ((dirLight->dirx != 0) || (dirLight->diry != 0) || (dirLight->dirz != 0)) {
            dstLookAt = hlslpp::normalize(hlslpp::float3(float(dirLight->dirx), float(dirLight->diry), float(dirLight->dirz)));
        }
        else {
            dstLookAt = { 0.0f, 0.0f, 0.0f };
        }

        lookAtChanged = true;
    }

    void RSP::setLookAtVectors(interop::float3 x, interop::float3 y) {
        lookAt.x = x;
        lookAt.y = y;
        lookAtChanged = true;
    }

    void RSP::setFog(int16_t mul, int16_t offset) {
        fog.mul = mul;
        fog.offset = offset;
        fogChanged = true;
    }
    
    void RSP::setTexture(uint8_t tile, uint8_t level, uint8_t on, uint16_t sc, uint16_t tc) {
        textureState.tile = tile;
        textureState.levels = level + 1;
        textureState.on = on;
        textureState.sc = sc;
        textureState.tc = tc;
    }

    void RSP::setOtherMode(uint32_t high, uint32_t low) {
        interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
        otherMode.H = high;
        otherMode.L = low;
        state->rdp->setOtherMode(otherMode.H, otherMode.L);
    }

    void RSP::pushOtherMode() {
        if (otherModeStackSize < RSP_EXTENDED_STACK_SIZE) {
            otherModeStack[otherModeStackSize] = otherModeStack[otherModeStackSize - 1];
            otherModeStackSize++;
        }
    }

    void RSP::popOtherMode() {
        if (otherModeStackSize > 1) {
            otherModeStackSize--;
            const interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
            state->rdp->setOtherMode(otherMode.H, otherMode.L);
        }
    }

    void RSP::setOtherModeL(uint32_t size, uint32_t off, uint32_t data) {
        interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
        const uint32_t mask = (((uint64_t)(1) << size) - 1) << off;
        otherMode.L = (otherMode.L & (~mask)) | data;
        state->rdp->setOtherMode(otherMode.H, otherMode.L);
    }

    void RSP::setOtherModeH(uint32_t size, uint32_t off, uint32_t data) {
        interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
        const uint32_t mask = (((uint64_t)(1) << size) - 1) << off;
        otherMode.H = (otherMode.H & (~mask)) | data;
        state->rdp->setOtherMode(otherMode.H, otherMode.L);
    }

    void RSP::setColorImage(uint8_t fmt, uint8_t siz, uint16_t width, uint32_t segAddress) {
        state->rdp->setColorImage(fmt, siz, width, fromSegmented(segAddress));
    }

    void RSP::setDepthImage(uint32_t segAddress) {
        state->rdp->setDepthImage(fromSegmented(segAddress));
    }

    void RSP::setTextureImage(uint8_t fmt, uint8_t siz, uint16_t width, uint32_t segAddress) {
        state->rdp->setTextureImage(fmt, siz, width, fromSegmented(segAddress));
    }

    void RSP::drawIndexedTri(uint32_t a, uint32_t b, uint32_t c, bool rawGlobalIndices) {
        // Copy mode is not supported when drawing regular tris and crashes the hardware.
        const uint32_t cycleType = state->rdp->otherMode.cycleType();
        assert(cycleType != G_CYC_COPY);

        // Don't draw anything if both tris are being culled.
        const uint32_t &geometryMode = geometryModeStack[geometryModeStackSize - 1];
        if ((geometryMode & cullBothMask) == cullBothMask) {
            return;
        }
        
        state->rdp->checkFramebufferPair();
        
        // We must add the current projection again if we're not in the right state.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        const Projection::Type projType = getCurrentProjectionType();
        if (!fbPair.inProjection(curViewProjIndex, projType)) {
            state->flush();
            addCurrentProjection(projType);
        }

        // Check if the texture needs to be updated.
        auto &drawCall = state->drawCall;
        if ((drawCall.textureOn != textureState.on) || (drawCall.textureTile != textureState.tile) || (drawCall.textureLevels != textureState.levels)) {
            drawCall.textureOn = textureState.on;
            drawCall.textureTile = textureState.tile;
            drawCall.textureLevels = textureState.levels;
            state->updateDrawStatusAttribute(DrawAttribute::Texture);
        }

        if (state->checkDrawState()) {
            state->loadDrawState();
        }

        const bool usesShade = geometryMode & G_SHADE;
        const bool usesLighting = geometryMode & G_LIGHTING;
        const bool usesFog = geometryMode & G_FOG;
        const bool computeSmoothNormals = !usesLighting;

        // Swap the indices around if and only if front face culling is enabled.
        if ((geometryMode & cullBothMask) == cullFrontMask) {
            uint8_t swap = c;
            c = a;
            a = swap;
        }
        
        auto &faceIndices = workload.drawData.faceIndices;
        auto &viewProjIndices = workload.drawData.viewProjIndices;
        auto &worldIndices = workload.drawData.worldIndices;
        auto &posScreen = workload.drawData.posScreen;
        auto &tcFloats = workload.drawData.tcFloats;
        auto &minMatrix = drawCall.minWorldMatrix;
        auto &maxMatrix = drawCall.maxWorldMatrix;
        uint32_t globalIndices[3];
        if (rawGlobalIndices) {
            globalIndices[0] = a;
            globalIndices[1] = b;
            globalIndices[2] = c;
        }
        else {
            globalIndices[0] = indices[a];
            globalIndices[1] = indices[b];
            globalIndices[2] = indices[c];
            used[a] = used[b] = used[c] = true;
        }

        // Indicates the vertex has been used in a tri. Whatever routines modify the vertex afterwards must use a new index instead.

        for (int i = 0; i < 3; i++) {
            // TODO: Figure out how to handle texcoord tracking on TEXGEN cases.
            const uint32_t globalIndex = globalIndices[i];
            state->rdp->updateCallTexcoords(tcFloats[globalIndex * 2 + 0], tcFloats[globalIndex * 2 + 1]);
            faceIndices.push_back(globalIndices[i]);
            minMatrix = std::min(minMatrix, worldIndices[globalIndex]);
            maxMatrix = std::max(maxMatrix, worldIndices[globalIndex]);
        }

        bool visibleTri = true;
        const bool usesCulling = geometryMode & cullBothMask;
        if (usesCulling) {
            const hlslpp::float3 U = posScreen[globalIndices[1]] - posScreen[globalIndices[0]];
            const hlslpp::float3 V = posScreen[globalIndices[2]] - posScreen[globalIndices[0]];
            const hlslpp::float3 N = hlslpp::cross(V, U);
            visibleTri = (N.z >= 0.0f);
        }

        const FixedRect &scissorRect = state->rdp->scissorRectStack[state->rdp->scissorStackSize - 1];
        if (visibleTri && !scissorRect.isNull()) {
            fbPair.scissorRect.merge(scissorRect);

            FixedRect drawRect;
            for (int i = 0; i < 3; i++) {
                const hlslpp::float3 &v = posScreen[globalIndices[i]];
                drawRect.ulx = std::min(drawRect.ulx, int32_t(v[0] * 4.0f));
                drawRect.uly = std::min(drawRect.uly, int32_t(v[1] * 4.0f));
                drawRect.lrx = std::max(drawRect.lrx, int32_t(hlslpp::ceil(v.x).x * 4.0f));
                drawRect.lry = std::max(drawRect.lry, int32_t(hlslpp::ceil(v.y).x * 4.0f));
            }

            const interop::RSPViewport &viewport = viewportStack[viewportStackSize - 1];
            const FixedRect viewportClipRect = viewport.rect(clipRatios.data());
            drawRect = viewportClipRect.intersection(scissorRect.intersection(drawRect));
            if (!drawRect.isNull()) {
                fbPair.drawColorRect.merge(drawRect);
                // JFG PORT: Jet Force Gemini sets its render modes with G_RDPSETOTHERMODE, which only reaches the RDP
                // state, so the RSP copy never had Z update on and no depth was ever written back to RDRAM. The game
                // reads its Z buffer on the CPU (mainUpdateZBCheck) to hide lens flares behind scenery; with a buffer
                // that only ever held the clear value, flares showed through everything. The RDP state is the one
                // the draw actually uses.
                if (otherModeStack[otherModeStackSize - 1].zUpd() || state->rdp->otherMode.zUpd()) {
                    fbPair.drawDepthRect.merge(drawRect);
                }
            }
        }

        drawCall.triangleCount++;
    }

    void RSP::drawIndexedTri(uint32_t a, uint32_t b, uint32_t c) {
        drawIndexedTri(a, b, c, false);
    }

    void RSP::setViewportAlign(uint16_t ori, int16_t offx, int16_t offy) {
        extended.global.viewportOrigin = ori;
        extended.global.viewportOffsetX = offx;
        extended.global.viewportOffsetY = offy;
    }

    void RSP::vertexTestZ(uint8_t vtxIndex) {
        extended.drawExtendedType = DrawExtendedType::VertexTestZ;
        extended.drawExtendedData.vertexTestZ.vertexIndex = indices[vtxIndex];
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
        drawIndexedTri(vtxIndex, vtxIndex, vtxIndex, false);
        extended.drawExtendedType = DrawExtendedType::None;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
    }

    void RSP::endVertexTestZ() {
        uint32_t vtxIndex = extended.drawExtendedData.vertexTestZ.vertexIndex;
        extended.drawExtendedType = DrawExtendedType::EndVertexTestZ;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
        drawIndexedTri(vtxIndex, vtxIndex, vtxIndex, true);
        extended.drawExtendedType = DrawExtendedType::None;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
    }

    void RSP::matrixId(uint32_t id, bool push, bool proj, bool decompose, uint8_t pos, uint8_t rot, uint8_t scale, uint8_t skew, uint8_t persp, uint8_t vpos, uint8_t vtc, uint8_t tile, uint8_t lookat, uint8_t order, uint8_t aspect, uint8_t editable, bool idIsAddress, bool editGroup) {
        assert((idIsAddress == editGroup) && "This case is not supported yet.");

        auto setGroupProperties = [=](TransformGroup* dstGroup, bool newGroup) {
            if (newGroup || (dstGroup->editable == G_EX_EDIT_ALLOW)) {
                dstGroup->decompose = decompose;
                dstGroup->positionInterpolation = pos;
                dstGroup->rotationInterpolation = rot;
                dstGroup->scaleInterpolation = scale;
                dstGroup->skewInterpolation = skew;
                dstGroup->perspectiveInterpolation = persp;
                dstGroup->vertexInterpolation = vpos;
                dstGroup->texcoordInterpolation = vtc;
                dstGroup->tileInterpolation = tile;
                dstGroup->lookAtInterpolation = lookat;
                dstGroup->ordering = order;
                dstGroup->aspectMode = aspect;
                dstGroup->editable = editable;
            }
        };

        if (idIsAddress && editGroup) {
            const uint32_t rdramAddress = fromSegmentedMasked(id);
            const int workloadCursor = state->ext.workloadQueue->writeCursor;
            Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];

            auto range = workload.physicalAddressTransformMap.equal_range(rdramAddress);
            for (auto it = range.first; it != range.second; it++) {
                uint32_t matrix_id = it->second;
                if (proj && (matrix_id < workload.drawData.viewProjTransformGroups.size())) {
                    uint32_t groupIndex = workload.drawData.viewProjTransformGroups[matrix_id];
                    setGroupProperties(&workload.drawData.transformGroups[groupIndex], false);
                }
                else if (matrix_id < workload.drawData.worldTransformGroups.size()) {
                    uint32_t groupIndex = workload.drawData.worldTransformGroups[matrix_id];
                    setGroupProperties(&workload.drawData.transformGroups[groupIndex], false);
                }
            }
        }
        else {
            auto &stack = proj ? extended.viewProjMatrixIdStack : extended.modelMatrixIdStack;
            int &stackSize = proj ? extended.viewProjMatrixIdStackSize : extended.modelMatrixIdStackSize;
            bool &stackChanged = proj ? extended.viewProjMatrixIdStackChanged : extended.modelMatrixIdStackChanged;
            if (push) {
                if (size_t(stackSize) < stack.size()) {
                    stackSize++;
                }
                else {
                    assert(false && "Stack is full.");
                }
            }

            const int stackIndex = stackSize - 1;
            TransformGroup* dstGroup = &stack[stackIndex];
            dstGroup->matrixId = id;
            setGroupProperties(dstGroup, true);
            stackChanged = true;
        }
    }

    void RSP::popMatrixId(uint8_t count, bool proj) {
        int &stackSize = proj ? extended.viewProjMatrixIdStackSize : extended.modelMatrixIdStackSize;
        bool &stackChanged = proj ? extended.viewProjMatrixIdStackChanged : extended.modelMatrixIdStackChanged;
        assert((count <= stackSize) && "Pop has requested a larger amount of matrices than the ones present in the stack.");
        while ((count > 0) && (stackSize > 1)) {
            count--;
            stackSize--;
            stackChanged = true;
        }
    }

    void RSP::forceBranch(bool force) {
        extended.forceBranch = force;
    }

    void RSP::clearExtended() {
        extended.drawExtendedType = DrawExtendedType::None;
        extended.drawExtendedData = {};
        extended.viewportOriginStack[0] = G_EX_ORIGIN_NONE;
        extended.global.viewportOrigin = G_EX_ORIGIN_NONE;
        extended.global.viewportOffsetX = 0;
        extended.global.viewportOffsetY = 0;
        extended.modelMatrixIdStack[0] = TransformGroup();
        extended.modelMatrixIdStackSize = 1;
        extended.modelMatrixIdStackChanged = false;
        extended.curModelMatrixIdGroupIndex = 0;
        extended.viewProjMatrixIdStack[0] = TransformGroup();
        extended.viewProjMatrixIdStackSize = 1;
        extended.viewProjMatrixIdStackChanged = false;
        extended.curViewProjMatrixIdGroupIndex = 0;
        extended.viewMatrix = hlslpp::float4x4::identity();
        extended.projMatrix = hlslpp::float4x4::identity();
        extended.viewProjMatrix = hlslpp::float4x4::identity();
        extended.viewProjRotationMatrix = hlslpp::float3x3::identity();
        extended.invViewMatrix = hlslpp::float4x4::identity();
        extended.invProjMatrix = hlslpp::float4x4::identity();
        extended.invViewProjMatrix = hlslpp::float4x4::identity();
        extended.vertexAddresses = {};
        extended.baseSegmentAddresses = {};
        extended.vertexSegmentEnabled = {};
        extended.forceBranch = false;
    }

    void RSP::setGBI(GBI *gbi) {
        NoN = gbi->flags.NoN;
        cullBothMask = gbi->constants[F3DENUM::G_CULL_BOTH];
        cullFrontMask = gbi->constants[F3DENUM::G_CULL_FRONT];
        projMask = gbi->constants[F3DENUM::G_MTX_PROJECTION];
        loadMask = gbi->constants[F3DENUM::G_MTX_LOAD];
        pushMask = gbi->constants[F3DENUM::G_MTX_PUSH];
        shadingSmoothMask = gbi->constants[F3DENUM::G_SHADING_SMOOTH];
    }

    void RSP::setNoN(bool NoN) {
        if (this->NoN != NoN) {
            state->flush();
            this->NoN = NoN;
        }
    }
};
