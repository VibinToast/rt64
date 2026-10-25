//
// RT64 - Diddy Kong Racing's custom Fast3D microcode
//

#include "rt64_gbi_f3ddkr.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "hle/rt64_interpreter.h"
#include "hle/rt64_rdp.h"
#include "rt64_gbi_f3d.h"
#include "rt64_gbi_extended.h"

// JFG PORT: the character shadows' triangle batches of the last two game frames (the game double-buffers them) and how
// far back each one's image slides on in-between frames: the shadow is drawn on fixed copies of the ground triangles,
// and only their texture coordinates follow the character once per game frame, so without this it stepped ahead of
// the interpolated character. Registered by the game thread (src/jfg/jfg_shadow_interp.cpp), read by the RSP.
namespace {
    struct JfgShadowBatch { uint32_t start, end; float dUls, dUlt; };
    std::mutex jfgShadowMutex;
    std::vector<JfgShadowBatch> jfgShadowBatches;
}

// [start, end): physical address range of the batch's triangles; dUls/dUlt: its tile origin one game frame back.
void rt64_jfg_shadow_batch(uint32_t start, uint32_t end, float dUls, float dUlt) {
    std::lock_guard<std::mutex> lock(jfgShadowMutex);
    // A batch written over an older one's place (the buffer two game frames back) replaces it.
    for (size_t i = 0; i < jfgShadowBatches.size(); ) {
        const JfgShadowBatch &b = jfgShadowBatches[i];
        if ((b.start < end) && (start < b.end)) {
            jfgShadowBatches[i] = jfgShadowBatches.back();
            jfgShadowBatches.pop_back();
        }
        else {
            i++;
        }
    }
    if (jfgShadowBatches.size() < 256) {
        jfgShadowBatches.push_back({ start, end, dUls, dUlt });
    }
}

static bool jfgShadowFind(uint32_t address, float &dUls, float &dUlt) {
    static const bool disabled = std::getenv("RT64_JFG_NO_SHADOW_SLIDE") != nullptr; // kill switch: shadows step as before
    if (disabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(jfgShadowMutex);
    for (const JfgShadowBatch &b : jfgShadowBatches) {
        if ((address >= b.start) && (address < b.end)) {
            dUls = b.dUls;
            dUlt = b.dUlt;
            return true;
        }
    }
    return false;
}

namespace RT64 {
    namespace GBI_F3DDKR {
        static uint8_t readU8(State *state, uint32_t address) {
            return *state->fromRDRAM(address ^ 0x3);
        }

        static int16_t readS16(State *state, uint32_t address) {
            return *reinterpret_cast<const int16_t *>(state->fromRDRAM(address ^ 0x2));
        }

        void noOp(State *, DisplayList **) {
        }

        void textureOffset(State *state, DisplayList **dl) {
            state->rsp->setDMATextureOffsetDKR((*dl)->w1);
        }

        void matrix(State *state, DisplayList **dl) {
            if ((*dl)->p0(0, 16) != 64) {
                return;
            }

            state->rsp->matrixDKR((*dl)->w1, (*dl)->p0(22, 2));
        }

        void vertex(State *state, DisplayList **dl) {
            // F3DDKR G_VTX: param byte holds (n-1)<<3 | (address & 6) | append.
            // Bits 9-15 of w0 belong to the DMA length field and are never read
            // by the microcode; the vertex count comes only from bits 19-23.
            const bool append = ((*dl)->w0 & 0x00010000U) != 0;
            const uint32_t count = (*dl)->p0(19, 5) + 1;
            const uint32_t sourceOffset = ((*dl)->w0 >> 16) & 0x6U;

            state->ext.interpreter->dkrDiagnostics.noteVertexBatch(state, count);
            if (state->ext.interpreter->dkrDiagnostics.shouldAbort()) {
                return;
            }

            state->rsp->setVertexDKR((*dl)->w1, count, sourceOffset, append);
        }

        void triangles(State *state, DisplayList **dl) {
            const uint32_t count = (*dl)->p0(20, 4) + 1;
            const uint8_t textureOn = (*dl)->p0(16, 4);
            const uint32_t address = state->rsp->fromSegmentedMasked((*dl)->w1);

            // JFG PORT: a character shadow's batch starts a draw call of its own, whose tiles remember how far the
            // image slides (State::loadDrawState); so does the first batch after one.
            if (state->rsp->DKR.jfg) {
                float dUls = 0.0f, dUlt = 0.0f;
                const bool shadow = jfgShadowFind(address, dUls, dUlt);
                auto &dkr = state->rsp->DKR;
                if ((shadow != dkr.jfgShadow) || (shadow && ((dUls != dkr.jfgShadowUls) || (dUlt != dkr.jfgShadowUlt)))) {
                    dkr.jfgShadow = shadow;
                    dkr.jfgShadowUls = dUls;
                    dkr.jfgShadowUlt = dUlt;
                    state->updateDrawStatusAttribute(DrawAttribute::Texture);
                }
            }

            auto &texture = state->rsp->textureState;
            // JFG PORT: setTexture() stores level + 1, so passing the stored value back raised the level count by one
            // for every triangle batch. A draw call using LOD then claimed up to 255 tiles and overran fixed 8-entry
            // arrays (stack corruption in State::fullSync).
            state->rsp->setTexture(texture.tile, (texture.levels > 0) ? uint8_t(texture.levels - 1) : 0, textureOn, texture.sc, texture.tc);

            uint32_t drawnCount = 0;
            uint32_t skippedCount = 0;
            // debug aids: batch index within the current display list; RT64_SKIP_TRIS="lo,hi" skips batches lo..hi-1.
            static uint32_t dbgLastRoot = 0, dbgIndex = 0;
            {
                const uint32_t root = state->ext.interpreter->dkrDiagnostics.rootAddress;
                if (root != dbgLastRoot) { dbgLastRoot = root; dbgIndex = 0; }
                else dbgIndex++;
                static const char *sk = std::getenv("RT64_SKIP_TRIS");
                if (sk != nullptr) {
                    static int lo = 0, hi = 0, parsed = 0;
                    if (!parsed) { sscanf(sk, "%d,%d", &lo, &hi); parsed = 1; }
                    if (int(dbgIndex) >= lo && int(dbgIndex) < hi) {
                        static uint32_t printed = 0;
                        if (printed++ % 97 == 0) {
                            const auto &t0 = state->rdp->tiles[state->rsp->textureState.tile & 7];
                            const uint8_t vi = readU8(state, address + 1);
                            const auto &v = state->rsp->vertices[vi < RSP_MAX_VERTICES ? vi : 0];
                            fprintf(stderr, "[SKIP] #%u root=%06X slot=%u bb=%d n=%u tris@%06X timg=%06X fmt=%u siz=%u %ux%u v=(%d,%d,%d) om=%08X:%08X\n", dbgIndex, root, uint32_t(state->rsp->DKR.activeMatrix), int(state->rsp->DKR.billboard), count, address, state->rdp->texture.address, t0.fmt, t0.siz, (t0.lrs >> 2) + 1, (t0.lrt >> 2) + 1, v.x, v.y, v.z, uint32_t(state->rdp->otherMode.H), uint32_t(state->rdp->otherMode.L));
                        }
                        return;
                    }
                }
            }
            static const bool triTrace = std::getenv("RT64_TRI_TRACE") != nullptr; // debug aid
            if (triTrace) {
                const auto &m = state->rsp->DKR.matrices[state->rsp->DKR.activeMatrix];
                float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, w0 = 1e9f, w1 = -1e9f;
                uint32_t col = 0;
                for (uint32_t i = 0; i < count; i++) {
                    for (uint32_t k = 1; k <= 3; k++) {
                        const uint8_t vi = readU8(state, address + i * 16 + k);
                        if (vi >= RSP_MAX_VERTICES) continue;
                        const auto &v = state->rsp->vertices[vi];
                        const hlslpp::float4 c = hlslpp::mul(hlslpp::float4(float(v.x), float(v.y), float(v.z), 1.0f), m);
                        const float w = c[3];
                        w0 = std::min(w0, w); w1 = std::max(w1, w);
                        if (w > 0.01f) { const float nx = float(c[0]) / w, ny = float(c[1]) / w; x0 = std::min(x0, nx); x1 = std::max(x1, nx); y0 = std::min(y0, ny); y1 = std::max(y1, ny); }
                        col = (uint32_t(v.color.r) << 24) | (uint32_t(v.color.g) << 16) | (uint32_t(v.color.b) << 8) | v.color.a;
                    }
                }
                fprintf(stderr, "[TRI] #%u slot=%u bb=%d tex=%u n=%u x[%.2f,%.2f] y[%.2f,%.2f] w[%.1f,%.1f] col=%08X tris@%08X om=%08X:%08X geom=%08X\n", dbgIndex, uint32_t(state->rsp->DKR.activeMatrix), int(state->rsp->DKR.billboard), uint32_t(textureOn), count, x0, x1, y0, y1, w0, w1, col, address, uint32_t(state->rdp->otherMode.H), uint32_t(state->rdp->otherMode.L), uint32_t(state->rsp->geometryModeStack[state->rsp->geometryModeStackSize - 1]));
                {
                    const auto &t0 = state->rdp->tiles[state->rsp->textureState.tile & 7];
                    const auto &t7 = state->rdp->tiles[7];
                    fprintf(stderr, "[TEX] timg=%06X tile%u fmt=%u siz=%u line=%u tmem=%u pal=%u cm=%u/%u mask=%u/%u size %u,%u-%u,%u | load fmt=%u siz=%u line=%u size %u,%u-%u,%u | texOff=%06X cnt=%u shift=%u sc=%04X tc=%04X\n", state->rdp->texture.address, uint32_t(state->rsp->textureState.tile), t0.fmt, t0.siz, t0.line, t0.tmem, t0.palette, t0.cms, t0.cmt, t0.masks, t0.maskt, t0.uls, t0.ult, t0.lrs, t0.lrt, t7.fmt, t7.siz, t7.line, t7.uls, t7.ult, t7.lrs, t7.lrt, state->rsp->DKR.textureOffset, state->rsp->DKR.textureCount, state->rsp->DKR.textureShift, uint32_t(state->rsp->textureState.sc), uint32_t(state->rsp->textureState.tc));
                    if (state->rsp->DKR.billboard) {
                        const auto &v0 = state->rsp->vertices[0];
                        const auto &mm = state->rsp->DKR.matrices[state->rsp->DKR.activeMatrix];
                        const auto &m0 = state->rsp->DKR.matrices[0];
                        const hlslpp::float4 c0 = hlslpp::mul(hlslpp::float4(float(v0.x), float(v0.y), float(v0.z), 1.0f), m0);
                        fprintf(stderr, "[ANC] v0=(%d,%d,%d) rgba=%02X%02X%02X%02X viaM0=(%g,%g,%g,%g) active diag=(%g,%g) tr=(%g,%g,%g,%g)\n", v0.x, v0.y, v0.z, v0.color.r, v0.color.g, v0.color.b, v0.color.a,
                            float(c0[0]), float(c0[1]), float(c0[2]), float(c0[3]), float(mm[0][0]), float(mm[1][1]), float(mm[3][0]), float(mm[3][1]), float(mm[3][2]), float(mm[3][3]));
                    }
                    if (state->rsp->DKR.billboard && t0.siz == 2) {
                        const uint32_t w = (t0.lrs >> 2) + 1, h = (t0.lrt >> 2) + 1, base = state->rdp->texture.address & 0x7FFFFF;
                        uint32_t opaque = 0, nonzero = 0;
                        for (uint32_t i = 0; i < w * h && i < 4096; i++) {
                            const uint16_t v = *reinterpret_cast<const uint16_t *>(state->fromRDRAM((base + i * 2) ^ 2));
                            opaque += v & 1; nonzero += (v != 0);
                        }
                        fprintf(stderr, "[TXD] %ux%u opaque=%u nonzero=%u prim=%g,%g,%g,%g env=%g,%g,%g,%g cc=%08X:%08X\n", w, h, opaque, nonzero,
                            float(state->rdp->primColorStack[state->rdp->primColorStackSize - 1][0]), float(state->rdp->primColorStack[state->rdp->primColorStackSize - 1][1]), float(state->rdp->primColorStack[state->rdp->primColorStackSize - 1][2]), float(state->rdp->primColorStack[state->rdp->primColorStackSize - 1][3]),
                            float(state->rdp->envColorStack[state->rdp->envColorStackSize - 1][0]), float(state->rdp->envColorStack[state->rdp->envColorStackSize - 1][1]), float(state->rdp->envColorStack[state->rdp->envColorStackSize - 1][2]), float(state->rdp->envColorStack[state->rdp->envColorStackSize - 1][3]),
                            uint32_t(state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1].H), uint32_t(state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1].L));
                    }
                }
            }
            for (uint32_t i = 0; i < count; i++) {
                const uint32_t triAddress = address + i * 16;
                const uint8_t flags = readU8(state, triAddress + 0);
                const uint8_t v0 = readU8(state, triAddress + 1);
                const uint8_t v1 = readU8(state, triAddress + 2);
                const uint8_t v2 = readU8(state, triAddress + 3);

                // F3DDKR keeps its own vertex cursor and only the vertices
                // loaded by the current DMA batch are valid. Drawing an
                // unloaded slot reuses stale RT64 index data and can turn a
                // small menu primitive into a screen-spanning polygon.
                const auto &loadedVertices = state->rsp->DKR.loadedVertices;
                if ((v0 >= loadedVertices.size()) || (v1 >= loadedVertices.size()) ||
                    (v2 >= loadedVertices.size()) || !loadedVertices.test(v0) ||
                    !loadedVertices.test(v1) || !loadedVertices.test(v2)) {
                    skippedCount++;
                    continue;
                }

                const uint32_t cullMode = (flags & 0x40U)
                    ? 0
                    : (((state->rsp->viewportStack[state->rsp->viewportStackSize - 1].scale.x > 0.0f) != (state->rsp->DKR.jfg && state->rsp->DKR.jfgCullFlip))
                        ? (state->rsp->cullBothMask ^ state->rsp->cullFrontMask)
                        : state->rsp->cullFrontMask);
                state->rsp->modifyGeometryMode(~state->rsp->cullBothMask, cullMode);

                const uint32_t st0 = (uint32_t(uint16_t(readS16(state, triAddress + 4))) << 16)
                    | uint16_t(readS16(state, triAddress + 6));
                const uint32_t st1 = (uint32_t(uint16_t(readS16(state, triAddress + 8))) << 16)
                    | uint16_t(readS16(state, triAddress + 10));
                const uint32_t st2 = (uint32_t(uint16_t(readS16(state, triAddress + 12))) << 16)
                    | uint16_t(readS16(state, triAddress + 14));

                state->rsp->modifyVertex(v0, G_MWO_POINT_ST, st0);
                state->rsp->modifyVertex(v1, G_MWO_POINT_ST, st1);
                state->rsp->modifyVertex(v2, G_MWO_POINT_ST, st2);
                state->rsp->drawIndexedTri(v0, v1, v2);
                drawnCount++;
            }

            state->ext.interpreter->dkrDiagnostics.noteTriangleBatch(state, count, drawnCount, skippedCount);
            // The microcode's vertex cursor (DMEM 0x14C) is only written by
            // non-append vertex loads; G_TRIN does not touch it. Sprites rely
            // on this: consecutive append loads reuse slots 1..n across
            // triangle batches while slot 0 keeps the anchor vertex.
        }

        void dmaDisplayList(State *state, DisplayList **dl) {
            const uint32_t commandCount = (*dl)->p0(16, 8);
            const uint32_t address = state->rsp->fromSegmentedMasked((*dl)->w1);
            auto &diagnostics = state->ext.interpreter->dkrDiagnostics;
            if (!diagnostics.enterDMA(state, commandCount, address)) {
                return;
            }

            struct DMADepthGuard {
                DKRDisplayListDiagnostics &diagnostics;
                ~DMADepthGuard() {
                    diagnostics.leaveDMA();
                }
            } depthGuard{ diagnostics };

            DisplayList *command = reinterpret_cast<DisplayList *>(state->fromRDRAM(address));
            uint32_t consumed = 0;

            while (consumed < commandCount) {
                if (!diagnostics.noteCommand(state, command, true)) {
                    break;
                }

                // G_DMADL is not a nested Fast3D display list. F3DDKR DMAs a
                // fixed number of 64-bit words and sends them directly to the
                // RDP. Dispatching those words through the Fast3D map is
                // dangerous: a non-RDP byte such as 0x06 would be mistaken for
                // G_DL, corrupt the display-list return stack, and make the
                // interpreter run into unrelated RDRAM.
                DisplayList *executed = command;
                const uint8_t opCode = uint8_t(command->w0 >> 24);
                if ((opCode & 0xC0U) == 0xC0U) {
                    GBIFunction function = state->ext.interpreter->hleGBI->map[opCode];
                    if (function != nullptr) {
                        function(state, &executed);
                    }
                    else {
                        diagnostics.noteUnknownDMACommand();
                    }
                }
                else {
                    // The RDP ignores words outside its command range. Keep
                    // consuming the counted DMA payload without giving them
                    // Fast3D control-flow semantics.
                    diagnostics.noteUnknownDMACommand();
                }

                if (diagnostics.shouldAbort()) {
                    break;
                }

                const uintptr_t commandPointer = reinterpret_cast<uintptr_t>(command);
                const uintptr_t executedPointer = reinterpret_cast<uintptr_t>(executed);
                if ((executedPointer < commandPointer) ||
                    (((executedPointer - commandPointer) % sizeof(DisplayList)) != 0)) {
                    diagnostics.abortMalformedDMA(state, address + consumed * sizeof(DisplayList));
                    break;
                }

                const uint64_t length = ((executedPointer - commandPointer) / sizeof(DisplayList)) + 1;
                if (length > (commandCount - consumed)) {
                    diagnostics.abortMalformedDMA(state, address + consumed * sizeof(DisplayList));
                    break;
                }

                command = executed + 1;
                consumed += uint32_t(length);
            }
        }

        void setPerspNorm(State *state, DisplayList **dl) {
            state->rsp->setPerspNorm((*dl)->p1(16, 16));
        }

        void dmaOffsets(State *state, DisplayList **dl) {
            state->rsp->setDMAOffsetsDKR((*dl)->p0(0, 24), (*dl)->p1(0, 24));
        }

        void setTextureImage(State *state, DisplayList **dl) {
            const uint8_t fmt = (*dl)->p0(21, 3);
            const uint8_t siz = (*dl)->p0(19, 2);
            const uint16_t width = (*dl)->p0(0, 12) + 1;
            const uint32_t address = state->rsp->textureImageAddressDKR(fmt, (*dl)->w1);
            state->ext.interpreter->dkrDiagnostics.noteTextureImage(width);
            state->rdp->setTextureImage(fmt, siz, width, address);
        }

        void loadBlock(State *state, DisplayList **dl) {
            const uint8_t tile = (*dl)->p1(24, 3);
            const uint16_t uls = (*dl)->p0(12, 12);
            const uint16_t ult = (*dl)->p0(0, 12);
            const uint16_t lrs = (*dl)->p1(12, 12);
            const uint16_t dxt = (*dl)->p1(0, 12);
            const bool rejected = (uls > lrs) || (lrs >= 0x800);
            uint64_t loadBytes = 0;
            if (!rejected) {
                const uint8_t size = state->rdp->tiles[tile].siz;
                const uint32_t wordCount = ((lrs - uls) >> (4 - size)) + 1;
                loadBytes = uint64_t(wordCount) << 3;
            }

            state->ext.interpreter->dkrDiagnostics.noteLoadBlock(state, loadBytes, rejected);
            if (state->ext.interpreter->dkrDiagnostics.shouldAbort()) {
                return;
            }

            state->rsp->loadBlockDKR(lrs);
            state->rdp->loadBlock(tile, uls, ult, lrs, dxt);
        }

        void moveWord(State *state, DisplayList **dl) {
            switch ((*dl)->p0(0, 8)) {
            case 0x02:
                state->rsp->setBillboardDKR(((*dl)->w1 & 1U) != 0);
                break;
            case 0x0A:
                state->rsp->selectMatrixDKR((*dl)->p1(6, 2));
                break;
            default:
                GBI_F3D::moveWord(state, dl);
                break;
            }
        }

        void reset(State *state) {
            GBI_F3D::reset(state);
            // DKR is an 8 MB N64 title. Recompiled game pointers can retain
            // KSEG0/KSEG1 bits, which must wrap through the RSP DMA mask
            // instead of being interpreted as RT64 extended-RDRAM offsets.
            state->setExtendedRDRAM(false);
            state->rsp->DKR.force8MBAddressMask = true;
            state->rsp->DKR.segmentedAddresses.fill(0);
            state->rsp->DKR.physicalAddresses.fill(0);
            state->rsp->DKR.vertexCursor = 0;
            state->rsp->DKR.loadedVertices.reset();
            state->rsp->DKR.billboard = false;
        }

        // ---- JFG PORT: Jet Force Gemini's revision of the microcode -------------------------------------------
        // Differences from DKR's, taken from a disassembly of the microcode in the US ROM (0x9FFD0):
        //  * G_MTX: parameter byte selects destination/source slots (8 slots) and load vs multiply.
        //  * G_VTX: count is bits 19-23 as is (not count - 1); the destination slot is bits 9-15 of w0;
        //           there is no vertex cursor.
        //  * G_MW_MTXOFFSET: the active matrix index has 3 bits.
        //  * G_DMAOFFSETS also ends the per-texture DMA offset list.
        static bool traceJFG() {
            static const bool enabled = std::getenv("RT64_JFG_TRACE") != nullptr;
            return enabled;
        }

        void matrixJFG(State *state, DisplayList **dl) {
            state->rsp->matrixJFG((*dl)->w1, uint8_t((*dl)->p0(16, 8)));
            if (traceJFG()) {
                {
                    const uint32_t raw = state->rsp->DKR.physicalAddresses[(*dl)->p0(16, 3)];
                    fprintf(stderr, "[JFG-GFX] MTX raw@%08X (mtxOff=%X vtxOff=%X):", raw, state->rsp->DKR.matrixOffset, state->rsp->DKR.vertexOffset);
                    for (uint32_t k = 0; k < 0x40; k += 2) fprintf(stderr, " %04X", uint16_t(readS16(state, raw + k)));
                    fprintf(stderr, "\n");
                }
                const auto &m = state->rsp->DKR.matrices[(*dl)->p0(16, 3)];
                fprintf(stderr, "[JFG-GFX] MTX w0=%08X w1=%08X  [%g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g]\n", (*dl)->w0, (*dl)->w1,
                    float(m[0][0]), float(m[0][1]), float(m[0][2]), float(m[0][3]), float(m[1][0]), float(m[1][1]), float(m[1][2]), float(m[1][3]),
                    float(m[2][0]), float(m[2][1]), float(m[2][2]), float(m[2][3]), float(m[3][0]), float(m[3][1]), float(m[3][2]), float(m[3][3]));
            }
        }

        void vertexJFG(State *state, DisplayList **dl) {
            const uint32_t count = (*dl)->p0(19, 5);
            const uint32_t sourceOffset = ((*dl)->w0 >> 16) & 0x6U;
            const uint32_t dstIndex = (*dl)->p0(9, 7);
            if (count == 0) {
                return;
            }

            state->ext.interpreter->dkrDiagnostics.noteVertexBatch(state, count);
            if (state->ext.interpreter->dkrDiagnostics.shouldAbort()) {
                return;
            }

            state->rsp->setVertexDKR((*dl)->w1, count, sourceOffset, false, int32_t(dstIndex));
            if (traceJFG()) {
                const auto &v = state->rsp->vertices[dstIndex];
                fprintf(stderr, "[JFG-GFX] VTX w0=%08X w1=%08X n=%u dst=%u first=(%d %d %d) rgba=%02X%02X%02X%02X billboard=%d\n", (*dl)->w0, (*dl)->w1,
                    count, dstIndex, v.x, v.y, v.z, v.color.r, v.color.g, v.color.b, v.color.a, int(state->rsp->DKR.billboard));
            }
        }

        void moveWordJFG(State *state, DisplayList **dl) {
            if (traceJFG()) {
                fprintf(stderr, "[JFG-GFX] MW  w0=%08X w1=%08X\n", (*dl)->w0, (*dl)->w1);
            }
            if ((*dl)->p0(0, 8) == 0x0A) {
                state->rsp->selectMatrixDKR((*dl)->p1(6, 3));
            }
            else {
                moveWord(state, dl);
            }
        }

        void dmaOffsetsJFG(State *state, DisplayList **dl) {
            if (traceJFG()) {
                fprintf(stderr, "[JFG-GFX] DMAOFFSETS w0=%08X w1=%08X\n", (*dl)->w0, (*dl)->w1);
            }
            dmaOffsets(state, dl);
            state->rsp->DKR.textureOffset = 0;
            state->rsp->DKR.textureShift = 0;
            state->rsp->DKR.textureCount = 0;
        }

        // The microcode only rewrites the cull-back bit (0x2000) for each triangle. Bit 0x1000 of the geometry
        // mode is left to the game, and when it is set the rasteriser culls the other side instead. The game sets
        // it while it draws objects mirrored in a reflective floor, whose triangles are wound the other way round.
        void setGeometryModeJFG(State *state, DisplayList **dl) {
            if ((*dl)->w1 & 0x1000U) {
                state->rsp->DKR.jfgCullFlip = true;
            }
            GBI_F3D::setGeometryMode(state, dl);
        }

        void clearGeometryModeJFG(State *state, DisplayList **dl) {
            if ((*dl)->w1 & 0x1000U) {
                state->rsp->DKR.jfgCullFlip = false;
            }
            GBI_F3D::clearGeometryMode(state, dl);
        }

        void resetJFG(State *state) {
            reset(state);
            state->rsp->DKR.jfg = true;
            state->rsp->DKR.jfgCullFlip = false;
            // JFG PORT: not DKR's 8 MB mask: JFG's addresses go through jfgPhysicalAddress (rt64_state.h), so mods can
            // draw from their own memory (0x81000000 and up).
            state->rsp->DKR.force8MBAddressMask = false;
        }

        void setupJFG(GBI *gbi) {
            setup(gbi);
            gbi->map[F3DDKR_G_MTX] = &matrixJFG;
            gbi->map[F3DDKR_G_VTX] = &vertexJFG;
            gbi->map[F3DDKR_G_MOVEWORD] = &moveWordJFG;
            gbi->map[F3DDKR_G_DMA_OFFSETS] = &dmaOffsetsJFG;
            gbi->map[F3D_G_SETGEOMETRYMODE] = &setGeometryModeJFG;
            gbi->map[F3D_G_CLEARGEOMETRYMODE] = &clearGeometryModeJFG;
            gbi->resetFromTask = &resetJFG;
        }

        void setup(GBI *gbi) {
            GBI_F3D::setup(gbi);

            gbi->map[F3DDKR_G_MTX] = &matrix;
            gbi->map[F3DDKR_G_TEX_OFFSET] = &textureOffset;
            gbi->map[F3DDKR_G_VTX] = &vertex;
            gbi->map[F3DDKR_G_TRIN] = &triangles;
            gbi->map[F3DDKR_G_DMADL] = &dmaDisplayList;
            gbi->map[F3DDKR_G_SETPERSPNORM] = &setPerspNorm;
            gbi->map[F3D_G_CULLDL] = &noOp;
            gbi->map[F3D_G_POPMTX] = &noOp;
            gbi->map[F3DDKR_G_MOVEWORD] = &moveWord;
            gbi->map[F3DDKR_G_DMA_OFFSETS] = &dmaOffsets;
            gbi->map[G_SETTIMG] = &setTextureImage;
            gbi->map[G_LOADBLOCK] = &loadBlock;
            gbi->resetFromTask = &reset;
        }
    };
};
