//
// RT64 - Diddy Kong Racing's custom Fast3D microcode
//

#include "rt64_gbi_f3ddkr.h"

#include "hle/rt64_interpreter.h"
#include "hle/rt64_rdp.h"
#include "rt64_gbi_f3d.h"
#include "rt64_gbi_extended.h"

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

            auto &texture = state->rsp->textureState;
            state->rsp->setTexture(texture.tile, texture.levels, textureOn, texture.sc, texture.tc);

            uint32_t drawnCount = 0;
            uint32_t skippedCount = 0;
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
                    : ((state->rsp->viewportStack[state->rsp->viewportStackSize - 1].scale.x > 0.0f)
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
