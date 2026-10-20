//
// RT64
//

#include "rt64_interpreter.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

//#define DUMP_DISPLAY_LISTS

namespace RT64 {
    static FILE *displayListFp = nullptr;

    namespace {
        constexpr uint64_t DKRWarningCommandCount = 1ULL << 16;
        constexpr uint64_t DKRMaxCommandCount = 1ULL << 20;
        constexpr uint64_t DKRWarningVertexCount = 1ULL << 18;
        constexpr uint64_t DKRMaxVertexCount = 1ULL << 22;
        constexpr uint64_t DKRWarningTriangleCount = 1ULL << 18;
        constexpr uint64_t DKRMaxTriangleCount = 1ULL << 22;
        constexpr uint64_t DKRWarningTextureBytes = 32ULL << 20;
        constexpr uint64_t DKRMaxTextureBytes = 256ULL << 20;
        constexpr uint32_t DKRWarningDMADepth = 4;
        constexpr uint32_t DKRMaxDMADepth = 64;
        constexpr uint32_t DKRWarningReturnDepth = 16;
        constexpr uint32_t DKRMaxReturnDepth = 64;

        const char *dkrAbortReasonName(DKRDisplayListDiagnostics::AbortReason reason) {
            switch (reason) {
            case DKRDisplayListDiagnostics::AbortReason::CommandBudget:
                return "command-budget";
            case DKRDisplayListDiagnostics::AbortReason::ReturnDepth:
                return "return-depth";
            case DKRDisplayListDiagnostics::AbortReason::DMADepth:
                return "dma-depth";
            case DKRDisplayListDiagnostics::AbortReason::VertexBudget:
                return "vertex-budget";
            case DKRDisplayListDiagnostics::AbortReason::TriangleBudget:
                return "triangle-budget";
            case DKRDisplayListDiagnostics::AbortReason::TextureBudget:
                return "texture-budget";
            case DKRDisplayListDiagnostics::AbortReason::MalformedDMA:
                return "malformed-dma";
            case DKRDisplayListDiagnostics::AbortReason::InvalidCommand:
                return "invalid-command";
            default:
                return "none";
            }
        }

        bool dkrValidWorkloadCursor(const State *state, int cursor) {
            return (state != nullptr) && (state->ext.workloadQueue != nullptr) &&
                (cursor >= 0) && (cursor < int(state->ext.workloadQueue->workloads.size()));
        }

        uint32_t dkrCommandAddress(const State *state, const DisplayList *dl) {
            if ((state == nullptr) || (state->RDRAM == nullptr) || (dl == nullptr)) {
                return 0xFFFFFFFFU;
            }

            const uintptr_t rdramStart = reinterpret_cast<uintptr_t>(state->RDRAM);
            const uintptr_t commandAddress = reinterpret_cast<uintptr_t>(dl);
            const uintptr_t rdramEnd = rdramStart + 0x00800000U;
            if ((commandAddress < rdramStart) || ((commandAddress + sizeof(DisplayList)) > rdramEnd)) {
                return 0xFFFFFFFFU;
            }

            return uint32_t(commandAddress - rdramStart);
        }
    }

    void DKRDisplayListDiagnostics::begin(State *state, uint32_t dlStartAddress, bool enabled) {
        *this = DKRDisplayListDiagnostics{};
        active = enabled;
        if (!active) {
            return;
        }

        rootAddress = dlStartAddress & 0x007FFFFFU;
        nextVertexReport = DKRWarningVertexCount;
        nextTriangleReport = DKRWarningTriangleCount;
        nextTextureReport = DKRWarningTextureBytes;
        const char *traceValue = std::getenv("RT64_DKR_DL_TRACE");
        traceAll = (traceValue != nullptr) && (traceValue[0] != '\0') && (traceValue[0] != '0');

        if (state->ext.workloadQueue != nullptr) {
            workloadCursorStart = state->ext.workloadQueue->writeCursor;
            if (dkrValidWorkloadCursor(state, workloadCursorStart)) {
                const DrawData &drawData = state->ext.workloadQueue->workloads[workloadCursorStart].drawData;
                workloadVerticesStart = drawData.vertexCount();
                workloadTrianglesStart = drawData.faceIndices.size() / 3;
                workloadLoadsStart = drawData.loadOperations.size();
            }
        }
    }

    bool DKRDisplayListDiagnostics::noteCommand(State *state, const DisplayList *dl, bool fromDMA) {
        if (!active || abortRequested) {
            return !abortRequested;
        }

        const uint32_t commandAddress = dkrCommandAddress(state, dl);
        if (commandAddress == 0xFFFFFFFFU) {
            if (fromDMA) {
                dmaFaultAddress = commandAddress;
                requestAbort(state, AbortReason::MalformedDMA);
            }
            else {
                requestAbort(state, AbortReason::InvalidCommand);
            }
            return false;
        }

        commandCount++;
        if (fromDMA) {
            dmaExecutedCommandCount++;
            dmaOpCodeCounts[uint8_t(dl->w0 >> 24)]++;
        }
        else {
            topLevelCommandCount++;
        }

        lastOpCode = uint8_t(dl->w0 >> 24);
        lastCommandAddress = commandAddress;

        const uint32_t returnDepth = uint32_t(state->returnAddressStack.size());
        if (returnDepth > maxReturnDepth) {
            maxReturnDepth = returnDepth;
        }

        if (maxReturnDepth > DKRMaxReturnDepth) {
            requestAbort(state, AbortReason::ReturnDepth);
            return false;
        }

        if (commandCount >= DKRMaxCommandCount) {
            requestAbort(state, AbortReason::CommandBudget);
            return false;
        }

        if ((commandCount >= DKRWarningCommandCount) && ((commandCount & (commandCount - 1)) == 0)) {
            report(state, "progress");
        }

        return true;
    }

    bool DKRDisplayListDiagnostics::enterDMA(State *state, uint32_t declaredCommandCount, uint32_t address) {
        if (!active || abortRequested) {
            return !abortRequested;
        }

        dmaCommandCount++;
        dmaDeclaredCommandCount += declaredCommandCount;
        dmaDepth++;
        if (dmaDepth > maxDMADepth) {
            maxDMADepth = dmaDepth;
        }

        if (dmaDepth > DKRMaxDMADepth) {
            dmaFaultAddress = address;
            dmaDepth--;
            requestAbort(state, AbortReason::DMADepth);
            return false;
        }

        if (dmaDepth == DKRWarningDMADepth) {
            report(state, "dma-depth-warning");
        }

        return true;
    }

    void DKRDisplayListDiagnostics::leaveDMA() {
        if (active && (dmaDepth > 0)) {
            dmaDepth--;
        }
    }

    void DKRDisplayListDiagnostics::noteUnknownDMACommand() {
        if (active) {
            unknownDMACommandCount++;
        }
    }

    void DKRDisplayListDiagnostics::noteVertexBatch(State *state, uint32_t count) {
        if (!active || abortRequested) {
            return;
        }

        vertexCommandCount++;
        vertexCount += count;
        if (vertexCount >= DKRMaxVertexCount) {
            requestAbort(state, AbortReason::VertexBudget);
            return;
        }

        if (vertexCount >= nextVertexReport) {
            report(state, "vertex-progress");
            do {
                nextVertexReport *= 2;
            } while ((nextVertexReport != 0) && (vertexCount >= nextVertexReport));
        }
    }

    void DKRDisplayListDiagnostics::noteTriangleBatch(State *state, uint32_t declaredCount, uint32_t drawnCount, uint32_t skippedCount) {
        if (!active || abortRequested) {
            return;
        }

        triangleCommandCount++;
        declaredTriangleCount += declaredCount;
        drawnTriangleCount += drawnCount;
        skippedTriangleCount += skippedCount;
        if (declaredTriangleCount >= DKRMaxTriangleCount) {
            requestAbort(state, AbortReason::TriangleBudget);
            return;
        }

        if (declaredTriangleCount >= nextTriangleReport) {
            report(state, "triangle-progress");
            do {
                nextTriangleReport *= 2;
            } while ((nextTriangleReport != 0) && (declaredTriangleCount >= nextTriangleReport));
        }
    }

    void DKRDisplayListDiagnostics::noteTextureImage(uint32_t width) {
        if (!active) {
            return;
        }

        textureImageCommandCount++;
        if (width > maxTextureWidth) {
            maxTextureWidth = width;
        }
    }

    void DKRDisplayListDiagnostics::noteLoadBlock(State *state, uint64_t bytes, bool rejected) {
        if (!active || abortRequested) {
            return;
        }

        loadBlockCommandCount++;
        if (rejected) {
            rejectedTextureLoadCount++;
            return;
        }

        textureLoadBytes += bytes;
        if (bytes > maxTextureLoadBytes) {
            maxTextureLoadBytes = bytes;
        }

        if (textureLoadBytes >= DKRMaxTextureBytes) {
            requestAbort(state, AbortReason::TextureBudget);
            return;
        }

        if (textureLoadBytes >= nextTextureReport) {
            report(state, "texture-progress");
            do {
                nextTextureReport *= 2;
            } while ((nextTextureReport != 0) && (textureLoadBytes >= nextTextureReport));
        }
    }

    void DKRDisplayListDiagnostics::abortMalformedDMA(State *state, uint32_t address) {
        if (!active || abortRequested) {
            return;
        }

        dmaFaultAddress = address;
        requestAbort(state, AbortReason::MalformedDMA);
    }

    void DKRDisplayListDiagnostics::requestAbort(State *state, AbortReason reason) {
        if (!active || abortRequested) {
            return;
        }

        abortRequested = true;
        abortReason = reason;
        report(state, "abort");
        abortReported = true;
    }

    bool DKRDisplayListDiagnostics::shouldAbort() const {
        return active && abortRequested;
    }

    void DKRDisplayListDiagnostics::finish(State *state) {
        if (!active) {
            return;
        }

        const bool suspicious = abortRequested || (unknownDMACommandCount != 0) ||
            (commandCount >= DKRWarningCommandCount) || (vertexCount >= DKRWarningVertexCount) ||
            (declaredTriangleCount >= DKRWarningTriangleCount) || (textureLoadBytes >= DKRWarningTextureBytes) ||
            (maxDMADepth >= DKRWarningDMADepth) || (maxReturnDepth >= DKRWarningReturnDepth);
        if (!abortReported && (traceAll || suspicious)) {
            report(state, abortRequested ? "aborted" : (suspicious ? "suspicious" : "complete"));
        }

        active = false;
    }

    void DKRDisplayListDiagnostics::report(State *state, const char *status) {
        if (!active) {
            return;
        }

        int workloadCursor = -1;
        uint64_t workloadVertices = 0;
        uint64_t workloadTriangles = 0;
        uint64_t workloadLoads = 0;
        bool workloadChanged = false;
        if (state->ext.workloadQueue != nullptr) {
            workloadCursor = state->ext.workloadQueue->writeCursor;
            if (dkrValidWorkloadCursor(state, workloadCursor)) {
                const DrawData &drawData = state->ext.workloadQueue->workloads[workloadCursor].drawData;
                workloadVertices = drawData.vertexCount();
                workloadTriangles = drawData.faceIndices.size() / 3;
                workloadLoads = drawData.loadOperations.size();
                workloadChanged = (workloadCursor != workloadCursorStart);
                if (!workloadChanged) {
                    workloadVertices -= (workloadVertices >= workloadVerticesStart) ? workloadVerticesStart : workloadVertices;
                    workloadTriangles -= (workloadTriangles >= workloadTrianglesStart) ? workloadTrianglesStart : workloadTriangles;
                    workloadLoads -= (workloadLoads >= workloadLoadsStart) ? workloadLoadsStart : workloadLoads;
                }
            }
        }

        std::fprintf(stderr,
            "[DKR-DL] status=%s reason=%s root=0x%08X last=0x%08X op=0x%02X "
            "commands=%llu top=%llu dma_exec=%llu dma_calls=%llu dma_declared=%llu dma_depth=%u "
            "return_depth=%u unknown_dma=%llu vertices=%llu/%llu triangles=%llu/%llu/%llu/%llu "
            "textures=%llu loads=%llu bytes=%llu max_load=%llu rejected=%llu max_width=%u "
            "workload=%d->%d switched=%u work_vertices=%llu work_triangles=%llu work_loads=%llu dma_fault=0x%08X\n",
            status, dkrAbortReasonName(abortReason), rootAddress, lastCommandAddress, lastOpCode,
            static_cast<unsigned long long>(commandCount),
            static_cast<unsigned long long>(topLevelCommandCount),
            static_cast<unsigned long long>(dmaExecutedCommandCount),
            static_cast<unsigned long long>(dmaCommandCount),
            static_cast<unsigned long long>(dmaDeclaredCommandCount), maxDMADepth, maxReturnDepth,
            static_cast<unsigned long long>(unknownDMACommandCount),
            static_cast<unsigned long long>(vertexCommandCount),
            static_cast<unsigned long long>(vertexCount),
            static_cast<unsigned long long>(triangleCommandCount),
            static_cast<unsigned long long>(declaredTriangleCount),
            static_cast<unsigned long long>(drawnTriangleCount),
            static_cast<unsigned long long>(skippedTriangleCount),
            static_cast<unsigned long long>(textureImageCommandCount),
            static_cast<unsigned long long>(loadBlockCommandCount),
            static_cast<unsigned long long>(textureLoadBytes),
            static_cast<unsigned long long>(maxTextureLoadBytes),
            static_cast<unsigned long long>(rejectedTextureLoadCount), maxTextureWidth,
            workloadCursorStart, workloadCursor, workloadChanged ? 1U : 0U,
            static_cast<unsigned long long>(workloadVertices),
            static_cast<unsigned long long>(workloadTriangles),
            static_cast<unsigned long long>(workloadLoads), dmaFaultAddress);
        std::fflush(stderr);

        if (traceAll && (dmaExecutedCommandCount != 0)) {
            std::fprintf(stderr, "[DKR-DMA] opcodes=");
            bool first = true;
            for (uint32_t opCode = 0; opCode < dmaOpCodeCounts.size(); opCode++) {
                if (dmaOpCodeCounts[opCode] != 0) {
                    std::fprintf(stderr, "%s%02X:%llu", first ? "" : ",", opCode,
                        static_cast<unsigned long long>(dmaOpCodeCounts[opCode]));
                    first = false;
                }
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
    }

    // Interpreter

    Interpreter::Interpreter() {
        state = nullptr;
        hleGBI = nullptr;
        extendedFunction = gbiManager.getExtendedFunction();
    }

    void Interpreter::setup(State *state) {
        this->state = state;
    }

    void Interpreter::loadUCodeGBI(uint32_t textAddress, uint32_t dataAddress, bool resetFromTask) {
        if (!resetFromTask) {
            state->flush();
        }

        const uint32_t AddressMask = 0xFFFFF8;
        const uint32_t maskedTextAddress = textAddress & AddressMask;
        const uint32_t maskedDataAddress = dataAddress & AddressMask;
        if ((UCode.textAddress != maskedTextAddress) || (UCode.dataAddress != maskedDataAddress)) {
            hleGBI = gbiManager.getGBIForUCode(state->RDRAM, maskedTextAddress, maskedDataAddress);
            if (hleGBI != nullptr) {
                state->rsp->setGBI(hleGBI);
            }

            UCode.textAddress = maskedTextAddress;
            UCode.dataAddress = maskedDataAddress;
        }

        if (hleGBI != nullptr) {
            GBIReset resetFunction = resetFromTask ? hleGBI->resetFromTask : hleGBI->resetFromLoad;
            if (resetFunction != nullptr) {
                resetFunction(state);
            }
        }
    }

    void Interpreter::processRDPLists(uint32_t dlStartAdddress, DisplayList *dlStart, DisplayList *dlEnd) {
        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        GBI *rdpGBI = state->rdp->gbi;
        constexpr unsigned int opCodeMask = 0x3F;

        // Run the command interpreter.
        assert(rdpGBI != nullptr);
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        uint32_t cmdLength;
        size_t pendingCommandRemainingBytes = state->rdp->pendingCommandRemainingBytes;

        if (dlStart >= dlEnd) {
            state->dlCpuProfiler.end();
            return;
        }

        if (pendingCommandRemainingBytes != 0) {
            // Copy the remaining command bytes from the current displaylist
            uint32_t toCopy = (uint32_t)std::min(pendingCommandRemainingBytes, (uintptr_t)dlEnd - (uintptr_t)dl);
            memcpy(state->rdp->pendingCommandBuffer.data() + state->rdp->pendingCommandCurrentBytes, dl, toCopy);

            // Modify start to skip the copied bytes
            dl = (DisplayList *)(toCopy + (uintptr_t)dl);

            // Check if we've copied all of the bytes of the command into the buffer
            if (pendingCommandRemainingBytes == toCopy) {
                // All bytes have been copied, so run the completed command
                DisplayList *pendingCommand = (DisplayList *)state->rdp->pendingCommandBuffer.data();
                opCode = (pendingCommand->w0 >> 24) & opCodeMask;
                func = rdpGBI->map[opCode];

                if (func != nullptr) {
                    func(state, &pendingCommand);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }

                state->rdp->pendingCommandCurrentBytes = 0;
                state->rdp->pendingCommandRemainingBytes = 0;
            }
            // Not all of the bytes were copied, so adjust RDP state accordingly and exit.
            else {
                state->rdp->pendingCommandCurrentBytes += toCopy;
                state->rdp->pendingCommandRemainingBytes -= toCopy;
                state->dlCpuProfiler.end();
                return;
            }
        }

        // Create a dummy pointer and pass that, since displaylist pointer incrementing is handled differently in LLE.
        DisplayList *dummy;
        while ((dl != nullptr) && ((dlEnd == nullptr) || (dl < dlEnd))) {
            opCode = (dl->w0 >> 24) & opCodeMask;

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                dummy = dl;
                extendedFunction(state, &dl);
                cmdLength = 1;
            }
            else {
                func = rdpGBI->map[opCode];
                cmdLength = state->rdp->commandWordLengths[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                // Check if this command is unfinished and store the partial contents if so.
                if (dl + cmdLength > dlEnd) {
                    uint32_t toCopy = (uint32_t)((uintptr_t)dlEnd - (uintptr_t)dl);
                    memcpy(state->rdp->pendingCommandBuffer.data(), dl, toCopy);
                    state->rdp->pendingCommandCurrentBytes = toCopy;
                    state->rdp->pendingCommandRemainingBytes = cmdLength * sizeof(DisplayList) - toCopy;
                    break;
                }

                if (func != nullptr) {
                    dummy = dl;
                    func(state, &dummy);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }
            }

            if (dl != nullptr) {
                dl += cmdLength;
            }
        }

        state->dlCpuProfiler.end();
    }

    void Interpreter::processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart) {
        assert(hleGBI != nullptr);

        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        dkrDiagnostics.begin(state, dlStartAdddress, hleGBI->ucode == GBIUCode::F3DDKR);

        // Every RSP task starts with an empty display-list stack on hardware.
        // A malformed or torn display list that aborted mid-run can leave
        // stale return addresses behind; carrying them over makes every
        // subsequent task trip the return-depth guard and blacks out the
        // screen permanently.
        state->returnAddressStack.clear();

        // Run the command interpreter.
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        while (dl != nullptr) {
            if (!dkrDiagnostics.noteCommand(state, dl, false)) {
                dl = nullptr;
                break;
            }

            opCode = (dl->w0 >> 24);

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                extendedFunction(state, &dl);
            }
            else {
                func = hleGBI->map[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                if (func != nullptr) {
                    func(state, &dl);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown opCode (GBI %u): %u / 0x%X", uint32_t(hleGBI->ucode), opCode, opCode);
                }
            }

            if (dkrDiagnostics.shouldAbort()) {
                dl = nullptr;
                break;
            }

            if (dl != nullptr) {
                dl++;
            }
        }

        dkrDiagnostics.finish(state);

        state->dlCpuProfiler.end();
    }
};
