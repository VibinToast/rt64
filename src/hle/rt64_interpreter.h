//
// RT64
//

#pragma once

#include "rt64_state.h"

#include "gbi/rt64_f3d.h"
#include "gbi/rt64_gbi.h"

namespace RT64 {
    struct DKRDisplayListDiagnostics {
        enum class AbortReason {
            None,
            CommandBudget,
            ReturnDepth,
            DMADepth,
            VertexBudget,
            TriangleBudget,
            TextureBudget,
            MalformedDMA,
            InvalidCommand
        };

        bool active = false;
        bool traceAll = false;
        bool abortRequested = false;
        bool abortReported = false;
        AbortReason abortReason = AbortReason::None;
        uint32_t rootAddress = 0;
        uint32_t lastCommandAddress = 0xFFFFFFFFU;
        uint32_t dmaFaultAddress = 0xFFFFFFFFU;
        uint8_t lastOpCode = 0;
        uint64_t commandCount = 0;
        uint64_t topLevelCommandCount = 0;
        uint64_t dmaCommandCount = 0;
        uint64_t dmaDeclaredCommandCount = 0;
        uint64_t dmaExecutedCommandCount = 0;
        uint64_t unknownDMACommandCount = 0;
        uint64_t vertexCommandCount = 0;
        uint64_t vertexCount = 0;
        uint64_t triangleCommandCount = 0;
        uint64_t declaredTriangleCount = 0;
        uint64_t drawnTriangleCount = 0;
        uint64_t skippedTriangleCount = 0;
        uint64_t textureImageCommandCount = 0;
        uint64_t loadBlockCommandCount = 0;
        uint64_t textureLoadBytes = 0;
        uint64_t maxTextureLoadBytes = 0;
        uint64_t rejectedTextureLoadCount = 0;
        uint32_t maxTextureWidth = 0;
        uint32_t dmaDepth = 0;
        uint32_t maxDMADepth = 0;
        uint32_t maxReturnDepth = 0;
        int workloadCursorStart = -1;
        uint64_t workloadVerticesStart = 0;
        uint64_t workloadTrianglesStart = 0;
        uint64_t workloadLoadsStart = 0;
        uint64_t nextVertexReport = 0;
        uint64_t nextTriangleReport = 0;
        uint64_t nextTextureReport = 0;
        std::array<uint64_t, 256> dmaOpCodeCounts{};

        void begin(State *state, uint32_t dlStartAddress, bool enabled);
        bool noteCommand(State *state, const DisplayList *dl, bool fromDMA);
        bool enterDMA(State *state, uint32_t declaredCommandCount, uint32_t address);
        void leaveDMA();
        void noteUnknownDMACommand();
        void noteVertexBatch(State *state, uint32_t count);
        void noteTriangleBatch(State *state, uint32_t declaredCount, uint32_t drawnCount, uint32_t skippedCount);
        void noteTextureImage(uint32_t width);
        void noteLoadBlock(State *state, uint64_t bytes, bool rejected);
        void abortMalformedDMA(State *state, uint32_t address);
        void requestAbort(State *state, AbortReason reason);
        bool shouldAbort() const;
        void finish(State *state);
        void report(State *state, const char *status);
    };

    struct Interpreter {
        State *state;
        GBIManager gbiManager;
        GBI *hleGBI;
        DKRDisplayListDiagnostics dkrDiagnostics;
        uint8_t extendedOpCode = 0;
        GBIFunction extendedFunction = nullptr;

        struct {
            uint32_t textAddress = 0;
            uint32_t dataAddress = 0;
        } UCode;

        Interpreter();
        void setup(State *state);
        void loadUCodeGBI(uint32_t textAddress, uint32_t dataAddress, bool resetFromTask);
        void processRDPLists(uint32_t dlStartAdddress, DisplayList* dlStart, DisplayList* dlEnd);
        void processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart);
    };
};
