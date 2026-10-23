//
// RT64
//

#pragma once

#include <deque>
#include <filesystem>
#include <map>
#include <set>
#include <tuple>

#include "hle/rt64_draw_call.h"
#include "render/rt64_texture_cache.h"

namespace RT64 {
    struct State;

    struct TextureManager {
        // JFG PORT: one source of dumped textures, an RDRAM address read through one tile shape. A source whose texture
        // changes on most frames (an effect the game redraws every frame) is marked dynamic and no longer dumped, since
        // its textures can never be matched by a replacement.
        typedef std::tuple<uint32_t, uint8_t, uint8_t, uint16_t, uint16_t, uint32_t, uint8_t> DumpSourceKey; // address, fmt, siz, width, height, tlut, palette
        struct DumpSource {
            uint64_t lastHash = 0;
            uint64_t firstFrame = 0;
            std::set<uint64_t> hashes;
            std::deque<uint64_t> changeFrames;
            std::vector<std::filesystem::path> written;
            bool dynamic = false;
        };

        std::set<uint64_t> hashSet;
        std::set<uint64_t> dumpedSet;
        std::vector<uint32_t> dumpRGBAPixels;
        std::map<DumpSourceKey, DumpSource> dumpSources; // JFG PORT

        void uploadEmpty(State *state, TextureCache *textureCache, uint64_t creationFrame, uint16_t width, uint16_t height, uint64_t replacementHash);
        uint64_t uploadTMEM(State *state, const LoadTile &loadTile, TextureCache *textureCache, uint64_t creationFrame, uint16_t byteOffset, uint16_t byteCount, uint16_t width, uint16_t height, uint32_t tlut);
        uint64_t uploadTexture(State *state, const LoadTile &loadTile, TextureCache *textureCache, uint64_t creationFrame, uint16_t width, uint16_t height, uint32_t tlut);
        void dumpTexture(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, uint64_t creationFrame, bool savePNG);
        bool dumpSourceIsDynamic(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, uint64_t creationFrame, DumpSource *&source); // JFG PORT
        bool dumpTexturePng(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, std::filesystem::path &path); // JFG PORT
        void resetDump(const std::filesystem::path &directory); // JFG PORT
        static bool jfgDecodeTMEM(const uint8_t *TMEM, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, std::vector<uint8_t> &rgba); // JFG PORT
        void removeHashes(const std::vector<uint64_t> &hashes);
    };
};