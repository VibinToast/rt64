//
// RT64
//

#include "rt64_rdp_tmem.h"

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb/stb_image_write.h"
#include "xxHash/xxh3.h"

#include "common/rt64_tmem_decoder.h"
#include "common/rt64_tmem_hasher.h"

#include "rt64_state.h"

namespace RT64 {
    static void stbi_write_func_ofstream(void *context, void *data, int size) {
        ((std::ofstream *)(context))->write((const char *)(data), size);
    }

    // TextureManager

    void TextureManager::uploadEmpty(State *state, TextureCache *textureCache, uint64_t creationFrame, uint16_t width, uint16_t height, uint64_t replacementHash) {
        if (hashSet.find(replacementHash) == hashSet.end()) {
            hashSet.insert(replacementHash);
            textureCache->queueGPUUploadTMEM(replacementHash, creationFrame, nullptr, 0, width, height, 0, LoadTile(), false);
        }
    }

    uint64_t TextureManager::uploadTMEM(State *state, const LoadTile &loadTile, TextureCache *textureCache, uint64_t creationFrame, uint16_t byteOffset, uint16_t byteCount, uint16_t width, uint16_t height, uint32_t tlut) {
        XXH3_state_t xxh3;
        XXH3_64bits_reset(&xxh3);
        const uint8_t *TMEM = reinterpret_cast<const uint8_t *>(state->rdp->TMEM);
        XXH3_64bits_update(&xxh3, &TMEM[byteOffset], byteCount);
        XXH3_64bits_update(&xxh3, &byteOffset, sizeof(byteOffset));
        XXH3_64bits_update(&xxh3, &byteCount, sizeof(byteCount));
        const uint64_t hash = XXH3_64bits_digest(&xxh3);
        if (hashSet.find(hash) == hashSet.end()) {
            hashSet.insert(hash);
            textureCache->queueGPUUploadTMEM(hash, creationFrame, TMEM, RDP_TMEM_BYTES, width, height, 0, LoadTile(), false);
        }
        
        // Dump memory contents into a file if the process is active. JFG PORT: not for PNG dumps, raw TMEM is not an
        // image (it shows up as noise).
        if (!state->dumpingTexturesDirectory.empty() && !state->dumpingTexturesPng) {
            // Since width and height are not exactly guaranteed to be sane values when using raw TMEM, ensure we only dump textures when the values make sense.
            const bool validTextureCheck = (width > 0x0) && (height > 0x0);
            const bool bigTextureCheck = (width > 0x1000) || (height > 0x1000);
            if (validTextureCheck && !bigTextureCheck) {
                dumpTexture(hash, state, loadTile, width, height, tlut, creationFrame, false);
            }
        }

        return hash;
    }

    uint64_t TextureManager::uploadTexture(State *state, const LoadTile &loadTile, TextureCache *textureCache, uint64_t creationFrame, uint16_t width, uint16_t height, uint32_t tlut) {
        const uint8_t *TMEM = reinterpret_cast<const uint8_t *>(state->rdp->TMEM);
        uint64_t hash = TMEMHasher::hash(TMEM, loadTile, width, height, tlut, TMEMHasher::CurrentHashVersion);
        if (hashSet.find(hash) == hashSet.end()) {
            hashSet.insert(hash);
            textureCache->queueGPUUploadTMEM(hash, creationFrame, TMEM, RDP_TMEM_BYTES, width, height, tlut, loadTile, true);
        }

        // Dump memory contents into a file if the process is active.
        if (!state->dumpingTexturesDirectory.empty()) {
            dumpTexture(hash, state, loadTile, width, height, tlut, creationFrame, true);
        }

        return hash;
    }

    // JFG PORT: CPU version of sampleTMEM() in shaders/TextureDecoder.hlsli, so a dumped texture looks exactly the
    // way RT64 decodes it for drawing.
    namespace {
        struct TexelRGBA { uint8_t r, g, b, a; };

        uint8_t loadTMEMMasked(const uint8_t *T, uint32_t rel, uint32_t mask, uint32_t orAddress, bool oddRow, uint32_t start, uint32_t stride) {
            const uint32_t rowStart = (rel / stride) * stride;
            const uint32_t wordIndex = (rel - rowStart) / 4;
            const uint32_t finalAddress = oddRow ? (start + rowStart + ((wordIndex ^ 1) * 4) + (rel & 0x3)) : (start + rel);
            return T[((finalAddress & mask) | orAddress) & 0xFFF];
        }

        TexelRGBA expand(uint32_t r, uint32_t g, uint32_t b, uint32_t a) { return { uint8_t(r), uint8_t(g), uint8_t(b), uint8_t(a) }; }
        TexelRGBA fromI(uint32_t i) { return expand(i, i, i, i); }
        TexelRGBA fromRGBA16(uint32_t v) {
            const uint32_t r = (v >> 11) & 0x1F, g = (v >> 6) & 0x1F, b = (v >> 1) & 0x1F;
            return expand((r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2), (v & 1) ? 255 : 0);
        }
        TexelRGBA fromIA16(uint32_t v) { return expand((v >> 8) & 0xFF, (v >> 8) & 0xFF, (v >> 8) & 0xFF, v & 0xFF); }

        TexelRGBA sampleTMEMCPU(const uint8_t *T, int x, int y, uint32_t siz, uint32_t fmt, uint32_t address, uint32_t stride, uint32_t tlut, uint32_t palette) {
            const bool oddRow = (y & 1) != 0;
            const bool oddColumn = (x & 1) != 0;
            const bool isRgba32 = (fmt == G_IM_FMT_RGBA) && (siz == G_IM_SIZ_32b);
            const bool usesTlut = tlut > 0;
            const uint32_t tmemShift = isRgba32 ? 2 : siz;
            const uint32_t addressMask = (isRgba32 || usesTlut) ? 0x7FF : 0xFFF;
            const uint32_t pixelAddress = uint32_t(y) * stride + ((uint32_t(x) << tmemShift) >> 1);
            const uint32_t p0 = loadTMEMMasked(T, pixelAddress + 0, addressMask, 0, oddRow, address, stride);
            const uint32_t p1 = loadTMEMMasked(T, pixelAddress + 1, addressMask, 0, oddRow, address, stride);
            const uint32_t p4 = (p0 >> (oddColumn ? 0 : 4)) & 0xF;
            if (usesTlut) {
                const uint32_t paletteAddress = (siz == G_IM_SIZ_4b) ? (0x800 + (palette << 7) + (p4 << 3)) : (0x800 + (p0 << 3));
                const uint32_t value = T[(paletteAddress + 1) & 0xFFF] | (uint32_t(T[paletteAddress & 0xFFF]) << 8);
                if (tlut == G_TT_RGBA16) return fromRGBA16(value);
                if (tlut == G_TT_IA16) return fromIA16(value);
                return expand(0, 0, 0, 255);
            }

            const uint32_t pixelAddress2 = isRgba32 ? pixelAddress : pixelAddress + 2;
            const uint32_t orAddress = isRgba32 ? 0x800 : 0;
            const uint32_t p2 = loadTMEMMasked(T, pixelAddress2 + 0, addressMask, orAddress, oddRow, address, stride);
            const uint32_t p3 = loadTMEMMasked(T, pixelAddress2 + 1, addressMask, orAddress, oddRow, address, stride);
            switch (siz) {
            case G_IM_SIZ_4b:
                if (fmt == G_IM_FMT_CI) return fromI((palette << 4) | p4);
                if (fmt == G_IM_FMT_IA) { uint32_t i = p4 & 0xE; i = (i << 4) | (i << 1) | (i >> 2); return expand(i, i, i, (p4 & 1) ? 255 : 0); }
                if ((fmt == G_IM_FMT_I) || (fmt == G_IM_FMT_RGBA)) return fromI((p4 << 4) | p4);
                return expand(0, 0, 0, 255);
            case G_IM_SIZ_8b:
                if (fmt == G_IM_FMT_IA) { const uint32_t i = (p0 >> 4) & 0xF, a = p0 & 0xF; return expand((i << 4) | i, (i << 4) | i, (i << 4) | i, (a << 4) | a); }
                if ((fmt == G_IM_FMT_I) || (fmt == G_IM_FMT_RGBA) || (fmt == G_IM_FMT_CI)) return fromI(p0);
                return expand(0, 0, 0, 255);
            case G_IM_SIZ_16b: {
                const uint32_t v = p1 | (p0 << 8);
                if (fmt == G_IM_FMT_RGBA) return fromRGBA16(v);
                if (fmt == G_IM_FMT_IA) return fromIA16(v);
                if ((fmt == G_IM_FMT_CI) || (fmt == G_IM_FMT_I)) return expand(p0, p1, p0, p1);
                return expand(0, 0, 0, 255);
            }
            case G_IM_SIZ_32b:
                if (fmt == G_IM_FMT_RGBA) return expand(p0, p1, p2, p3);
                if ((fmt == G_IM_FMT_CI) || (fmt == G_IM_FMT_IA) || (fmt == G_IM_FMT_I)) return oddColumn ? expand(p0, p1, p0, p1) : expand(p2, p3, p2, p3);
                return expand(0, 0, 0, 255);
            default:
                return expand(0, 0, 0, 255);
            }
        }

        // A source is dynamic once it has shown more than DynamicHashLimit distinct textures and changed on at least
        // DynamicChangeFrames of the last DynamicWindowFrames frames. 12 of 30, not the 20 of 30 Toasty_Upscaler's
        // RT64_INTEGRATION.md suggests: at 60 fps the game draws a frame per tick, but enemies and tribals update only
        // every 2nd tick, so a texture they redraw per update changes on 15 of 30 frames.
        const size_t DynamicHashLimit = 16;
        const size_t DynamicChangeFrames = 12;
        const uint64_t DynamicWindowFrames = 30;
    };

    bool TextureManager::dumpSourceIsDynamic(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, uint64_t creationFrame, const std::filesystem::path &directory, DumpSource *&source) {
        const LoadOperation &loadOp = state->rdp->rice.lastLoadOpByTMEM[loadTile.tmem];
        const DumpSourceKey key(loadOp.texture.address, loadTile.fmt, loadTile.siz, width, height, tlut, loadTile.palette);
        auto inserted = dumpSources.emplace(key, DumpSource());
        DumpSource &src = inserted.first->second;
        source = &src;
        if (inserted.second) {
            src.firstFrame = creationFrame;
        }

        if (src.dynamic) {
            return true;
        }

        // Tracked before the already-dumped check: a hash seen before is still a change when it differs from the last
        // one, which is how a looping animation looks.
        if (hash != src.lastHash) {
            if ((src.lastHash != 0) && (src.changeFrames.empty() || (src.changeFrames.back() != creationFrame))) {
                src.changeFrames.push_back(creationFrame);
            }

            src.lastHash = hash;
            if (src.hashes.size() <= DynamicHashLimit) {
                src.hashes.insert(hash);
            }
        }

        while (!src.changeFrames.empty() && (src.changeFrames.front() + DynamicWindowFrames <= creationFrame)) {
            src.changeFrames.pop_front();
        }

        if ((src.hashes.size() <= DynamicHashLimit) || (src.changeFrames.size() < DynamicChangeFrames)) {
            return false;
        }

        // Flagged: move what was already written aside (not deleted, so a false positive can be undone) and log why.
        src.dynamic = true;
        const std::filesystem::path dynamicDirectory = directory / "_dynamic";
        std::error_code ec;
        std::filesystem::create_directories(dynamicDirectory, ec);
        for (const std::filesystem::path &path : src.written) {
            std::filesystem::create_directories(path.parent_path() / "_dynamic", ec);
            std::filesystem::rename(path, path.parent_path() / "_dynamic" / path.filename(), ec);
        }

        FILE *log = fopen((directory / "dynamic_sources.txt").u8string().c_str(), "ab");
        if (log != nullptr) {
            fprintf(log, "address %08x size %ux%u fmt %u siz %u tlut %u palette %u: %zu+ distinct textures, first frame %" PRIu64 ", flagged frame %" PRIu64 ", %zu files moved to _dynamic\n",
                loadOp.texture.address, width, height, loadTile.fmt, loadTile.siz, tlut, loadTile.palette, src.hashes.size(), src.firstFrame, creationFrame, src.written.size());
            fclose(log);
        }

        src.written.clear();
        return true;
    }

    // JFG PORT: a texture as RT64 decodes it for drawing, as RGBA8 (the lighting tool's texture preview); false when its
    // size isn't one that was really loaded (as dumpTexturePng checks).
    bool TextureManager::jfgDecodeTMEM(const uint8_t *TMEM, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, std::vector<uint8_t> &rgba) {
        const uint32_t stride = uint32_t(loadTile.line) << 3;
        if ((width == 0) || (height == 0) || (width > 1024) || (height > 1024) || (stride == 0)) {
            return false;
        }

        const bool halfTMEM = (tlut > 0) || ((loadTile.fmt == G_IM_FMT_RGBA) && (loadTile.siz == G_IM_SIZ_32b));
        const bool isRgba32 = (loadTile.fmt == G_IM_FMT_RGBA) && (loadTile.siz == G_IM_SIZ_32b);
        const uint32_t rowBytes = (uint32_t(width) << (isRgba32 ? 2 : loadTile.siz)) >> 1;
        if ((uint32_t(height) * stride > (halfTMEM ? 0x800U : 0x1000U)) || (rowBytes > stride)) {
            return false;
        }

        rgba.resize(size_t(width) * height * 4);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const TexelRGBA p = sampleTMEMCPU(TMEM, x, y, loadTile.siz, loadTile.fmt, uint32_t(loadTile.tmem) << 3, stride, tlut, loadTile.palette);
                uint8_t *dst = &rgba[(size_t(y) * width + x) * 4];
                dst[0] = p.r;
                dst[1] = p.g;
                dst[2] = p.b;
                dst[3] = p.a;
            }
        }

        return true;
    }

    bool TextureManager::dumpTexturePng(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, const std::filesystem::path &directory, std::filesystem::path &path) {
        const uint32_t stride = uint32_t(loadTile.line) << 3;
        if ((width == 0) || (height == 0) || (width > 1024) || (height > 1024) || (stride == 0)) {
            return false;
        }

        // A texture whose rows would not fit in TMEM cannot have been loaded whole: the size comes from the tile's wrap
        // mask (up to 256 x 256), and sampling it reads whatever else TMEM holds, which dumps as noise. RGBA32 and
        // palette textures only have half of TMEM for their texels.
        const bool halfTMEM = (tlut > 0) || ((loadTile.fmt == G_IM_FMT_RGBA) && (loadTile.siz == G_IM_SIZ_32b));
        if (uint32_t(height) * stride > (halfTMEM ? 0x800U : 0x1000U)) {
            return false;
        }

        // Likewise a row wider than the tile's line (rows would overlap): the width is a wrap size, not loaded data.
        const bool isRgba32 = (loadTile.fmt == G_IM_FMT_RGBA) && (loadTile.siz == G_IM_SIZ_32b);
        const uint32_t rowBytes = (uint32_t(width) << (isRgba32 ? 2 : loadTile.siz)) >> 1;
        if (rowBytes > stride) {
            static int logged = 0;
            if (logged++ < 8) fprintf(stderr, "[JFG] texture dump: skipped %ux%u fmt %u siz %u line %u (%u bytes per row)\n", width, height, loadTile.fmt, loadTile.siz, loadTile.line, rowBytes);
            return false;
        }

        const uint8_t *T = reinterpret_cast<const uint8_t *>(state->rdp->TMEM);
        std::vector<TexelRGBA> pixels(size_t(width) * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixels[size_t(y) * width + x] = sampleTMEMCPU(T, x, y, loadTile.siz, loadTile.fmt, uint32_t(loadTile.tmem) << 3, stride, tlut, loadTile.palette);
            }
        }

        // Character shadows: the game renders each character's silhouette into a 64x64 texture every frame (black,
        // with the soft edge only in the alpha), so every pose is a new texture that a replacement would never match
        // again. Not dumped: 64x64 with every visible texel black. (Glows and other white or grey alpha masks of the
        // same size are kept.)
        if ((width == 64) && (height == 64)) {
            int visible = 0;
            bool shadow = true;
            for (const TexelRGBA &p : pixels) {
                if (p.a == 0) continue;
                if ((p.r > 8) || (p.g > 8) || (p.b > 8)) { shadow = false; break; }
                visible++;
            }
            if (shadow && (visible > 0)) {
                static int skipped = 0;
                if (++skipped <= 3) fprintf(stderr, "[JFG] texture dump: skipped character shadow %016" PRIx64 "\n", hash);
                return false;
            }
        }

        char name[64];
        snprintf(name, sizeof(name), "%016" PRIx64 ".png", hash);
        path = directory / name;
        const std::string pathString = path.u8string();
        if (stbi_write_png(pathString.c_str(), width, height, 4, pixels.data(), int(width) * 4) == 0) {
            return false;
        }

        // texture_info.txt: what never changes for this hash, one line as its PNG is written (Toasty_Upscaler's
        // RT64_INTEGRATION.md, "Per texture"): the format the PNG was decoded with, the palette, and the RDRAM
        // addresses it was loaded from.
        static const char *const FmtNames[] = { "RGBA", "YUV", "CI", "IA", "I" };
        const char *fmtName = (loadTile.fmt < 5) ? FmtNames[loadTile.fmt] : "RGBA";
        const char *tlutName = (tlut == G_TT_RGBA16) ? "rgba16" : ((tlut == G_TT_IA16) ? "ia16" : "none");
        const bool CI4 = (loadTile.fmt == G_IM_FMT_CI) && (loadTile.siz == G_IM_SIZ_4b);
        const uint32_t src = state->rdp->rice.lastLoadOpByTMEM[loadTile.tmem].texture.address;
        FILE *info = fopen((directory / "texture_info.txt").u8string().c_str(), "ab");
        if (info != nullptr) {
            fprintf(info, "tex %016" PRIx64 " fmt %s siz %u tlut %s pal %u line %u src %08x", hash, fmtName, 4U << loadTile.siz, tlutName, CI4 ? uint32_t(loadTile.palette) : 0U, uint32_t(loadTile.line), src);
            if ((loadTile.fmt == G_IM_FMT_CI) && (tlut > 0)) {
                const uint32_t paletteTMEM = (RDP_TMEM_WORDS >> 1) + (CI4 ? (uint32_t(loadTile.palette) << 4) : 0);
                fprintf(info, " tlutsrc %08x", state->rdp->rice.lastLoadOpByTMEM[paletteTMEM].texture.address);
            }

            fprintf(info, "\n");
            fclose(info);
        }

        return true;
    }

    // JFG PORT: where a texture's PNG goes: the folder the host gives for the RDRAM address it was loaded from (a mod's
    // own texture, so a texture pack made from the dump keeps the game's textures apart from each mod's), else the dump
    // folder. Each folder has its own _dynamic, texture_info.txt and dynamic_sources.txt.
    std::filesystem::path TextureManager::dumpDirectory(State *state, const LoadTile &loadTile) {
        if (!state->dumpingTexturesPng || !state->dumpingTexturesDirectoryFor) {
            return state->dumpingTexturesDirectory;
        }

        const uint32_t src = state->rdp->rice.lastLoadOpByTMEM[loadTile.tmem].texture.address;
        const std::filesystem::path directory = state->dumpingTexturesDirectoryFor(src);
        if (directory.empty()) {
            return state->dumpingTexturesDirectory;
        }

        if (dumpScannedDirectories.insert(directory).second) {
            std::error_code ec;
            std::filesystem::create_directories(directory, ec);
            scanDumped(directory);
        }

        return directory;
    }

    void TextureManager::resetDump(const std::filesystem::path &directory) {
        dumpedSet.clear();
        dumpSources.clear();
        dumpScannedDirectories.clear();
        if (!directory.empty()) {
            scanDumped(directory);
        }
    }

    // JFG PORT: a dump continued in a folder that already holds PNGs (a new session) does not write those textures, or
    // their texture_info.txt lines, again. Their names start with the hash, in the folder or _dynamic/.
    void TextureManager::scanDumped(const std::filesystem::path &directory) {
        std::error_code ec;
        for (const std::filesystem::path &folder : { directory, directory / "_dynamic" }) {
            for (std::filesystem::directory_iterator it(folder, ec), end; !ec && (it != end); it.increment(ec)) {
                const std::string name = it->path().filename().u8string();
                if ((name.size() >= 20) && (it->path().extension() == ".png") && (name.find_first_not_of("0123456789abcdef") == 16)) {
                    dumpedSet.insert(std::strtoull(name.substr(0, 16).c_str(), nullptr, 16));
                }
            }
        }
    }

    void TextureManager::dumpTexture(uint64_t hash, State *state, const LoadTile &loadTile, uint16_t width, uint16_t height, uint32_t tlut, uint64_t creationFrame, bool savePNG) {
        // JFG PORT: textures the game redraws on most frames are not dumped (see dumpSourceIsDynamic). Only for PNG dumps.
        DumpSource *source = nullptr;
        const std::filesystem::path directory = dumpDirectory(state, loadTile);
        if (state->dumpingTexturesPng && state->dumpingTexturesSkipDynamic && dumpSourceIsDynamic(hash, state, loadTile, width, height, tlut, creationFrame, directory, source)) {
            return;
        }

        if (dumpedSet.find(hash) != dumpedSet.end()) {
            return;
        }

        // Insert into set regardless of whether the dump is successful or not.
        dumpedSet.insert(hash);

        // JFG PORT: PNG dumping for texture packs. The file name is the hash RT64 looks replacements up by, which is
        // what the "rt64" auto path of a replacement database matches on.
        if (state->dumpingTexturesPng) {
            std::filesystem::path path;
            if (dumpTexturePng(hash, state, loadTile, width, height, tlut, directory, path) && (source != nullptr)) {
                source->written.push_back(path);
            }

            return;
        }
        
        // Dump the entirety of TMEM.
        char baseName[64];
        snprintf(baseName, sizeof(baseName), "%016" PRIx64 ".v%u", hash, TMEMHasher::CurrentHashVersion);
        std::filesystem::path dumpTmemPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".tmem");
        std::ofstream dumpTmemStream(dumpTmemPath, std::ios::binary);
        if (dumpTmemStream.is_open()) {
            const char *TMEM = reinterpret_cast<const char *>(state->rdp->TMEM);
            dumpTmemStream.write(TMEM, RDP_TMEM_BYTES);
            dumpTmemStream.close();
        }

        // Dump as a PNG if it was specified.
        if (savePNG) {
            std::filesystem::path dumpPngPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".png");
            std::ofstream dumpPngStream(dumpPngPath, std::ios::binary);
            if (dumpPngStream.is_open()) {
                TMEMDecoder::decodeToRGBA32(reinterpret_cast<const uint8_t *>(state->rdp->TMEM), loadTile, width, height, tlut, dumpRGBAPixels);
                stbi_write_png_to_func(&stbi_write_func_ofstream, &dumpPngStream, width, height, 4, dumpRGBAPixels.data(), width * sizeof(uint32_t));
                dumpPngStream.close();
            }
        }

        // Dump the RDRAM last loaded into the TMEM address pointed to by the tile. Required for generating hashes used by Rice.
        const LoadOperation &loadOp = state->rdp->rice.lastLoadOpByTMEM[loadTile.tmem];
        uint32_t rdramStart = loadOp.texture.address;
        uint32_t rdramCount = 0;
        uint32_t commonBytesOffset = (loadOp.tile.uls >> 2) << loadOp.texture.siz >> 1;
        uint32_t commonBytesPerRow = loadOp.texture.width << loadOp.texture.siz >> 1;
        if (loadOp.type == LoadOperation::Type::Block) {
            uint32_t wordCount = ((loadOp.tile.lrs - loadOp.tile.uls) >> (4 - loadOp.tile.siz)) + 1;
            rdramStart = loadOp.texture.address + commonBytesOffset + commonBytesPerRow * loadOp.tile.ult;
            rdramCount = (wordCount << 3);

            // Increase the amount of RDRAM dumped by textures that require padding when using load block.
            commonBytesPerRow = std::max(commonBytesPerRow, uint32_t(loadTile.line) << 3U);
        }
        else if (loadOp.type == LoadOperation::Type::Tile) {
            uint32_t rowCount = 1 + ((loadOp.tile.lrt >> 2) - (loadOp.tile.ult >> 2));
            uint32_t tileWidth = ((loadOp.tile.lrs >> 2) - (loadOp.tile.uls >> 2));
            uint32_t wordsPerRow = (tileWidth >> (4 - loadOp.tile.siz)) + 1;
            rdramStart = loadOp.texture.address + commonBytesOffset + commonBytesPerRow * (loadOp.tile.ult >> 2);
            rdramCount = rowCount * commonBytesPerRow;
        }
        
        // Dump more RDRAM if necessary if it doesn't cover what the tile could possibly sample.
        uint32_t loadTileBpr = width << loadTile.siz >> 1;
        rdramCount = std::max(rdramCount, std::max(loadTileBpr, commonBytesPerRow) * height);

        if (rdramCount > 0) {
            std::filesystem::path dumpRdramPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.rdram");
            std::ofstream dumpRdramStream(dumpRdramPath, std::ios::binary);
            if (dumpRdramStream.is_open()) {
                const char *RDRAM = reinterpret_cast<const char *>(state->RDRAM);
                dumpRdramStream.write(&RDRAM[rdramStart], rdramCount);
                dumpRdramStream.close();
            }

            std::filesystem::path dumpRdramInfoPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.json");
            std::ofstream dumpRdramInfoStream(dumpRdramInfoPath);
            if (dumpRdramInfoStream.is_open()) {
                json jroot;
                jroot["tile"] = loadOp.tile;
                jroot["type"] = loadOp.type;
                jroot["texture"] = loadOp.texture;
                dumpRdramInfoStream << std::setw(4) << jroot << std::endl;
                dumpRdramInfoStream.close();
            }
        }
        
        // Repeat a similar process for dumping the palette.
        if (tlut > 0) {
            const bool CI4 = (loadTile.siz == G_IM_SIZ_4b);
            const int32_t paletteTMEM = (RDP_TMEM_WORDS >> 1) + (CI4 ? (loadTile.palette << 4) : 0);
            const LoadOperation &paletteLoadOp = state->rdp->rice.lastLoadOpByTMEM[paletteTMEM];
            uint32_t paletteBytesOffset = (paletteLoadOp.tile.uls >> 2) << paletteLoadOp.texture.siz >> 1;
            uint32_t paletteBytesPerRow = paletteLoadOp.texture.width << paletteLoadOp.texture.siz >> 1;
            const uint32_t rowCount = 1 + ((paletteLoadOp.tile.lrt >> 2) - (paletteLoadOp.tile.ult >> 2));
            const uint32_t wordsPerRow = ((paletteLoadOp.tile.lrs >> 2) - (paletteLoadOp.tile.uls >> 2)) + 1;
            uint32_t paletteRdramStart = paletteLoadOp.texture.address + paletteBytesOffset + paletteBytesPerRow * (paletteLoadOp.tile.ult >> 2);
            uint32_t paletteRdramCount = (rowCount - 1) * paletteBytesPerRow + (wordsPerRow << 3);
            if (paletteRdramCount > 0) {
                std::filesystem::path dumpPaletteRdramPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.palette.rdram");
                std::ofstream dumpPaletteRdramStream(dumpPaletteRdramPath, std::ios::binary);
                if (dumpPaletteRdramStream.is_open()) {
                    const char *RDRAM = reinterpret_cast<const char *>(state->RDRAM);
                    dumpPaletteRdramStream.write(&RDRAM[paletteRdramStart], paletteRdramCount);
                    dumpPaletteRdramStream.close();
                }
            }

            std::filesystem::path dumpPaletteRdramInfoPath = state->dumpingTexturesDirectory / (std::string(baseName) + ".rice.palette.json");
            std::ofstream dumpPaletteRdramInfoStream(dumpPaletteRdramInfoPath);
            if (dumpPaletteRdramInfoStream.is_open()) {
                json jroot;
                jroot["tile"] = paletteLoadOp.tile;
                jroot["type"] = paletteLoadOp.type;
                jroot["texture"] = paletteLoadOp.texture;
                dumpPaletteRdramInfoStream << std::setw(4) << jroot << std::endl;
                dumpPaletteRdramInfoStream.close();
            }
        }

        // Dump the parameters of the tile into a JSON file.
        std::filesystem::path dumpTilePath = state->dumpingTexturesDirectory / (std::string(baseName) + ".tile.json");
        std::ofstream dumpTileStream(dumpTilePath);
        if (dumpTileStream.is_open()) {
            json jroot;
            jroot["tile"] = loadTile;
            jroot["width"] = width;
            jroot["height"] = height;

            // Serialize the TLUT into an enum instead.
            if (tlut == G_TT_RGBA16) {
                jroot["tlut"] = LoadTLUT::RGBA16;
            }
            else if (tlut == G_TT_IA16) {
                jroot["tlut"] = LoadTLUT::IA16;
            }
            else {
                jroot["tlut"] = LoadTLUT::None;
            }

            dumpTileStream << std::setw(4) << jroot << std::endl;
            dumpTileStream.close();
        }
    }

    void TextureManager::removeHashes(const std::vector<uint64_t> &hashes) {
        for (uint64_t hash : hashes) {
            hashSet.erase(hash);
        }
    }
};