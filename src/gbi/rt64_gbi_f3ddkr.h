//
// RT64 - Diddy Kong Racing's custom Fast3D microcode
//

#pragma once

#include "rt64_gbi.h"

#define F3DDKR_G_MTX          0x01
#define F3DDKR_G_TEX_OFFSET   0x02
#define F3DDKR_G_VTX          0x04
#define F3DDKR_G_TRIN         0x05
#define F3DDKR_G_DMADL        0x07
#define F3DDKR_G_SETPERSPNORM 0xB5
#define F3DDKR_G_MOVEWORD     0xBC
#define F3DDKR_G_DMA_OFFSETS  0xBF

namespace RT64 {
    namespace GBI_F3DDKR {
        void setup(GBI *gbi);
    };
};
