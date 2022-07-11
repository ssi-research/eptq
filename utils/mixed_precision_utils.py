from enum import Enum


class MPCONFIG(Enum):
    MP_PARTIAL_CANDIDATES = 0
    MP_FULL_CANDIDATES = 0


MP_BITWIDTH_OPTIONS_DICT = {
    MPCONFIG.MP_PARTIAL_CANDIDATES: [2, 4, 8],
    MPCONFIG.MP_FULL_CANDIDATES: [2, 3, 4, 5, 6, 7, 8]
}


