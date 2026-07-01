# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...


class Action:
    """Action constants for red_mahjong (87 actions).

    Discard from hand:    0 ~ 36   (tile index 0-36, including red fives)
    Closed/Added Kan:    37 ~ 70   (tile index x 2 + closed(0)/added(1))
    Special actions:     71 ~ 86
    """

    # Discard range (0..36 maps to 37 tile types with red)
    # Closed/Added Kan: 37..70  (34 tile_types * 2, + extra for red fives)
    TSUMOGIRI: int = 71
    RIICHI: int = 72
    TSUMO: int = 73
    RON: int = 74
    PON: int = 75
    PON_RED: int = 76
    OPEN_KAN: int = 77
    CHI_L: int = 78          # [4]56
    CHI_L_RED: int = 79      # [4]5r6
    CHI_M: int = 80          # 4[5r]6
    CHI_M_RED: int = 81      # 4[5r]6
    CHI_R: int = 82          # 45[6]
    CHI_R_RED: int = 83      # 45[r6]
    PASS: int = 84
    KYUUSHU: int = 85
    DUMMY: int = 86          # info-sharing phase after round end
    NUM_ACTION: int = 87

    @staticmethod
    def is_selfkan(action: int) -> bool:
        return (37 <= action) & (action < 71)
