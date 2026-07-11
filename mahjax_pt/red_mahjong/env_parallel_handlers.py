# Copyright 2025 The Mahjax Authors.
# Action handlers for RedMahjongParallel — the "what happens" layer.
#
# Mixed into RedMahjongParallel via HandlersMixin.
# Each handler processes one action type across all B environments simultaneously.

from typing import Optional
import torch

from .action import Action
from .constants import (
    FIRST_DRAW_IDX, MAX_DISCARDS_PER_PLAYER, NUM_PLAYERS,
    NUM_TILE_TYPES, NUM_TILE_TYPES_WITH_RED,
    DEAD_WALL_TILES, LEGAL_ACTION_SIZE, STARTING_POINTS,
    RIICHI_BET, MAX_MELDS_PER_PLAYER, MAX_DORA_INDICATORS,
    SENTINEL_DISCARD_VALUE,
)
from .meld import Meld, EMPTY_MELD
from .tile import River, Tile, EMPTY_RIVER
from .hand import Hand
from .yaku import Yaku
from .shanten import Shanten
from .batch_state import BatchState, unstack_state
from .env_serial import _is_first_turn, _set_tile_type_action


class HandlersMixin:
    """Mixin for RedMahjongParallel: all 11 action handlers + draw helpers.

    These methods process game actions (discard, pon, chi, kan, ron, tsumo,
    riichi, pass, kyuushu, dummy) across all B environments at once using
    boolean masks and tensor indexing.
    """

    def _riichi_batch(self, bs: BatchState, mask: torch.Tensor):
        """Riichi declaration. Sets riichi_declared and builds discard-ok mask.

        VECTORIZED: Uses batched Hand.sub + to_34 + Shanten.number_batch
        for all M envs per tile type (37 iterations, each batching M envs).
        """
        if not mask.any():
            return bs
        B = bs.B
        device = bs.players.hand.device
        b_idx = torch.arange(B, device=device)
        cps = bs.current_player[mask]  # (M,)
        m_idx = b_idx[mask]  # (M,)

        hands_37 = bs.players.hand_with_red[m_idx, cps]  # (M, 37)
        last_draw = bs.round_state.last_draw[m_idx]  # (M,)

        M = m_idx.shape[0]
        discard_ok = torch.zeros(M, Tile.NUM_TILE_TYPE_WITH_RED, dtype=torch.bool, device=device)

        # Batched per-tile-type: check tenpai after discarding each tile type
        for t in range(Tile.NUM_TILE_TYPE_WITH_RED):
            has_tile = hands_37[:, t] > 0  # (M,)
            if not has_tile.any():
                continue
            h_idx = has_tile.nonzero(as_tuple=False).squeeze(-1)  # (K,)
            K_t = h_idx.shape[0]
            # Sub tile t from hand (batch) — need tensor for add_batch
            t_tensor = torch.full((K_t,), t, dtype=torch.int32, device=device)
            sub_hands = Hand.sub_batch(hands_37[h_idx], t_tensor)  # (K, 37)
            sub_34 = Hand.to_34_batch(sub_hands)  # (K, 34)
            is_tenpai = Shanten.number_batch(sub_34) <= 0  # (K,)
            discard_ok[h_idx, t] = is_tenpai

        # Build masks vectorized
        masks = torch.zeros(M, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device)
        masks[:, :Tile.NUM_TILE_TYPE_WITH_RED] = discard_ok

        # Handle last_draw special case (vectorized)
        ld_valid = (last_draw >= 0) & (last_draw < Tile.NUM_TILE_TYPE_WITH_RED)  # (M,)
        if ld_valid.any():
            ld_idx = ld_valid.nonzero(as_tuple=False).squeeze(-1)  # (L,)
            ld_val = last_draw[ld_idx].long().clamp(0, 36)  # (L,)
            masks[ld_idx, ld_val] = (hands_37[ld_idx, ld_val] >= 2) & discard_ok[ld_idx, ld_val]
            masks[ld_idx, Action.TSUMOGIRI] = discard_ok[ld_idx, ld_val]

        # Write to batch state (vectorized)
        bs.legal_action_mask[m_idx] = masks
        bs.players.riichi_declared[m_idx, cps] = True
        bs.round_state.draw_next[m_idx] = False

        return bs

    # ═════════════════════════════════════════════════════════════
    # _accept_riichi_batch — vectorized riichi acceptance
    # ═════════════════════════════════════════════════════════════

    def _accept_riichi_batch(self, bs: BatchState, mask: torch.Tensor):
        """Accept pending riichi for masked envs. Batch version of _accept_riichi.

        Processes the riichi declaration from last_player (the player who declared
        riichi on the previous step). Called at the beginning of _draw / meld handlers.
        """
        if not mask.any():
            return bs
        device = bs.players.hand.device
        m_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)

        lp = bs.round_state.last_player[m_idx]  # (M,) — last player in each env

        # Check riichi status for last players
        riichi_status = bs.players.riichi[m_idx, lp]         # (M,)
        declared_status = bs.players.riichi_declared[m_idx, lp]  # (M,)

        # only process envs where last_player is NOT already in riichi
        need_process = ~riichi_status  # (M,)

        if need_process.any():
            p_idx = m_idx[need_process]        # (K,) env indices
            p_lp = lp[need_process]             # (K,) last players

            # Zero rewards for these envs (all 4 players)
            bs.rewards[p_idx] = 0.0

            # Envs where riichi was actually declared
            has_decl = declared_status[need_process]  # (K,)
            if has_decl.any():
                d_idx = p_idx[has_decl]  # (L,) env indices that declared riichi
                d_lp = p_lp[has_decl]    # (L,) players who declared

                # Pay riichi bet
                bs.round_state.score[d_idx, d_lp] -= RIICHI_BET // 100
                bs.rewards[d_idx, d_lp] = -10.0
                bs.round_state.kyotaku[d_idx] += 1

                # Set riichi flags
                bs.players.riichi[d_idx, d_lp] = True
                bs.players.riichi_declared[d_idx, d_lp] = False
                bs.players.ippatsu[d_idx, d_lp] = True

                # Double riichi: is_first_turn AND no melds from anyone
                nxt = bs.round_state.next_deck_ix[d_idx]          # (L,)
                is_first = nxt >= FIRST_DRAW_IDX - 4              # (L,)
                no_melds = bs.players.meld_counts[d_idx].sum(dim=1) == 0  # (L,)
                bs.players.double_riichi[d_idx, d_lp] = is_first & no_melds

        return bs

    # ═════════════════════════════════════════════════════════════
    # _ron_batch — FULLY VECTORIZED
    # ═════════════════════════════════════════════════════════════

    def _ron_batch(self, bs: BatchState, mask: torch.Tensor):
        """Ron (win by discard) — batch implementation."""
        if not mask.any():
            return bs
        B = bs.B
        device = bs.players.hand.device
        H_idx = torch.arange(B, device=device)[mask]  # (H,)
        H = H_idx.shape[0]

        cps_H = bs.current_player[H_idx]  # (H,)
        discarded_H = bs.round_state.last_player[H_idx]  # (H,)
        targets_H = bs.round_state.target[H_idx]  # (H,)

        # ── 1. Yaku judgement (batch) ──
        hands_H = bs.players.hand_with_red[H_idx, cps_H]  # (H, 37)
        melds_H = bs.players.melds[H_idx, cps_H]  # (H, MAX_MELDS)
        n_meld_H = bs.players.meld_counts[H_idx, cps_H]  # (H,)
        riichi_H = bs.players.riichi[H_idx, cps_H]  # (H,)
        seat_winds_H = bs.round_state.seat_wind[H_idx, cps_H]  # (H,)
        prevalent_winds_H = bs.round_state.round[H_idx] // 4  # (H,)
        dora_H = bs.round_state.dora_indicators[H_idx]  # (H, 5)
        ura_H = bs.round_state.ura_dora_indicators[H_idx]  # (H, 5)

        yaku_H, fan_H, fu_H = Yaku.judge_hand_related_batch(
            hands_H, melds_H, n_meld_H, targets_H, riichi_H,
            torch.ones(H, dtype=torch.bool, device=device),  # is_ron=True
            prevalent_winds_H, seat_winds_H, dora_H, ura_H)

        # ── 2. Adjust fan for special conditions ──
        ippatsu_H = bs.players.ippatsu[H_idx, cps_H] & riichi_H  # (H,)
        double_riichi_H = bs.players.double_riichi[H_idx, cps_H]  # (H,)
        robbing_kan_H = bs.round_state.kan_declared[H_idx]  # (H,)
        houtei_H = bs.round_state.is_haitei[H_idx] & ~robbing_kan_H  # (H,)
        is_yakuman_H = fu_H == 0  # (H,)

        fan_H = fan_H.to(torch.int32)
        extra_fan = (ippatsu_H.to(torch.int32) + double_riichi_H.to(torch.int32) +
                     robbing_kan_H.to(torch.int32) + houtei_H.to(torch.int32))
        fan_H = torch.where(is_yakuman_H, fan_H, fan_H + extra_fan)

        # ── 3. Settle payments (batch) ──
        bs = self._settle_ron_batch(bs, H_idx, cps_H, discarded_H, fan_H, fu_H)

        # ── 4. Kyotaku bonus ──
        kyotaku_H = bs.round_state.kyotaku[H_idx]  # (H,)
        kyotaku_bonus = kyotaku_H.to(torch.float32) * 10.0
        bs.rewards[H_idx, cps_H] += kyotaku_bonus
        bs.round_state.score[H_idx, cps_H] += kyotaku_bonus.to(torch.int32)
        bs.round_state.kyotaku[H_idx] = 0

        # ── 5. Mark won and terminate round ──
        bs.players.has_won[H_idx, cps_H] = True
        bs.round_state.terminated_round[H_idx] = True

        return bs

    # ═════════════════════════════════════════════════════════════
    # _tsumo_batch — FULLY VECTORIZED
    # ═════════════════════════════════════════════════════════════

    def _tsumo_batch(self, bs: BatchState, mask: torch.Tensor):
        """Tsumo (self-draw win) — batch implementation.

        Uses PRECOMPUTED fan/fu (col 1 = TSUMO) from _draw_batch / _draw_after_kan_batch,
        matching serial env.  We cannot call Yaku.judge_hand_related_batch here because
        the drawn tile is already in the hand (added by _draw), and the yaku function
        would add it again via Hand.add, corrupting the hand.
        """
        if not mask.any():
            return bs
        B = bs.B
        device = bs.players.hand.device
        H_idx = torch.arange(B, device=device)[mask]  # (H,)
        H = H_idx.shape[0]

        cps_H = bs.current_player[H_idx]  # (H,)

        # ── 1. Read precomputed yaku (col 1 = TSUMO) ──
        fan_H = bs.players.fan[H_idx, cps_H, 1].to(torch.int32)  # (H,)
        fu_H = bs.players.fu[H_idx, cps_H, 1].to(torch.int32)  # (H,)
        riichi_H = bs.players.riichi[H_idx, cps_H]  # (H,)

        # ── 2. Adjust fan for special conditions ──
        ippatsu_H = bs.players.ippatsu[H_idx, cps_H] & riichi_H  # (H,)
        double_riichi_H = bs.players.double_riichi[H_idx, cps_H]  # (H,)
        can_after_kan_H = bs.round_state.can_after_kan[H_idx]  # (H,)
        haitei_H = bs.round_state.is_haitei[H_idx] & ~can_after_kan_H  # (H,)
        is_yakuman_H = fu_H == 0  # (H,)

        fan_H = fan_H.to(torch.int32)
        extra_fan = (can_after_kan_H.to(torch.int32) + ippatsu_H.to(torch.int32) +
                     double_riichi_H.to(torch.int32) + haitei_H.to(torch.int32))
        fan_H = torch.where(is_yakuman_H, fan_H, fan_H + extra_fan)

        # ── 3. Settle payments (batch) ──
        bs = self._settle_tsumo_batch(bs, H_idx, cps_H, fan_H, fu_H)

        # ── 4. Kyotaku bonus ──
        kyotaku_H = bs.round_state.kyotaku[H_idx]  # (H,)
        kyotaku_bonus = kyotaku_H.to(torch.float32) * 10.0
        bs.rewards[H_idx, cps_H] += kyotaku_bonus
        bs.round_state.score[H_idx, cps_H] += kyotaku_bonus.to(torch.int32)
        bs.round_state.kyotaku[H_idx] = 0

        # ── 5. Mark won and terminate round ──
        bs.players.has_won[H_idx, cps_H] = True
        bs.round_state.terminated_round[H_idx] = True

        return bs

    # ═════════════════════════════════════════════════════════════
    # _pon_batch
    # ═════════════════════════════════════════════════════════════

    def _pon_batch(self, bs: BatchState, mask: torch.Tensor, actions: torch.Tensor):
        """Pon (claim discarded tile for a triple) — batch implementation."""
        if not mask.any():
            return bs
        B = bs.B
        P = 4
        device = bs.players.hand.device
        H_idx_full = torch.arange(B, device=device)[mask]  # (H,)
        H = H_idx_full.shape[0]

        cps_H = bs.current_player[H_idx_full]  # (H,)
        targets_H = bs.round_state.target[H_idx_full]  # (H,)
        discarders_H = bs.round_state.last_player[H_idx_full]  # (H,)
        actions_H = actions[mask]  # (H,)

        # ── 1. Accept riichi (vectorized) ──
        bs = self._accept_riichi_batch(bs, mask)

        # ── 2. Form melds ──
        rel_src_H = (discarders_H - cps_H) % 4  # (H,)
        melds_H = Meld.init_batch(actions_H, targets_H, rel_src_H)  # (H,)

        # ── 3. Append meld to caller's meld list ──
        meld_counts_H = bs.players.meld_counts[H_idx_full, cps_H].long()  # (H,)
        bs.players.melds[H_idx_full, cps_H, meld_counts_H] = melds_H
        bs.players.meld_counts[H_idx_full, cps_H] += 1

        # ── 4. Mark river on discarder ──
        disc_counts_H = bs.players.discard_counts[H_idx_full, discarders_H].long()  # (H,)
        d_idx_H = disc_counts_H - 1  # last discard index
        bs.players.river = River.add_meld_batch(
            bs.players.river, actions_H, discarders_H, d_idx_H, rel_src_H,
            batch_idx=H_idx_full)

        # ── 5. Hand mutation (remove 2 claimed tiles) ──
        target_tt_H = Tile.to_tile_type_tensor(targets_H).long()  # (H,)
        is_pon_red_H = (actions_H == Action.PON_RED)  # (H,)

        # PON: -2 from target_tt
        hands_4p = bs.players.hand_with_red.clone()
        hands_4p[H_idx_full, cps_H, target_tt_H] -= 2
        # PON_RED fix: +1 to target_tt (net -1), -1 from red pos
        if is_pon_red_H.any():
            red_idx = H_idx_full[is_pon_red_H]
            red_pos = Tile.to_red_batch(target_tt_H[is_pon_red_H]).long()
            hands_4p[red_idx, cps_H[is_pon_red_H], target_tt_H[is_pon_red_H]] += 1
            hands_4p[red_idx, cps_H[is_pon_red_H], red_pos] -= 1
        bs.players.hand_with_red = hands_4p
        # Update hand_34
        bs.players.hand = Hand.to_34_batch(hands_4p)

        # ── 6. Clear flags ──
        bs.players.is_hand_concealed[H_idx_full, cps_H] = False
        bs.players.ippatsu[H_idx_full] = False

        # ── 7. Build kuikae mask (VECTORIZED) ──
        # After pon: all tiles in hand except the EXACT claimed target tile.
        # JAX serial: m[state.round_state.target] = False (raw tile, no to_tile_type).
        # We must NOT also prohibit the red-five counterpart — JAX does not.
        post_hands_H = hands_4p[H_idx_full, cps_H]  # (H, 37)
        masks_H = post_hands_H > 0  # (H, 37)
        # Kuikae: prohibit only the exact raw target tile (may be a red five 34-36)
        targets_raw_H = bs.round_state.target[H_idx_full].long().clamp(0, 36)  # (H,)
        masks_H[torch.arange(H, device=device), targets_raw_H] = False
        # Extend to full LEGAL_ACTION_SIZE
        masks_full = torch.zeros(H, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device)
        masks_full[:, :Tile.NUM_TILE_TYPE_WITH_RED] = masks_H
        bs.legal_action_mask[H_idx_full] = masks_full
        bs.players.legal_action_mask[H_idx_full, cps_H] = masks_full

        # ── 8. Clear target ──
        bs.round_state.target[H_idx_full] = -1
        bs.round_state.draw_next[H_idx_full] = False
        bs.current_player[H_idx_full] = cps_H

        return bs

    # ═════════════════════════════════════════════════════════════
    # _open_kan_batch — FULLY VECTORIZED
    # ═════════════════════════════════════════════════════════════

    def _open_kan_batch(self, bs: BatchState, mask: torch.Tensor):
        """Open kan (claim discarded tile for a kan) — batch implementation."""
        if not mask.any():
            return bs
        B = bs.B
        P = 4
        device = bs.players.hand.device
        H_idx_full = torch.arange(B, device=device)[mask]  # (H,)
        H = H_idx_full.shape[0]

        cps_H = bs.current_player[H_idx_full]  # (H,)
        targets_H = bs.round_state.target[H_idx_full]  # (H,)
        discarders_H = bs.round_state.last_player[H_idx_full]  # (H,)

        # ── 1. Accept riichi (vectorized) ──
        bs = self._accept_riichi_batch(bs, mask)

        # ── 2. Form melds ──
        rel_src_H = (discarders_H - cps_H) % 4  # (H,)
        act_open_kan = torch.full((H,), Action.OPEN_KAN, dtype=torch.int32, device=device)
        melds_H = Meld.init_batch(act_open_kan, targets_H, rel_src_H)  # (H,)

        # ── 3. Append meld to caller's meld list ──
        meld_counts_H = bs.players.meld_counts[H_idx_full, cps_H].long()  # (H,)
        bs.players.melds[H_idx_full, cps_H, meld_counts_H] = melds_H
        bs.players.meld_counts[H_idx_full, cps_H] += 1

        # ── 4. Mark river on discarder ──
        disc_counts_H = bs.players.discard_counts[H_idx_full, discarders_H].long()  # (H,)
        d_idx_H = disc_counts_H - 1  # last discard index
        bs.players.river = River.add_meld_batch(
            bs.players.river, act_open_kan, discarders_H, d_idx_H, rel_src_H,
            batch_idx=H_idx_full)

        # ── 5. Hand mutation (remove 3 claimed tiles) ──
        hands_H = bs.players.hand_with_red[H_idx_full, cps_H]  # (H, 37)
        hands_H = Hand.open_kan_batch(hands_H, targets_H)  # (H, 37)
        bs.players.hand_with_red[H_idx_full, cps_H] = hands_H
        bs.players.hand = Hand.to_34_batch(bs.players.hand_with_red)

        # ── 6. Clear flags ──
        bs.players.is_hand_concealed[H_idx_full, cps_H] = False
        bs.players.ippatsu[H_idx_full] = False  # all 4 players

        # ── 7. Clear target ──
        bs.round_state.target[H_idx_full] = -1

        # ── 8. Yaku precompute for rinshan ──
        n_kan_before_H = bs.players.n_kan[H_idx_full].sum(dim=1)  # (H,)
        rinshan_ix_H = 10 + n_kan_before_H.long()  # (H,)
        rinshan_tiles_H = bs.round_state.deck[H_idx_full, rinshan_ix_H.clamp(0, 135)]  # (H,)
        self._precompute_yaku_batch(bs, H_idx_full, targets_H, cps_H,
                                     tsumo_tiles=rinshan_tiles_H)

        # ── 9. Draw after kan (batch) ──
        bs = self._draw_after_kan_batch(bs, H_idx_full, cps_H,
                                        pre_flip_dora=torch.zeros(H, dtype=torch.bool, device=device))

        return bs

    # ═════════════════════════════════════════════════════════════
    # _chi_batch — FULLY VECTORIZED
    # ═════════════════════════════════════════════════════════════

    def _chi_batch(self, bs: BatchState, mask: torch.Tensor, actions: torch.Tensor):
        """Chi (claim discarded tile for a sequence) — batch implementation."""
        if not mask.any():
            return bs
        B = bs.B
        P = 4
        device = bs.players.hand.device
        H_idx_full = torch.arange(B, device=device)[mask]  # (H,)
        H = H_idx_full.shape[0]

        cps_H = bs.current_player[H_idx_full]  # (H,)
        targets_H = bs.round_state.target[H_idx_full]  # (H,)
        discarders_H = bs.round_state.last_player[H_idx_full]  # (H,)
        actions_H = actions[mask]  # (H,)

        # ── 1. Accept riichi (vectorized) ──
        bs = self._accept_riichi_batch(bs, mask)

        # ── 2. Form melds ──
        rel_src_H = (discarders_H - cps_H) % 4  # (H,)
        melds_H = Meld.init_batch(actions_H, targets_H, rel_src_H)  # (H,)

        # ── 3. Append meld to caller's meld list ──
        meld_counts_H = bs.players.meld_counts[H_idx_full, cps_H].long()  # (H,)
        bs.players.melds[H_idx_full, cps_H, meld_counts_H] = melds_H
        bs.players.meld_counts[H_idx_full, cps_H] += 1

        # ── 4. Mark river on discarder ──
        disc_counts_H = bs.players.discard_counts[H_idx_full, discarders_H].long()  # (H,)
        d_idx_H = disc_counts_H - 1  # last discard index
        bs.players.river = River.add_meld_batch(
            bs.players.river, actions_H, discarders_H, d_idx_H, rel_src_H,
            batch_idx=H_idx_full)

        # ── 5. Hand mutation (remove 2 non-claimed tiles) ──
        hands_H = bs.players.hand_with_red[H_idx_full, cps_H]  # (H, 37)
        hands_H = Hand.chi_batch(hands_H, targets_H, actions_H)  # (H, 37)
        bs.players.hand_with_red[H_idx_full, cps_H] = hands_H
        bs.players.hand = Hand.to_34_batch(bs.players.hand_with_red)

        # ── 6. Clear flags ──
        bs.players.is_hand_concealed[H_idx_full, cps_H] = False
        bs.players.ippatsu[H_idx_full] = False  # all 4 players

        # ── 7. Build kuikae mask (VECTORIZED) ──
        # After chi: all tiles in hand except claimed tile + prohibitive tile
        masks_H = hands_H > 0  # (H, 37) base: all held tiles
        target_tt_H = Tile.to_tile_type_tensor(targets_H).long()  # (H,)

        # Kuikae: prohibit the claimed target tile type
        masks_H[torch.arange(H, device=device), target_tt_H] = False
        # Also prohibit red counterpart for 5's
        is_five = Tile.is_tile_type_five_batch(target_tt_H)  # (H,)
        if is_five.any():
            fi = is_five.nonzero(as_tuple=False).squeeze(-1)
            masks_H[fi, Tile.to_red_batch(target_tt_H[is_five]).long()] = False

        # Kuikae: additionally prohibit alternate-sequence tile for CHI_L/CHI_R
        chi_idx_H = Meld._chi_index_batch(actions_H)  # (H,) 0/1/2/-1
        is_chi_l = chi_idx_H == 0  # (H,)
        is_chi_r = chi_idx_H == 2  # (H,)
        not_seven = ~((target_tt_H % 9 == 6) & (target_tt_H < 27))  # (H,)
        not_three = ~((target_tt_H % 9 == 2) & (target_tt_H < 27))  # (H,)

        # CHI_L: prohibit target+3 (if not at position 6)
        chi_l_mask = is_chi_l & not_seven  # (H,)
        if chi_l_mask.any():
            cl_idx = chi_l_mask.nonzero(as_tuple=False).squeeze(-1)
            prohib_l = target_tt_H[cl_idx] + 3
            masks_H[cl_idx, prohib_l] = False
            # Red counterpart for prohibitive tile
            p_is_five = Tile.is_tile_type_five_batch(prohib_l)
            if p_is_five.any():
                pf = cl_idx[p_is_five]
                masks_H[pf, Tile.to_red_batch(prohib_l[p_is_five]).long()] = False

        # CHI_R: prohibit target-3 (if not at position 2)
        chi_r_mask = is_chi_r & not_three  # (H,)
        if chi_r_mask.any():
            cr_idx = chi_r_mask.nonzero(as_tuple=False).squeeze(-1)
            prohib_r = target_tt_H[cr_idx] - 3
            masks_H[cr_idx, prohib_r.clamp(0, 36)] = False
            # Red counterpart for prohibitive tile
            p_is_five_r = Tile.is_tile_type_five_batch(prohib_r)
            if p_is_five_r.any():
                pf = cr_idx[p_is_five_r]
                masks_H[pf, Tile.to_red_batch(prohib_r[p_is_five_r]).long()] = False

        # Extend to full LEGAL_ACTION_SIZE
        masks_full = torch.zeros(H, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device)
        masks_full[:, :Tile.NUM_TILE_TYPE_WITH_RED] = masks_H
        bs.legal_action_mask[H_idx_full] = masks_full
        bs.players.legal_action_mask[H_idx_full, cps_H] = masks_full

        # ── 8. Clear target ──
        bs.round_state.target[H_idx_full] = -1
        bs.round_state.draw_next[H_idx_full] = False
        bs.current_player[H_idx_full] = cps_H

        return bs

    # ═════════════════════════════════════════════════════════════
    # _selfkan_batch — FULLY VECTORIZED
    # ═════════════════════════════════════════════════════════════

    def _selfkan_batch(self, bs: BatchState, mask: torch.Tensor, actions: torch.Tensor):
        """Closed/Added kan (kan from own hand) — batch implementation."""
        if not mask.any():
            return bs
        B = bs.B
        P = 4
        device = bs.players.hand.device
        H_idx_full = torch.arange(B, device=device)[mask]  # (H,)
        H = H_idx_full.shape[0]

        cps_H = bs.current_player[H_idx_full]  # (H,)
        actions_H = actions[mask]  # (H,)
        tile_types_H = actions_H - 37  # (H,) 0-33

        # ── 1. Determine closed vs added kan ──
        is_added_H = torch.zeros(H, dtype=torch.bool, device=device)
        pon_meld_slot_H = torch.full((H,), -1, dtype=torch.int32, device=device)
        pon_src_H = torch.zeros(H, dtype=torch.int32, device=device)

        melds_H4 = bs.players.melds[H_idx_full, cps_H]  # (H, MAX_MELDS)
        n_meld_H = bs.players.meld_counts[H_idx_full, cps_H].long()  # (H,)

        for m_slot in range(MAX_MELDS_PER_PLAYER):
            has_slot = n_meld_H > m_slot  # (H,)
            if not has_slot.any():
                continue
            slot_melds = melds_H4[:, m_slot]  # (H,)
            is_pon = (slot_melds != EMPTY_MELD) & Meld.is_pon_batch(slot_melds) & \
                     (Meld.target_batch(slot_melds) == tile_types_H)
            match = has_slot & is_pon & ~is_added_H  # first match wins
            if match.any():
                mi = torch.arange(H, device=device)[match]
                is_added_H[mi] = True
                pon_meld_slot_H[mi] = m_slot
                pon_src_H[mi] = Meld.src_batch(slot_melds)[mi]

        is_closed_H = ~is_added_H  # (H,)

        # ── 2a. Process closed kan envs ──
        if is_closed_H.any():
            c_idx = H_idx_full[is_closed_H]  # (C,)
            c_cps = cps_H[is_closed_H]  # (C,)
            c_tt = tile_types_H[is_closed_H]  # (C,)
            C = c_idx.shape[0]

            # Form melds (src=0 for closed kan)
            c_actions = actions_H[is_closed_H]  # (C,)
            # For closed kan, target is tile type (0-33), Meld.init_batch handles it
            c_melds = Meld.init_batch(c_actions, c_tt, torch.zeros(C, dtype=torch.int32, device=device))

            # Append meld
            c_meld_counts = bs.players.meld_counts[c_idx, c_cps].long()
            bs.players.melds[c_idx, c_cps, c_meld_counts] = c_melds
            bs.players.meld_counts[c_idx, c_cps] += 1

            # Hand mutation
            hands_C = bs.players.hand_with_red[c_idx, c_cps]
            hands_C = Hand.closed_kan_batch(hands_C, c_tt)
            bs.players.hand_with_red[c_idx, c_cps] = hands_C

            # Flip dora NOW (before _draw_after_kan)
            c_dora = bs.round_state.n_kan_doras[c_idx].long()
            dora_ix_C = 9 - 2 * (c_dora + 1)
            ura_ix_C = 8 - 2 * (c_dora + 1)
            valid_dora = (c_dora + 1 < MAX_DORA_INDICATORS) & (dora_ix_C >= 0)
            if valid_dora.any():
                v_idx = c_idx[valid_dora]
                bs.round_state.dora_indicators[v_idx, c_dora[valid_dora] + 1] = \
                    bs.round_state.deck[v_idx, dora_ix_C[valid_dora].long().clamp(0, 135)]
                bs.round_state.ura_dora_indicators[v_idx, c_dora[valid_dora] + 1] = \
                    bs.round_state.deck[v_idx, ura_ix_C[valid_dora].long().clamp(0, 135)]
                bs.round_state.n_kan_doras[v_idx] += 1

            # Yaku precompute with rinshan tile
            c_n_kan = bs.players.n_kan[c_idx].sum(dim=1).long()
            c_rinshan = bs.round_state.deck[c_idx, (10 + c_n_kan).clamp(0, 135)]
            self._precompute_yaku_batch(bs, c_idx, c_tt, c_cps, tsumo_tiles=c_rinshan)

            # Draw after kan (pre_flip_dora=True since we already flipped)
            bs = self._draw_after_kan_batch(bs, c_idx, c_cps,
                                            pre_flip_dora=torch.ones(C, dtype=torch.bool, device=device))

        # ── 2b. Process added kan envs ──
        if is_added_H.any():
            a_idx = H_idx_full[is_added_H]  # (A,)
            a_cps = cps_H[is_added_H]  # (A,)
            a_tt = tile_types_H[is_added_H]  # (A,)
            a_src = pon_src_H[is_added_H]  # (A,)
            a_slot = pon_meld_slot_H[is_added_H]  # (A,)
            A = a_idx.shape[0]

            # Form melds (use original PON's src for added kan)
            a_actions = actions_H[is_added_H]  # (A,)
            # For added kan, target is tile type (0-33), Meld.init_batch handles it
            a_melds = Meld.init_batch(a_actions, a_tt, a_src)

            # Replace existing PON meld (don't increment meld_counts)
            bs.players.melds[a_idx, a_cps, a_slot] = a_melds

            # Hand mutation
            hands_A = bs.players.hand_with_red[a_idx, a_cps]
            hands_A = Hand.added_kan_batch(hands_A, a_tt)
            bs.players.hand_with_red[a_idx, a_cps] = hands_A

            # Yaku precompute with rinshan tile
            a_n_kan = bs.players.n_kan[a_idx].sum(dim=1).long()
            a_rinshan = bs.round_state.deck[a_idx, (10 + a_n_kan).clamp(0, 135)]
            self._precompute_yaku_batch(bs, a_idx, a_tt, a_cps, tsumo_tiles=a_rinshan)

            # Draw after kan (pre_flip_dora=False — dora will be flipped here)
            bs = self._draw_after_kan_batch(bs, a_idx, a_cps,
                                            pre_flip_dora=torch.zeros(A, dtype=torch.bool, device=device))

        # ── 3. Common updates ──
        bs.players.hand = Hand.to_34_batch(bs.players.hand_with_red)
        bs.round_state.target[H_idx_full] = -1

        return bs

    # ═════════════════════════════════════════════════════════════
    # _discard_batch — FULLY VECTORIZED (highest frequency path)
    # ═════════════════════════════════════════════════════════════

    def _discard_batch(self, bs: BatchState, mask: torch.Tensor, actions: torch.Tensor):
        """Batch discard/tsumogiri. Fully vectorized — the core hot path."""
        import time as _time
        if not mask.any():
            return bs
        B = bs.B
        P = 4
        device = bs.players.hand.device
        b_idx_full = torch.arange(B, device=device)
        m_idx = b_idx_full[mask]
        M = mask.sum().item()

        cps = bs.current_player[m_idx]  # (M,)

        # Resolve tile: discard action → tile index; tsumogiri → last_draw
        is_tsumo = actions[mask] == Action.TSUMOGIRI
        tiles = torch.where(is_tsumo, bs.round_state.last_draw[m_idx], actions[mask])
        tiles = tiles.clamp(0, Tile.NUM_TILE_TYPE_WITH_RED - 1)

        d_counts = bs.players.discard_counts[m_idx, cps]  # (M,)
        is_riichi_flag = bs.players.riichi_declared[m_idx, cps]  # (M,)

        # ── 1. Remove tile from hand ──
        hands_4p = bs.players.hand_with_red.clone()  # (B, 4, 37)
        hands_4p[m_idx, cps, tiles] -= 1
        bs.players.hand_with_red = hands_4p
        # Batch to_34 conversion for ALL envs (safe since only current players changed)
        _t0 = _time.time()
        bs.players.hand = Hand.to_34_batch(bs.players.hand_with_red)
        self._perf_add('hand.to_34_batch', _time.time() - _t0, B * P)

        # ── 2. Add to river ──
        tsumo_flags = is_tsumo.clone()
        bs.players.river = River.add_discard_batch(
            bs.players.river, tiles, cps, d_counts, tsumo_flags, is_riichi_flag,
            batch_idx=m_idx)

        # Update discards (vectorized scatter)
        d_safe = d_counts.long().clamp(0, MAX_DISCARDS_PER_PLAYER - 1)
        bs.players.discards[m_idx, cps, d_safe] = tiles.to(torch.int16)
        new_counts = (bs.players.discard_counts[m_idx, cps] + 1).clamp(max=MAX_DISCARDS_PER_PLAYER)
        bs.players.discard_counts[m_idx, cps] = new_counts

        # ── 3. Action history (vectorized) ──
        ah_M = bs.round_state.action_history[m_idx]  # (M, 3, 200)
        is_empty_first_row = ah_M[:, 0, :] == -1  # (M, 200)
        has_space = is_empty_first_row.any(dim=1)  # (M,)
        first_empty = is_empty_first_row.int().argmax(dim=1)  # (M,) — first -1 position

        # Write tsumogiri flag (1 if tsumogiri, 0 otherwise)
        tsumo_flag = is_tsumo.to(torch.int8)

        # Envs with space: direct write
        if has_space.any():
            hs_idx = torch.arange(M, device=device)[has_space]
            col = first_empty[has_space]  # (K,)
            ah_M[hs_idx, 0, col] = cps[has_space].to(torch.int8)
            ah_M[hs_idx, 1, col] = actions[m_idx][has_space].to(torch.int8)
            ah_M[hs_idx, 2, col] = tsumo_flag[has_space]

        # Full envs: shift left + write at end
        full = ~has_space  # (M,)
        if full.any():
            fi = torch.arange(M, device=device)[full]
            ah_M[fi, :, :-1] = ah_M[fi, :, 1:].clone()
            ah_M[fi, 0, -1] = cps[full].to(torch.int8)
            ah_M[fi, 1, -1] = actions[m_idx][full].to(torch.int8)
            ah_M[fi, 2, -1] = tsumo_flag[full]

        bs.round_state.action_history[m_idx] = ah_M

        # ── 4. Furiten by discard — vectorized ──
        hands_post_M = bs.players.hand_with_red[m_idx, cps]  # (M, 37)

        # Compute can_win for all 34 tile types in ONE batched call
        # Expand (M, 37) → (M, 34, 37) and add tile t at position t
        _t_furiten = _time.time()
        test_all = hands_post_M.unsqueeze(1).expand(M, 34, 37).clone()  # (M, 34, 37)
        test_all[:, torch.arange(34, device=device), torch.arange(34, device=device)] += 1
        test_flat = test_all.reshape(M * 34, 37)
        can_tsumo_flat = Hand.can_tsumo_batch(test_flat)  # (M*34,)
        can_win_M = can_tsumo_flat.reshape(M, 34)

        bs.players.can_win[m_idx, cps] = can_win_M

        # Furiten: check if any river tile is a waiting tile
        disc_offsets = bs.players.discard_counts[m_idx, cps].long()  # (M,)
        is_furiten_M = torch.zeros(M, dtype=torch.bool, device=device)

        # Decode all rivers at once (M, MAX_DISCARDS) → tile values
        river_tiles_M = torch.full((M, MAX_DISCARDS_PER_PLAYER), -1,
                                   dtype=torch.int32, device=device)
        for ri in range(MAX_DISCARDS_PER_PLAYER):
            valid_r = disc_offsets > ri  # (M,)
            if not valid_r.any():
                continue
            decoded = River.decode_tile(
                bs.players.river[m_idx[valid_r], cps[valid_r]])  # (V, MAX_DISCARDS)
            if decoded.ndim == 2:
                river_tiles_M[valid_r, ri] = decoded[torch.arange(valid_r.sum(), device=device), ri]
            else:
                river_tiles_M[valid_r, ri] = decoded[valid_r]

        # Vectorized check: for each river tile, check if it's a waiting tile
        for ri in range(MAX_DISCARDS_PER_PLAYER):
            valid_r = disc_offsets > ri  # (M,)
            if not valid_r.any():
                continue
            rt_r = river_tiles_M[valid_r, ri]  # (V,)
            rt_ok = (rt_r >= 0) & (rt_r <= 36)  # (V,) — include red fives (34-36)
            if rt_ok.any():
                rv_idx = torch.arange(M, device=device)[valid_r][rt_ok]
                rt_val = Tile.to_tile_type_tensor(rt_r[rt_ok]).long()  # red fives → tile types (34→4, 35→13, 36→22)
                is_furiten_M[rv_idx] |= can_win_M[rv_idx, rt_val]
                if is_furiten_M.all():
                    break

        bs.players.furiten_by_discard[m_idx, cps] = is_furiten_M
        self._perf_add('discard.can_win+furiten', _time.time() - _t_furiten, M)
        # Clear furiten_by_pass for furiten players
        if is_furiten_M.any():
            bs.players.furiten_by_pass[m_idx[is_furiten_M], cps[is_furiten_M]] = False

        # ── 5. Clear per-discard flags (vectorized) ──
        # NOTE: had_after_kan must be captured BEFORE clearing can_after_kan
        # (JAX captures it at top of _discard for is_four_kan_draw check)
        had_after_kan_M = bs.round_state.can_after_kan[m_idx].clone()  # (M,)
        bs.round_state.last_draw[m_idx] = -1
        bs.players.ippatsu[m_idx, cps] = False
        bs.round_state.can_after_kan[m_idx] = False

        # ── 6. Set target and last_player (vectorized) ──
        bs.round_state.target[m_idx] = tiles.to(bs.round_state.target.dtype)
        bs.round_state.last_player[m_idx] = cps

        # ── 7. Haitei check (vectorized) ──
        # JAX: is_haitei = is_haitei | (next_deck_ix < last_deck_ix)
        # is_abortive_draw_normal is set CONDITIONALLY later in _make_legal_mask_after_discard_batch
        nxt = bs.round_state.next_deck_ix[m_idx]
        lst = bs.round_state.last_deck_ix[m_idx]
        bs.round_state.is_haitei[m_idx] = bs.round_state.is_haitei[m_idx] | (nxt < lst)

        # ── 8. Precompute yaku for all 4 players ──
        self._precompute_yaku_batch(bs, m_idx, tiles, cps)

        # ── 9. Build meld/ron masks for other players ──
        self._make_legal_mask_after_discard_batch(bs, m_idx, cps, tiles, had_after_kan_M)

        return bs

    # ═════════════════════════════════════════════════════════════
    # _make_legal_mask_after_discard_batch — FULLY VECTORIZED
    # ═════════════════════════════════════════════════════════════

    def _pass_batch(self, bs: BatchState, mask: torch.Tensor):
        """Pass: move to next responder or draw. Vectorized.

        JAX reference: env.py _pass (L1599-1672) + _next_meld_player (L1130-1161).
        Key points:
        1. Zero out the passing player's legal_action_mask (JAX L1611)
        2. Find next meld player by priority: RON > OPEN_KAN > PON > CHI
        3. If multiple RON: closest to discarded_player wins
        4. If no meld player: clear target, set draw_next, draw
        """
        if not mask.any():
            return bs
        B = bs.B
        device = bs.players.hand.device
        m_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)
        M = m_idx.shape[0]

        cps = bs.current_player[m_idx]              # (M,)
        mask_4p = bs.players.legal_action_mask[m_idx]  # (M, 4, 87)

        # ── 1. Furiten by pass ──
        ron_mask_avail = mask_4p[torch.arange(M, device=device), cps, Action.RON]  # (M,)
        if ron_mask_avail.any():
            f_idx = m_idx[ron_mask_avail]
            f_cp = cps[ron_mask_avail]
            bs.players.furiten_by_pass[f_idx, f_cp] = True

        # ── 2. Zero out the passing player's mask (JAX L1611) ──
        mask_4p[torch.arange(M, device=device), cps, :] = False
        # Write back the cleared mask to state
        bs.players.legal_action_mask[m_idx] = mask_4p

        # ── 3. Priority-based search (JAX _next_meld_player) ──
        can_ron = mask_4p[:, :, Action.RON]  # (M, 4)
        can_open_kan = mask_4p[:, :, Action.OPEN_KAN]  # (M, 4)
        can_pon = mask_4p[:, :, Action.PON] | mask_4p[:, :, Action.PON_RED]  # (M, 4)
        can_chi = mask_4p[:, :, Action.CHI_L:Action.CHI_R_RED + 1].any(dim=2)  # (M, 4)

        has_any = can_ron | can_open_kan | can_pon | can_chi  # (M, 4)

        # Priority: RON=3, OPEN_KAN=2, PON=1, CHI=0, NONE=-1
        priority = torch.where(can_ron, torch.tensor(3, device=device),
                     torch.where(can_open_kan, torch.tensor(2, device=device),
                       torch.where(can_pon, torch.tensor(1, device=device),
                         torch.where(can_chi, torch.tensor(0, device=device),
                           torch.tensor(-1, device=device)))))  # (M, 4)

        # Multiple RON tie-breaking: closest to discarded_player (JAX L1157-1161)
        multi_ron = can_ron.sum(dim=1) > 1  # (M,)
        if multi_ron.any():
            mr_last = bs.round_state.last_player[m_idx[multi_ron]]  # (K,)
            mr_ron = can_ron[multi_ron]  # (K, 4)
            distance = (torch.arange(4, device=device).unsqueeze(0) - mr_last.unsqueeze(1)) % 4  # (K, 4)
            distance = torch.where(mr_ron, distance, torch.tensor(99, device=device))
            priority[multi_ron] = torch.where(
                mr_ron,
                torch.tensor(3, device=device),  # keep ron priority high
                priority[multi_ron])
            # Override: set priority to 3+small_bonus for closest ron player
            best_ron = distance.argmin(dim=1)  # (K,)
            # Set all ron priorities to 3 except the winner
            for k in range(multi_ron.sum().item()):
                ki = torch.where(multi_ron)[0][k]
                for p in range(4):
                    if mr_ron[k, p]:
                        priority[ki, p] = 2 if p != best_ron[k] else 3

        next_p = priority.argmax(dim=1).to(torch.int32)  # (M,) — argmax returns int64, must cast
        has_responder = has_any[torch.arange(M, device=device), next_p]  # (M,)

        # ── 4. Has responder → set next player ──
        if has_responder.any():
            r_idx = m_idx[has_responder]
            r_p = next_p[has_responder]
            bs.current_player[r_idx] = r_p
            bs.legal_action_mask[r_idx] = mask_4p[has_responder, r_p]
            # Ensure PASS is available for next responder (JAX L1666)
            bs.players.legal_action_mask[r_idx, r_p, Action.PASS] = True

        # ── 5. No responder → clear target, draw/abort (JAX L1646-1659) ──
        no_responder = ~has_responder  # (M,)
        if no_responder.any():
            nr_idx = m_idx[no_responder]
            nr_last = bs.round_state.last_player[nr_idx]
            bs.current_player[nr_idx] = (nr_last + 1) % 4
            bs.round_state.target[nr_idx] = -1  # clear target (JAX L1653)

            need_abort = bs.round_state.is_abortive_draw_normal[nr_idx]  # (N,)
            if need_abort.any():
                bs = self._abortive_draw_normal_batch(bs, nr_idx[need_abort])

            need_d = ~need_abort  # (N,)
            if need_d.any():
                draw_mask = torch.zeros(B, dtype=torch.bool, device=device)
                draw_mask[nr_idx[need_d]] = True
                bs = self._draw_batch(bs, draw_mask)

        return bs

    # ═════════════════════════════════════════════════════════════
    # _kyuushu_batch — VECTORIZED (JAX _special_next_round)
    # ═════════════════════════════════════════════════════════════

    def _kyuushu_batch(self, bs: BatchState, mask: torch.Tensor):
        """Abortive draw — batch implementation. JAX: redeal with same round, honba+1.

        If self._kyuushu_deck_overrides is set (dict of env_idx → deck_tensor),
        those decks are used instead of generating via torch.randperm. This allows
        replay tests to inject JAX-generated decks for PRNG-independent verification.
        """
        if not mask.any():
            return bs
        m_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)
        M = m_idx.shape[0]
        B = bs.B
        P = 4
        device = bs.players.hand.device
        deck_overrides = getattr(self, '_kyuushu_deck_overrides', None) or {}

        bs.round_state.is_abortive_draw_normal[m_idx] = True
        bs.rewards[m_idx] = 0.0
        bs.round_state.honba[m_idx] += 1

        # Rebuild wall and deal fresh hands
        deck_vals = torch.tensor([t for t in range(34) for _ in range(4)], dtype=torch.int8, device=device)
        for k in range(M):
            i = int(m_idx[k].item())
            if i in deck_overrides:
                bs.round_state.deck[i] = deck_overrides.pop(i).to(device)
            else:
                gen = torch.Generator(device='cpu').manual_seed(
                    int(bs.round_state.round[i].item()) * 100 + int(bs.round_state.honba[i].item()))
                perm = torch.randperm(136, generator=gen)
                bs.round_state.deck[i] = deck_vals[perm]
            bs.round_state.next_deck_ix[i] = FIRST_DRAW_IDX
            bs.round_state.last_deck_ix[i] = DEAD_WALL_TILES
            bs.round_state.draw_next[i] = True
            bs.round_state.dora_indicators[i, 0] = bs.round_state.deck[i, 9]
            bs.round_state.ura_dora_indicators[i, 0] = bs.round_state.deck[i, 8]
            bs.round_state.dora_indicators[i, 1:] = -1
            bs.round_state.ura_dora_indicators[i, 1:] = -1
            bs.round_state.n_kan_doras[i] = 0

            for p in range(P):
                bs.players.hand[i, p].zero_()
                bs.players.hand_with_red[i, p].zero_()
                bs.players.discard_counts[i, p] = 0
                bs.players.meld_counts[i, p] = 0
                bs.players.melds[i, p].fill_(EMPTY_MELD)
                bs.players.river[i, p].fill_(EMPTY_RIVER)
                bs.players.n_kan[i, p] = 0
                bs.players.is_hand_concealed[i, p] = True
                bs.players.riichi[i, p] = False
                bs.players.riichi_declared[i, p] = False
                bs.players.ippatsu[i, p] = False
                bs.players.double_riichi[i, p] = False
                bs.players.furiten_by_discard[i, p] = False
                bs.players.furiten_by_pass[i, p] = False
                bs.players.has_won[i, p] = False
                bs.players.has_nagashi_mangan[i, p] = True
                bs.players.has_yaku[i, p].zero_()
                bs.players.fan[i, p].zero_()
                bs.players.fu[i, p].zero_()
                bs.players.can_win[i, p].zero_()
                bs.players.legal_action_mask[i, p].zero_()

            bs.players.hand_with_red[i] = Hand.make_init_hand(bs.round_state.deck[i])
            bs.players.hand[i] = Hand.to_34_batch(bs.players.hand_with_red[i])

            bs.current_player[i] = int(bs.round_state.dealer[i].item())
            bs.round_state.target[i] = -1
            bs.round_state.last_draw[i] = -1
            bs.round_state.last_player[i] = -1
            bs.round_state.is_haitei[i] = False
            bs.round_state.is_abortive_draw_normal[i] = False
            bs.round_state.kan_declared[i] = False
            bs.round_state.can_after_kan[i] = False
            bs.round_state.can_robbing_kan[i] = False
            bs.round_state.terminated_round[i] = False

        # Draw for all redealt envs
        bs = self._draw_batch(bs, mask)

        return bs

    # ═════════════════════════════════════════════════════════════
    # _dummy_batch
    # ═════════════════════════════════════════════════════════════

    def _dummy_batch(self, bs: BatchState, mask: torch.Tensor):
        """Dummy step for round transition — mostly vectorized."""
        if not mask.any():
            return bs
        m_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)

        bs.round_state.dummy_count[m_idx] += 1

        # Envs that reached 4 dummy steps need round advance (delegate to serial)
        need_adv = bs.round_state.dummy_count[m_idx] >= 4
        if need_adv.any():
            a_idx = m_idx[need_adv]
            for idx in a_idx.cpu().numpy():
                idx = int(idx)
                s = unstack_state(bs, idx)
                s = self._serial._advance_to_next_round_auto(s)
                self._copy_state_into_batch(bs, idx, s)

        return bs

    # ═════════════════════════════════════════════════════════════
    # Batch helpers: draw, yaku precompute, settlement, round advance
    # ═════════════════════════════════════════════════════════════

    def _draw_batch(self, bs: BatchState, mask: torch.Tensor):
        """Batch draw (vectorized core path). Mirrors env_serial._draw.

        Handles: deck advance, tile draw, hand add, yaku precompute copy,
        shanten, flag updates. Special abortive draw checks are per-env (rare).
        Mask building is per-env (complex, to be vectorized later).
        """
        if not mask.any():
            return bs
        B = bs.B
        P = 4
        device = bs.players.hand.device
        m_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)
        M = m_idx.shape[0]

        # ── 1. Accept pending riichi (vectorized) ──
        bs = self._accept_riichi_batch(bs, mask)

        cps = bs.current_player[m_idx]  # (M,)

        # ── 2. Special abortive draw check (VECTORIZED) ──
        # Four-wind (四風連打): all 4 first discards same wind tile, first turn, no melds
        # Four-riichi (四家立直): all 4 players in riichi
        # Uses discards[m_idx, :, 0] raw tile values (no river decoding needed)
        special_abort = torch.zeros(B, dtype=torch.bool, device=device)
        if self.game_config.enable_special_abortive_draw:
            # Four-wind check: first discards from discards tensor (raw tile values)
            first_disc_M = bs.players.discards[m_idx, :, 0]  # (M, 4) — first discard per player
            disc_counts_M = bs.players.discard_counts[m_idx]  # (M, 4)
            all_have_discards = (disc_counts_M > 0).all(dim=1)  # (M,)

            # Check each first discard is a wind tile (27-30)
            is_wind_M = (first_disc_M >= 27) & (first_disc_M <= 30)  # (M, 4)
            all_winds = is_wind_M.all(dim=1)  # (M,)
            # Check all four are the same
            all_same = (first_disc_M == first_disc_M[:, 0:1]).all(dim=1)  # (M,)
            is_four_wind = all_have_discards & all_winds & all_same  # (M,)

            # Pure first turn check
            is_pure_first = (
                (bs.round_state.next_deck_ix[m_idx] >= FIRST_DRAW_IDX - 5) &
                (bs.players.meld_counts[m_idx].sum(dim=1) == 0)
            )  # (M,)

            is_four_wind_draw = is_four_wind & is_pure_first  # (M,)
            is_four_riichi_draw = bs.players.riichi[m_idx].sum(dim=1) == 4  # (M,)

            is_special = is_four_wind_draw | is_four_riichi_draw  # (M,)
            if is_special.any():
                s_idx = m_idx[is_special]  # (S,)
                s_cps = bs.current_player[s_idx]  # (S,)
                # Set KYUUSHU-only mask for affected envs (vectorized)
                ky_mask = torch.zeros(4, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device)
                ky_mask[:, Action.KYUUSHU] = True
                bs.players.legal_action_mask[s_idx] = ky_mask
                bs.legal_action_mask[s_idx] = ky_mask[0]  # env-level: any row works
                bs.round_state.draw_next[s_idx] = False
                bs.round_state.kan_declared[s_idx] = False
                bs.round_state.is_abortive_draw_normal[s_idx] = False
                special_abort[s_idx] = True

        # Filter to envs that did NOT abort; these proceed to normal draw.
        active_draw = mask.clone()
        if special_abort.any():
            active_draw = active_draw & ~special_abort
        if not active_draw.any():
            return bs

        active_idx = active_draw.nonzero(as_tuple=False).squeeze(-1)  # (K,)
        K = active_idx.shape[0]

        # Build a sub-index into m_idx for the surviving envs
        # Map active_idx → position in m_idx
        active_cps = bs.current_player[active_idx]  # (K,)

        # ── 3. Advance deck pointer and draw tile (vectorized) ──
        next_ix = bs.round_state.next_deck_ix[active_idx]  # (K,)
        is_haitei = next_ix == bs.round_state.last_deck_ix[active_idx]  # (K,)
        bs.round_state.is_haitei[active_idx] = is_haitei

        new_tiles = bs.round_state.deck[active_idx, next_ix.long()]  # (K,)
        bs.round_state.next_deck_ix[active_idx] = next_ix - 1
        bs.round_state.last_draw[active_idx] = new_tiles.to(torch.int32)

        # ── 4. Add tile to hand (vectorized) ──
        hands_4p = bs.players.hand_with_red.clone()
        hands_cp = hands_4p[active_idx, active_cps]  # (K, 37)
        hands_cp = Hand.add_batch(hands_cp, new_tiles)  # (K, 37)
        hands_4p[active_idx, active_cps] = hands_cp
        bs.players.hand_with_red = hands_4p
        bs.players.hand = Hand.to_34_batch(bs.players.hand_with_red)  # full batch

        # ── 5. Copy yaku precompute (col 1 → col 0) — vectorized ──
        bs.players.has_yaku[active_idx, active_cps, 0] = \
            bs.players.has_yaku[active_idx, active_cps, 1].clone()
        bs.players.fan[active_idx, active_cps, 0] = \
            bs.players.fan[active_idx, active_cps, 1].clone()
        bs.players.fu[active_idx, active_cps, 0] = \
            bs.players.fu[active_idx, active_cps, 1].clone()

        # ── 6. Build legal action mask (VECTORIZED via shared helper) ──
        bs = self._make_legal_mask_after_draw_batch(
            bs, active_idx, active_cps,
            bs.players.hand_with_red[active_idx, active_cps],  # post-draw hands
            bs.round_state.last_draw[active_idx],
            bs.players.riichi[active_idx, active_cps],
            bs.round_state.is_haitei[active_idx],
            bs.players.n_kan[active_idx].sum(dim=1),
            bs.players.meld_counts[active_idx, active_cps],
            bs.players.can_win[active_idx, active_cps],
            bs.players.has_yaku[active_idx, active_cps, 0],
            bs.players.is_hand_concealed[active_idx, active_cps],
            bs.round_state.score[active_idx, active_cps],
            bs.round_state.next_deck_ix[active_idx],
            bs.round_state.last_deck_ix[active_idx],
            bs.round_state.can_after_kan[active_idx],
        )

        # ── 7. Common flag updates (vectorized) ──
        bs.round_state.draw_next[active_idx] = False
        bs.round_state.kan_declared[active_idx] = False
        bs.round_state.target[active_idx] = -1

        # ── 8. Shanten (vectorized) ──
        hands_cp_34 = bs.players.hand[active_idx, active_cps]  # (K, 34)
        shanten_vals = Shanten.number_batch(hands_cp_34)  # (K,)
        bs.round_state.shanten_current_player[active_idx] = shanten_vals.to(torch.int32)

        # ── 9. Clear furiten_by_pass for non-riichi players (vectorized) ──
        is_riichi = bs.players.riichi[active_idx, active_cps]  # (K,)
        non_riichi = ~is_riichi  # (K,)
        if non_riichi.any():
            nr_idx = active_idx[non_riichi]
            nr_cp = active_cps[non_riichi]
            bs.players.furiten_by_pass[nr_idx, nr_cp] = False

        return bs

    def _draw_after_kan_batch(self, bs: BatchState, m_idx: torch.Tensor,
                               cps: torch.Tensor, pre_flip_dora: torch.Tensor):
        """Batch rinshan draw after a kan (closed, open, or added).

        Args:
            m_idx: (M,) batch indices
            cps: (M,) current player in each env
            pre_flip_dora: (M,) bool — True if dora was already flipped (closed kan)
        """
        M = m_idx.shape[0]
        if M == 0:
            return bs
        B = bs.B
        device = bs.players.hand.device

        # ── 1. Draw rinshan tile (from dead wall slot 10+n_kan) ──
        n_kan_sum_M = bs.players.n_kan[m_idx].sum(dim=1).long()  # (M,)
        rinshan_ix_M = 10 + n_kan_sum_M  # (M,)
        rinshan_tiles_M = bs.round_state.deck[m_idx, rinshan_ix_M.clamp(0, 135)]  # (M,)

        # ── 2. Increment n_kan for current player ──
        bs.players.n_kan[m_idx, cps] += 1

        # ── 3. Flip dora if not already flipped ──
        flip_dora_M = ~pre_flip_dora  # (M,)
        if flip_dora_M.any():
            f_idx = m_idx[flip_dora_M]  # (K,)
            n_dora_K = bs.round_state.n_kan_doras[f_idx].long()  # (K,)
            # deck index: 9 - 2*(n_dora+1) for dora, 8 - 2*(n_dora+1) for ura
            dora_ix_K = 9 - 2 * (n_dora_K + 1)  # (K,)
            ura_ix_K = 8 - 2 * (n_dora_K + 1)  # (K,)
            valid_ix = (n_dora_K + 1 < MAX_DORA_INDICATORS) & (dora_ix_K >= 0)  # (K,)
            if valid_ix.any():
                v_idx = f_idx[valid_ix]
                bs.round_state.dora_indicators[v_idx, n_dora_K[valid_ix] + 1] = \
                    bs.round_state.deck[v_idx, dora_ix_K[valid_ix].long().clamp(0, 135)]
                bs.round_state.ura_dora_indicators[v_idx, n_dora_K[valid_ix] + 1] = \
                    bs.round_state.deck[v_idx, ura_ix_K[valid_ix].long().clamp(0, 135)]
                bs.round_state.n_kan_doras[v_idx] += 1

        # ── 4. Extend dead wall for all kan types ──
        bs.round_state.last_deck_ix[m_idx] += 1

        # ── 5. Clear per-kan flags ──
        bs.players.ippatsu[m_idx, cps] = False
        bs.round_state.kan_declared[m_idx] = False
        bs.round_state.can_after_kan[m_idx] = True
        bs.round_state.can_robbing_kan[m_idx] = False

        # ── 6. Compute can_win from pre-draw (14-tile) hand (VECTORIZED) ──
        hands_pre_M = bs.players.hand_with_red[m_idx, cps]  # (M, 37)
        # Expand to (M, 34, 37) and add each tile type
        test_all = hands_pre_M.unsqueeze(1).expand(M, 34, 37).clone()  # (M, 34, 37)
        test_all[:, torch.arange(34, device=device), torch.arange(34, device=device)] += 1
        test_flat = test_all.reshape(M * 34, 37)
        can_tsumo_flat = Hand.can_tsumo_batch(test_flat)  # (M*34,)
        bs.players.can_win[m_idx, cps] = can_tsumo_flat.reshape(M, 34)

        # ── 7. Set last_draw and add tile to hand ──
        bs.round_state.last_draw[m_idx] = rinshan_tiles_M.to(torch.int32)
        hands_4p = bs.players.hand_with_red.clone()
        hands_4p[m_idx, cps] = Hand.add_batch(hands_4p[m_idx, cps], rinshan_tiles_M)
        bs.players.hand_with_red = hands_4p
        bs.players.hand = Hand.to_34_batch(bs.players.hand_with_red)

        # ── 8. Copy tsumo yaku precompute → col 0 ──
        bs.players.has_yaku[m_idx, cps, 0] = bs.players.has_yaku[m_idx, cps, 1].clone()
        bs.players.fan[m_idx, cps, 0] = bs.players.fan[m_idx, cps, 1].clone()
        bs.players.fu[m_idx, cps, 0] = bs.players.fu[m_idx, cps, 1].clone()

        # ── 9. Rinshan draws are never haitei ──
        bs.round_state.is_haitei[m_idx] = False

        # ── 10. Build legal action mask (VECTORIZED) ──
        bs.round_state.draw_next[m_idx] = False
        bs = self._make_legal_mask_after_draw_batch(bs, m_idx, cps,
            hands_4p[m_idx, cps],  # use the updated hand
            bs.round_state.last_draw[m_idx],
            bs.players.riichi[m_idx, cps],
            bs.round_state.is_haitei[m_idx],
            bs.players.n_kan[m_idx].sum(dim=1),
            bs.players.meld_counts[m_idx, cps],
            bs.players.can_win[m_idx, cps],
            bs.players.has_yaku[m_idx, cps, 0],
            bs.players.is_hand_concealed[m_idx, cps],
            bs.round_state.score[m_idx, cps],
            bs.round_state.next_deck_ix[m_idx],
            bs.round_state.last_deck_ix[m_idx],
            bs.round_state.can_after_kan[m_idx],
        )

        return bs
