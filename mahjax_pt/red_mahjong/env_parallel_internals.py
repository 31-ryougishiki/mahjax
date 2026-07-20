# Copyright 2025 The Mahjax Authors.
# Internal helpers for RedMahjongParallel: mask builders, settlement, yaku precompute.
#
# Mixed into RedMahjongParallel via InternalsMixin.

import dataclasses
from typing import Optional
import torch

from .action import Action
from .constants import (
    FIRST_DRAW_IDX, MAX_DISCARDS_PER_PLAYER, NUM_PLAYERS,
    NUM_TILE_TYPES, NUM_TILE_TYPES_WITH_RED,
    LEGAL_ACTION_SIZE, MAX_MELDS_PER_PLAYER, MAX_DORA_INDICATORS,
    RIICHI_BET, SENTINEL_MELD_VALUE,
)
from .meld import Meld, EMPTY_MELD
from .tile import River, Tile, EMPTY_RIVER
from .hand import Hand
from .yaku import Yaku
from .batch_state import BatchState, stack_states, unstack_state
from .state import EnvState
from .env_serial import _trigger_special_abortive_draw



def _copy_dataclass_row(dst, dst_idx, src, src_idx):
    """Copy row src_idx from a source dataclass to row dst_idx of a destination dataclass.

    Recursively handles nested dataclass fields (e.g. BatchPlayerState inside BatchState).
    Both dst and src must share the same dataclass structure with tensor fields
    of compatible shapes.  Scalar fields on src that became tensor fields on dst
    are handled naturally via tensor indexing.
    """
    for field in dataclasses.fields(dst):
        src_val = getattr(src, field.name)
        if src_val is None:
            continue
        if isinstance(src_val, torch.Tensor):
            getattr(dst, field.name)[dst_idx] = src_val[src_idx]
        elif dataclasses.is_dataclass(src_val):
            _copy_dataclass_row(
                getattr(dst, field.name), dst_idx,
                src_val, src_idx)


class InternalsMixin:
    """Mixin for RedMahjongParallel: mask builders, settlement, yaku, round management.

    These are the most complex internal methods.  Isolating them here keeps
    the main env_parallel.py focused on the high-level dispatch logic.
    """

    def _make_legal_mask_after_discard_batch(self, bs: BatchState, m_idx: torch.Tensor,
                                              cps: torch.Tensor, tiles: torch.Tensor,
                                              had_after_kan_M: torch.Tensor = None):
        """Build per-player masks after discard. Vectorized using batch 4p helpers."""
        B = bs.B
        P = 4
        device = bs.players.hand.device
        M = m_idx.shape[0]

        discarded_players = cps  # (M,)
        target_vals = tiles  # (M,)
        target_tts = Tile.to_tile_type_tensor(target_vals).long()  # (M,)

        # ── 1. Compute per-player conditions (batch) ──
        hands_4p = bs.players.hand_with_red[m_idx]  # (M, 4, 37)
        is_riichi_4p = bs.players.riichi[m_idx]     # (M, 4)
        meld_counts_4p = bs.players.meld_counts[m_idx]  # (M, 4)
        n_kan_sum_M = bs.players.n_kan[m_idx].sum(dim=1)  # (M,)

        # Haitei: is_haitei OR next_deck_ix < last_deck_ix
        haitei_M = bs.round_state.is_haitei[m_idx] | \
                   (bs.round_state.next_deck_ix[m_idx] < bs.round_state.last_deck_ix[m_idx])  # (M,)

        meld_full_4p = meld_counts_4p >= MAX_MELDS_PER_PLAYER  # (M, 4)
        cannot_meld_4p = is_riichi_4p | haitei_M.unsqueeze(1) | meld_full_4p  # (M, 4)
        cannot_kan_M = n_kan_sum_M >= 4  # (M,)
        cannot_kan_4p = cannot_kan_M.unsqueeze(1).expand(M, P)  # (M, 4)

        # Discarded player mask (M, 4)
        disc_player_mask = torch.zeros(M, P, dtype=torch.bool, device=device)
        disc_player_mask[torch.arange(M, device=device), discarded_players.long()] = True

        # Source mask for chi: src == 3 (player to the left of discarder)
        src = (discarded_players.unsqueeze(1) - torch.arange(P, device=device).unsqueeze(0)) % 4  # (M, 4)
        src3_mask = (src == 3)  # (M, 4)

        # ── 2. Batch chi check ──
        chi_mat = Hand.can_chi_matrix_batch_4p(hands_4p, target_vals, src3_mask)  # (M, 4, 6)
        chi_any = chi_mat.any(dim=2)  # (M, 4)

        # ── 3. Batch pon / open kan check ──
        can_no_red_pon = Hand.can_no_red_pon_batch_4p(hands_4p, target_tts)  # (M, 4)
        can_red_pon = Hand.can_red_pon_batch_4p(hands_4p, target_tts)  # (M, 4)
        can_open_kan_4p = Hand.can_open_kan_batch_4p(hands_4p, target_tts)  # (M, 4)
        can_pon_any = can_no_red_pon | can_red_pon  # (M, 4)

        # ── 4. Batch ron check (use fresh can_ron, NOT cached can_win) ──
        has_yaku_4p = bs.players.has_yaku[m_idx, :, 0]  # (M, 4)
        furiten_4p = bs.players.furiten_by_discard[m_idx] | bs.players.furiten_by_pass[m_idx]  # (M, 4)
        can_ron_val = torch.zeros(M, P, dtype=torch.bool, device=device)
        for p in range(P):
            can_ron_val[:, p] = Hand.can_ron_batch(hands_4p[:, p, :], target_vals)  # (M,)
        can_ron_4p = (has_yaku_4p | haitei_M.unsqueeze(1)) & ~furiten_4p & can_ron_val  # (M, 4)

        # ── 5. Combine into mask_4p ──
        # Initialize per-action masks
        chi_ok = chi_any & ~cannot_meld_4p & src3_mask & \
                  (Tile.to_tile_type_tensor(target_vals).unsqueeze(1) < 27) & ~disc_player_mask  # (M, 4)
        pon_ok = can_pon_any & ~cannot_meld_4p & ~disc_player_mask  # (M, 4)
        kan_ok = can_open_kan_4p & ~cannot_meld_4p & ~cannot_kan_4p & ~disc_player_mask  # (M, 4)
        ron_ok = can_ron_4p & ~disc_player_mask  # (M, 4)

        any_action = chi_ok | pon_ok | kan_ok | ron_ok  # (M, 4)
        pass_ok = any_action  # (M, 4)  PASS available if any action available

        # ── 6. Build mask tensor (M, 4, LEGAL_ACTION_SIZE) ──
        mask_4p_full = torch.zeros(M, P, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device)
        mask_4p_full[:, :, Action.RON] = ron_ok
        mask_4p_full[:, :, Action.PON] = pon_ok & can_no_red_pon  # refined per-env below
        mask_4p_full[:, :, Action.PON_RED] = pon_ok & can_red_pon
        mask_4p_full[:, :, Action.OPEN_KAN] = kan_ok
        mask_4p_full[:, :, Action.PASS] = pass_ok

        # Fill chi per-action slots from chi_mat
        chi_action_map = [
            (0, Action.CHI_L), (1, Action.CHI_L_RED),
            (2, Action.CHI_M), (3, Action.CHI_M_RED),
            (4, Action.CHI_R), (5, Action.CHI_R_RED),
        ]
        for col, act in chi_action_map:
            mask_4p_full[:, :, act] = chi_ok & chi_mat[:, :, col]

        # Refine: PON/PON_RED requires actual hand check (not just count>=2)
        # The 4p batch functions already do this, so we use them directly
        mask_4p_full[:, :, Action.PON] = mask_4p_full[:, :, Action.PON] & can_no_red_pon
        mask_4p_full[:, :, Action.PON_RED] = mask_4p_full[:, :, Action.PON_RED] & can_red_pon

        # ── 7. Determine next player per env ──
        can_ron_v = mask_4p_full[:, :, Action.RON]  # (M, 4)
        can_pon_v = mask_4p_full[:, :, Action.PON] | mask_4p_full[:, :, Action.PON_RED]  # (M, 4)
        can_kan_v = mask_4p_full[:, :, Action.OPEN_KAN]  # (M, 4)
        can_chi_v = mask_4p_full[:, :, Action.CHI_L:Action.CHI_R_RED + 1].any(dim=2)  # (M, 4)

        can_any = can_ron_v | can_pon_v | can_kan_v | can_chi_v  # (M, 4)
        no_meld_player = ~can_any.any(dim=1)  # (M,)
        no_ron_player = ~can_ron_v.any(dim=1)  # (M,)

        # JAX L1003-1008: is_four_kan_draw (四開槓流れ)
        if had_after_kan_M is not None and self.game_config.enable_special_abortive_draw:
            n_kan_sum = bs.players.n_kan[m_idx].sum(dim=1)  # (M,)
            n_kan_players = (bs.players.n_kan[m_idx] > 0).sum(dim=1)  # (M,)
            is_four_kan = (had_after_kan_M & (n_kan_sum >= 4) & (n_kan_players >= 2) & no_ron_player)  # (M,)
        else:
            is_four_kan = torch.zeros(M, dtype=torch.bool, device=device)

        # JAX: is_abortive_draw_normal computed locally, applied conditionally
        is_abort_M = bs.round_state.next_deck_ix[m_idx] < bs.round_state.last_deck_ix[m_idx]  # (M,)

        # JAX L1029: no_meld_player | (is_abortive_draw_normal & no_ron_player)
        go_to_draw = no_meld_player | (is_abort_M & no_ron_player)  # (M,)
        has_responder = ~go_to_draw & ~is_four_kan  # (M,)

        # Priority: RON(3) > OPEN_KAN(2) > PON(1) > CHI(0)
        priority = torch.where(can_ron_v, 3,
                    torch.where(can_kan_v, 2,
                    torch.where(can_pon_v, 1,
                    torch.where(can_chi_v, 0, -1))))  # (M, 4)

        next_p = torch.argmax(priority, dim=1).to(torch.int32)  # (M,)

        # Multiple ron: closest in turn order from discarder
        multi_ron = can_ron_v.sum(dim=1) > 1  # (M,)
        if multi_ron.any():
            distances = (torch.arange(P, device=device).unsqueeze(0) - discarded_players.unsqueeze(1)) % 4  # (M, 4)
            distances = torch.where(can_ron_v, distances, torch.full_like(distances, float('inf')))
            next_p = torch.where(multi_ron, torch.argmin(distances, dim=1), next_p)

        # ── 8. Apply results ──
        # Envs with responders
        if has_responder.any():
            r_idx = m_idx[has_responder]  # (R,)
            r_next_p = next_p[has_responder]  # (R,)
            bs.current_player[r_idx] = r_next_p
            bs.legal_action_mask[r_idx] = mask_4p_full[has_responder, r_next_p]
            bs.players.legal_action_mask[r_idx] = mask_4p_full[has_responder]
            bs.round_state.last_player[r_idx] = discarded_players[has_responder]
            bs.round_state.draw_next[r_idx] = False

        # Envs with four-kan draw (JAX L1024-1027)
        if is_four_kan.any():
            fk_idx = m_idx[is_four_kan]  # (F,)
            fk_disc = discarded_players[is_four_kan]  # (F,)
            for fk_i in fk_idx.cpu().numpy():
                fk_i = int(fk_i)
                s = unstack_state(bs, fk_i)
                s = _trigger_special_abortive_draw(s)
                self._copy_state_into_batch(bs, fk_i, s)

        # Envs without responders → draw or abortive draw (JAX L1029-1035)
        no_responder = ~has_responder & ~is_four_kan  # (M,)
        if no_responder.any():
            nr_idx = m_idx[no_responder]  # (N,)
            nr_disc = discarded_players[no_responder]  # (N,)
            bs.current_player[nr_idx] = (nr_disc + 1) % 4
            bs.round_state.target[nr_idx] = -1
            bs.round_state.draw_next[nr_idx] = True
            bs.round_state.last_player[nr_idx] = nr_disc

            # JAX L1035: is_abortive_draw_normal set conditionally here
            nr_is_abort = is_abort_M[no_responder]  # (N,)
            bs.round_state.is_abortive_draw_normal[nr_idx] = nr_is_abort

            if nr_is_abort.any():
                bs = self._abortive_draw_normal_batch(bs, nr_idx[nr_is_abort])

            need_d = ~nr_is_abort  # (N,)
            if need_d.any():
                draw_mask = torch.zeros(B, dtype=torch.bool, device=device)
                draw_mask[nr_idx[need_d]] = True
                bs = self._draw_batch(bs, draw_mask)

        return bs

    # ═════════════════════════════════════════════════════════════
    # _pass_batch
    # ═════════════════════════════════════════════════════════════

    def _make_legal_mask_after_draw_batch(self, bs: BatchState, idx: torch.Tensor,
                                           cps: torch.Tensor, hands: torch.Tensor,
                                           last_draw: torch.Tensor,
                                           is_riichi: torch.Tensor,
                                           is_haitei: torch.Tensor,
                                           n_kan_sum: torch.Tensor,
                                           meld_counts: torch.Tensor,
                                           can_win: torch.Tensor,
                                           has_yaku: torch.Tensor,
                                           is_concealed: torch.Tensor,
                                           scores: torch.Tensor,
                                           next_deck_ix: torch.Tensor,
                                           last_deck_ix: torch.Tensor,
                                           can_after_kan: torch.Tensor):
        """Build legal action masks after draw. VECTORIZED for all paths.

        Args:
            idx: (K,) batch indices into BatchState
            cps: (K,) current players
            hands: (K, 37) post-draw hands
            last_draw: (K,) last drawn tile
            is_riichi: (K,) whether player is in riichi
            is_haitei: (K,) haitei flag
            n_kan_sum: (K,) total kan count across all players
            meld_counts: (K,) meld count for current player
            can_win: (K, 34) can_win matrix
            has_yaku: (K,) has_yaku col 0
            is_concealed: (K,) hand concealed flag
            scores: (K,) current player's score
            next_deck_ix: (K,)
            last_deck_ix: (K,)
            can_after_kan: (K,) rinshan flag
        """
        K = idx.shape[0]
        if K == 0:
            return bs
        B = bs.B
        device = hands.device

        masks_K = torch.zeros(K, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device)
        ld_valid = (last_draw >= 0) & (last_draw < Tile.NUM_TILE_TYPE_WITH_RED)  # (K,)

        # ── Normal (non-riichi) path — VECTORIZED ──
        norm_mask = ~is_riichi  # (K,)
        if norm_mask.any():
            n_idx = norm_mask.nonzero(as_tuple=False).squeeze(-1)  # (K2,)

            # Discard actions: all tiles with count > 0, except last_draw needs >= 2
            disc_ok = hands[n_idx] > 0  # (K2, 37)
            masks_K[n_idx, :Tile.NUM_TILE_TYPE_WITH_RED] = disc_ok

            ld_n = ld_valid[n_idx]  # (K2,)
            if ld_n.any():
                l_idx = n_idx[ld_n]
                ld_val = last_draw[l_idx].long().clamp(0, 36)
                masks_K[l_idx, ld_val] = hands[l_idx, ld_val] >= 2

            # Tsumogiri
            tsumo_ok = ld_valid[n_idx] & (hands[n_idx, last_draw[n_idx].long().clamp(0, 36)] > 0)
            masks_K[n_idx[tsumo_ok], Action.TSUMOGIRI] = True

            # Self kan (closed + added)
            cannot_kan = is_haitei[n_idx] | (n_kan_sum[n_idx] >= 4)  # (K2,)
            can_kan = ~cannot_kan  # (K2,)
            if can_kan.any():
                k_idx = n_idx[can_kan]  # (K3,)
                K3 = k_idx.shape[0]
                tile_range = torch.arange(34, device=device)

                # Closed kan — fully vectorized via broadcasting
                closed_kan_mask = Hand.can_closed_kan_batch(hands[k_idx])  # (K3, 34)
                masks_K[k_idx.unsqueeze(1), 37 + tile_range.unsqueeze(0)] = closed_kan_mask

                # Added kan — check meld slots for PON with matching target
                melds_K3 = bs.players.melds[idx][k_idx, cps[k_idx]]  # (K3, MAX_MELDS)
                n_meld_K3 = meld_counts[n_idx][can_kan].long()  # (K3,)
                can_added_all = Hand.can_added_kan_batch(hands[k_idx])  # (K3, 34)
                for m_slot in range(MAX_MELDS_PER_PLAYER):
                    has_slot = n_meld_K3 > m_slot  # (K3,)
                    if not has_slot.any():
                        break
                    slot_melds = melds_K3[:, m_slot]  # (K3,)
                    is_pon = (slot_melds != EMPTY_MELD) & Meld.is_pon_batch(slot_melds)  # (K3,)
                    if not is_pon.any():
                        continue
                    # Match PON target against each of 34 tile types via broadcasting
                    slot_targets = Meld.target_batch(slot_melds)  # (K3,)
                    targets_match = slot_targets.unsqueeze(1) == tile_range.unsqueeze(0)  # (K3, 34)
                    add_ok = has_slot.unsqueeze(1) & is_pon.unsqueeze(1) & targets_match & can_added_all  # (K3, 34)
                    masks_K[k_idx.unsqueeze(1), 37 + tile_range.unsqueeze(0)] |= add_ok

            # TSUMO
            can_tsumo_K2 = Hand.can_tsumo_batch(hands[n_idx])  # (K2,)
            tsu_cond = can_tsumo_K2 & (
                is_concealed[n_idx] | can_after_kan[n_idx] | is_haitei[n_idx] | has_yaku[n_idx])
            masks_K[n_idx[tsu_cond], Action.TSUMO] = True

            # RIICHI
            tiles_left_K2 = next_deck_ix[n_idx] - last_deck_ix[n_idx]  # (K2,)
            riichi_cond = (
                ~bs.players.riichi[idx[n_idx], cps[n_idx]] &
                (scores[n_idx] >= RIICHI_BET // 100) &
                is_concealed[n_idx] &
                (tiles_left_K2 >= 4) &
                Hand.can_riichi_batch(hands[n_idx]))
            masks_K[n_idx[riichi_cond], Action.RIICHI] = True

            # KYUUSHU
            is_first = next_deck_ix[n_idx] >= FIRST_DRAW_IDX - 4  # (K2,)
            no_melds = bs.players.meld_counts[idx[n_idx]].sum(dim=1) == 0  # (K2,)
            kyu_cond = (is_first & Hand.can_kyuushu_batch(hands[n_idx]) & no_melds
                        if self.game_config.enable_special_abortive_draw
                        else torch.zeros_like(is_first))
            masks_K[n_idx[kyu_cond], Action.KYUUSHU] = True

        # ── Riichi path — VECTORIZED (was per-env) ──
        if is_riichi.any():
            rii_idx = is_riichi.nonzero(as_tuple=False).squeeze(-1)  # (R,)
            R = rii_idx.shape[0]
            rii_env_idx = idx[rii_idx]  # (R,) batch indices
            rii_cps = cps[rii_idx]  # (R,)

            # TSUMOGIRI always available for riichi players
            masks_K[rii_idx, Action.TSUMOGIRI] = True

            # Closed kan after riichi (if not haitei)
            not_haitei = ~is_haitei[rii_idx]  # (R,)
            if not_haitei.any():
                nh = not_haitei.nonzero(as_tuple=False).squeeze(-1)  # (R2,)
                nh_envs = rii_env_idx[nh]  # (R2,)
                nh_cps = rii_cps[nh]  # (R2,)
                nh_hands = hands[rii_idx][nh]  # (R2, 37)
                nh_can_win = can_win[rii_idx][nh]  # (R2, 34)
                R2 = nh.shape[0]

                # Which tiles can be closed-kan'd
                can_ck = Hand.can_closed_kan_batch(nh_hands)  # (R2, 34)

                # For each tile type that's a valid closed kan candidate
                for t in range(34):
                    can_t = can_ck[:, t]  # (R2,)
                    if not can_t.any():
                        continue
                    t_envs = nh_envs[can_t]  # (R3,)
                    t_cps = nh_cps[can_t]  # (R3,)
                    t_hands = nh_hands[can_t]  # (R3, 37)
                    t_can_win = nh_can_win[can_t]  # (R3, 34)
                    R3 = t_envs.shape[0]

                    # Do closed kan (batch)
                    t_tensor = torch.full((R3,), t, dtype=torch.int32, device=device)
                    post_kan = Hand.closed_kan_batch(t_hands, t_tensor)  # (R3, 37)

                    # Compute new can_ron for all 34 tiles (batch, same as can_win)
                    test_all = post_kan.unsqueeze(1).expand(R3, 34, 37).clone()  # (R3, 34, 37)
                    test_all[:, torch.arange(34, device=device), torch.arange(34, device=device)] += 1
                    test_flat = test_all.reshape(R3 * 34, 37)
                    new_can_ron = Hand.can_tsumo_batch(test_flat).reshape(R3, 34)  # (R3, 34)

                    # Check if wait pattern is preserved
                    matches = (new_can_ron == t_can_win).all(dim=1)  # (R3,)
                    if matches.any():
                        # Map R3 index back to K index
                        rii_k_idx = rii_idx[nh][can_t][matches]  # (R4,)
                        masks_K[rii_k_idx, 37 + t] = True

            # TSUMO: can_win on the drawn tile
            if ld_valid[rii_idx].any():
                ld_ok = ld_valid[rii_idx].nonzero(as_tuple=False).squeeze(-1)  # (L,)
                ld_k = rii_idx[ld_ok]  # (L,) K-indices
                ld_val = last_draw[ld_k].long().clamp(0, 36)  # (L,)
                ld_tt = Tile.to_tile_type_tensor(ld_val).long().clamp(0, 33)  # (L,)
                can_tsumo = can_win[ld_k, ld_tt]  # (L,)
                masks_K[ld_k[can_tsumo], Action.TSUMO] = True

        # ── Write masks back ──
        bs.legal_action_mask[idx] = masks_K
        bs.players.legal_action_mask[idx, cps] = masks_K

        return bs

    def _precompute_yaku_batch(self, bs: BatchState, m_idx: torch.Tensor,
                                tiles: torch.Tensor, cps: torch.Tensor,
                                tsumo_tiles: Optional[torch.Tensor] = None,
                                ron_mask: Optional[torch.Tensor] = None,
                                tsumo_mask: Optional[torch.Tensor] = None):
        """Fully vectorized yaku precompute using Yaku.judge_hand_related_batch.

        For each of the 4 players in all M envs, computes:
          col 0 = RON  on the given tiles (controlled by ron_mask)
          col 1 = TSUMO on next deck / tsumo_tiles (controlled by tsumo_mask)

        Args:
            ron_mask: (M, 4) bool or None. Which (env, player) pairs need col 0.
                      None = all True (backward compat).
            tsumo_mask: (M, 4) bool or None. Which (env, player) pairs need col 1.
                        None = all True (backward compat).

        Optimizations:
          - RON: batched across all 4 players via reshape (方案 C, -4→1 kernel launches)
          - TSUMO: controlled by mask (方案 A, skip entirely during discard)
        """
        M = m_idx.shape[0]
        if M == 0:
            return
        B = bs.B
        device = bs.players.hand.device

        # ── Extract per-env data ──
        hands_4p = bs.players.hand_with_red[m_idx]        # (M, 4, 37)
        melds_4p = bs.players.melds[m_idx]                 # (M, 4, MAX_MELDS)
        meld_counts_4p = bs.players.meld_counts[m_idx]     # (M, 4)
        riichi_4p = bs.players.riichi[m_idx]               # (M, 4)
        seat_winds_4p = bs.round_state.seat_wind[m_idx]    # (M, 4)
        prevalent_winds = bs.round_state.round[m_idx] // 4  # (M,)
        dora_inds = bs.round_state.dora_indicators[m_idx]  # (M, 5)
        ura_dora_inds = bs.round_state.ura_dora_indicators[m_idx]  # (M, 5)

        if ron_mask is None:
            ron_mask = torch.ones(M, 4, dtype=torch.bool, device=device)
        if tsumo_mask is None:
            tsumo_mask = torch.ones(M, 4, dtype=torch.bool, device=device)

        need_ron = ron_mask.any()
        need_tsumo = tsumo_mask.any()
        if not need_ron and not need_tsumo:
            return

        # Flatten common data once
        B4 = M * 4
        hands_flat = hands_4p.reshape(B4, 37)
        melds_flat = melds_4p.reshape(B4, melds_4p.shape[2])
        meld_counts_flat = meld_counts_4p.reshape(B4)
        riichi_flat = riichi_4p.reshape(B4)
        prevalent_flat = prevalent_winds.unsqueeze(1).expand(M, 4).reshape(B4)
        seat_flat = seat_winds_4p.reshape(B4)
        dora_flat = dora_inds.unsqueeze(1).expand(M, 4, 5).reshape(B4, 5)
        ura_flat = ura_dora_inds.unsqueeze(1).expand(M, 4, 5).reshape(B4, 5)

        # ═══════════════════════════════════════════════════════════════
        # RON + TSUMO merged into ONE call (common path: both masks full)
        # ═══════════════════════════════════════════════════════════════
        if need_ron and need_tsumo and tsumo_tiles is None and \
           ron_mask.all() and tsumo_mask.all():
            tiles_flat = tiles.unsqueeze(1).expand(M, 4).reshape(B4)
            nxt = bs.round_state.next_deck_ix[m_idx].long().clamp(0, 135)
            next_flat = bs.round_state.deck[m_idx, nxt].unsqueeze(1).expand(M, 4).reshape(B4)

            # Stack RON + TSUMO: (M*8, ...)
            B8 = B4 * 2
            yaku_all, fan_all, fu_all = Yaku.judge_hand_related_batch(
                torch.cat([hands_flat, hands_flat], dim=0),
                torch.cat([melds_flat, melds_flat], dim=0),
                torch.cat([meld_counts_flat, meld_counts_flat], dim=0),
                torch.cat([tiles_flat, next_flat], dim=0),
                torch.cat([riichi_flat, riichi_flat], dim=0),
                torch.cat([
                    torch.ones(B4, dtype=torch.bool, device=device),
                    torch.zeros(B4, dtype=torch.bool, device=device)]),
                torch.cat([prevalent_flat, prevalent_flat], dim=0),
                torch.cat([seat_flat, seat_flat], dim=0),
                torch.cat([dora_flat, dora_flat], dim=0),
                torch.cat([ura_flat, ura_flat], dim=0))

            # Split results: first half = RON, second half = TSUMO
            yaku_ron = yaku_all[:B4].reshape(M, 4, -1)
            yaku_tsumo = yaku_all[B4:].reshape(M, 4, -1)
            bs.players.has_yaku[m_idx, :, 0] = yaku_ron.any(dim=2)
            bs.players.fan[m_idx, :, 0] = fan_all[:B4].reshape(M, 4)
            bs.players.fu[m_idx, :, 0] = fu_all[:B4].reshape(M, 4)
            bs.players.has_yaku[m_idx, :, 1] = yaku_tsumo.any(dim=2)
            bs.players.fan[m_idx, :, 1] = fan_all[B4:].reshape(M, 4)
            bs.players.fu[m_idx, :, 1] = fu_all[B4:].reshape(M, 4)
            return

        # ═══════════════════════════════════════════════════════════════
        # Fallback: separate RON / TSUMO calls (non-standard mask paths)
        # ═══════════════════════════════════════════════════════════════
        if need_ron:
            tiles_flat = tiles.unsqueeze(1).expand(M, 4).reshape(B4)
            yaku_r, fan_r, fu_r = Yaku.judge_hand_related_batch(
                hands_flat, melds_flat, meld_counts_flat,
                tiles_flat, riichi_flat,
                torch.ones(B4, dtype=torch.bool, device=device),
                prevalent_flat, seat_flat, dora_flat, ura_flat)
            bs.players.has_yaku[m_idx, :, 0] = yaku_r.reshape(M, 4, -1).any(dim=2)
            bs.players.fan[m_idx, :, 0] = fan_r.reshape(M, 4)
            bs.players.fu[m_idx, :, 0] = fu_r.reshape(M, 4)

        if need_tsumo:
            if tsumo_tiles is None:
                nxt = bs.round_state.next_deck_ix[m_idx].long().clamp(0, 135)
                next_tiles = bs.round_state.deck[m_idx, nxt]
            else:
                next_tiles = tsumo_tiles
            next_flat = next_tiles.unsqueeze(1).expand(M, 4).reshape(B4)

            mask_flat = tsumo_mask.reshape(B4)
            need_idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)
            if need_idx.shape[0] > 0:
                yaku_t, fan_t, fu_t = Yaku.judge_hand_related_batch(
                    hands_flat[need_idx], melds_flat[need_idx], meld_counts_flat[need_idx],
                    next_flat[need_idx], riichi_flat[need_idx],
                    torch.zeros(need_idx.shape[0], dtype=torch.bool, device=device),
                    prevalent_flat[need_idx], seat_flat[need_idx],
                    dora_flat[need_idx], ura_flat[need_idx])

                yaku_out = torch.zeros(M, 4, yaku_t.shape[1], dtype=torch.bool, device=device)
                fan_out = torch.zeros(M, 4, dtype=torch.int32, device=device)
                fu_out = torch.zeros(M, 4, dtype=torch.int32, device=device)
                need_2d = torch.stack([need_idx // 4, need_idx % 4], dim=1)
                yaku_out[need_2d[:, 0], need_2d[:, 1]] = yaku_t
                fan_out[need_2d[:, 0], need_2d[:, 1]] = fan_t
                fu_out[need_2d[:, 0], need_2d[:, 1]] = fu_t
                bs.players.has_yaku[m_idx, :, 1] = yaku_out.any(dim=2)
                bs.players.fan[m_idx, :, 1] = fan_out
                bs.players.fu[m_idx, :, 1] = fu_out

    # ═════════════════════════════════════════════════════════════
    # Batch settlement helpers
    # ═════════════════════════════════════════════════════════════

    @staticmethod
    def _score_batch(fan, fu):
        """Batch version of Yaku.score: (H,) fan and fu → (H,) int32 base points."""
        H = fan.shape[0]
        device = fan.device
        SCORES = torch.tensor(
            [2000, 2000, 3000, 3000, 4000, 4000, 4000, 6000, 6000, 8000, 8000, 8000],
            dtype=torch.int32, device=device)

        fan_i = fan.to(torch.int32)
        fu_i = fu.to(torch.int32)

        # Yakuman: fu == 0 → base = 8000 * fan
        raw = fu_i * (1 << (fan_i + 2))  # (H,)
        idx = (fan_i - 4).clamp(0, 11)  # (H,)
        capped = SCORES[idx]  # (H,)

        return torch.where(fu_i == 0, 8000 * fan_i,
                          torch.where(raw < 2000, raw, capped))

    def _settle_ron_batch(self, bs: BatchState, m_idx: torch.Tensor,
                           winners: torch.Tensor, losers: torch.Tensor,
                           fan: torch.Tensor, fu: torch.Tensor):
        """Batch ron settlement. Matches serial _settle_ron line-for-line."""
        if m_idx.shape[0] == 0:
            return bs
        base = self._score_batch(fan, fu)  # (H,)
        is_dealer_H = winners == bs.round_state.dealer[m_idx]  # (H,)
        score_H = torch.where(is_dealer_H, base * 6, base * 4)  # (H,)
        score_H = ((score_H.float() / 100.0).ceil()).to(torch.int32)
        honba_H = bs.round_state.honba[m_idx]  # (H,)
        honba_pts = honba_H.to(torch.int32) * 3
        total_H = score_H + honba_pts

        bs.round_state.score[m_idx, winners] += total_H
        bs.round_state.score[m_idx, losers] -= total_H
        bs.rewards[m_idx, winners] = total_H.float()
        bs.rewards[m_idx, losers] = -total_H.float()

        return bs

    def _settle_tsumo_batch(self, bs: BatchState, m_idx: torch.Tensor,
                             winners: torch.Tensor, fan: torch.Tensor, fu: torch.Tensor):
        """Batch tsumo settlement. Matches serial _settle_tsumo line-for-line.

        VECTORIZED: per-player for-loops replaced with masked broadcasting.
        """
        if m_idx.shape[0] == 0:
            return bs
        P = 4
        device = bs.players.hand.device
        base = self._score_batch(fan, fu)  # (H,)
        is_dealer_H = winners == bs.round_state.dealer[m_idx]  # (H,)
        honba_H = bs.round_state.honba[m_idx].to(torch.int32)  # (H,)

        # ── Dealer tsumo ──
        if is_dealer_H.any():
            d_idx = m_idx[is_dealer_H]
            d_winners = winners[is_dealer_H]  # (D,)
            D = d_idx.shape[0]
            payment_d = ((base[is_dealer_H].float() * 2.0 / 100.0).ceil()).to(torch.int32) \
                        + honba_H[is_dealer_H]  # (D,)

            # Build (D, 4) payer mask: all non-winner players pay
            payers_d = torch.arange(P, device=device).unsqueeze(0).expand(D, P) != \
                       d_winners.unsqueeze(1)  # (D, 4)
            pay_d_exp = payment_d.unsqueeze(1).expand(D, P)  # (D, 4)

            # Subtract from non-winners
            bs.round_state.score[d_idx] -= pay_d_exp * payers_d
            bs.rewards[d_idx] -= pay_d_exp.float() * payers_d

            # Winner collects sum of all non-winner payments
            winner_gain_d = (pay_d_exp * payers_d).sum(dim=1)  # (D,)
            bs.round_state.score[d_idx, d_winners] += winner_gain_d
            bs.rewards[d_idx, d_winners] += winner_gain_d.float()

        # ── Non-dealer tsumo ──
        if (~is_dealer_H).any():
            nd_idx = m_idx[~is_dealer_H]
            nd_winners = winners[~is_dealer_H]  # (ND,)
            nd_dealers = bs.round_state.dealer[nd_idx]  # (ND,)
            ND = nd_idx.shape[0]
            non_dealer_pay = ((base[~is_dealer_H].float() / 100.0).ceil()).to(torch.int32) \
                             + honba_H[~is_dealer_H]  # (ND,)
            dealer_pay = ((base[~is_dealer_H].float() * 2.0 / 100.0).ceil()).to(torch.int32) \
                         + honba_H[~is_dealer_H]  # (ND,)

            # Build (ND, 4) payer mask and differentiated payment matrix
            payers_nd = torch.arange(P, device=device).unsqueeze(0).expand(ND, P) != \
                        nd_winners.unsqueeze(1)  # (ND, 4)
            is_dealer_payer = torch.arange(P, device=device).unsqueeze(0).expand(ND, P) == \
                              nd_dealers.unsqueeze(1)  # (ND, 4)
            pay_nd = torch.where(is_dealer_payer,
                                 dealer_pay.unsqueeze(1).expand(ND, P),
                                 non_dealer_pay.unsqueeze(1).expand(ND, P))  # (ND, 4)

            # Subtract from non-winners
            bs.round_state.score[nd_idx] -= pay_nd * payers_nd
            bs.rewards[nd_idx] -= pay_nd.float() * payers_nd

            # Winner collects sum of all non-winner payments
            winner_gain_nd = (pay_nd * payers_nd).sum(dim=1)  # (ND,)
            bs.round_state.score[nd_idx, nd_winners] += winner_gain_nd
            bs.rewards[nd_idx, nd_winners] += winner_gain_nd.float()

        return bs

    # ═════════════════════════════════════════════════════════════
    # _draw_after_kan_batch — VECTORIZED rinshan draw after any kan
    # ═════════════════════════════════════════════════════════════

    def _abortive_draw_normal_batch(self, bs: BatchState, m_idx: torch.Tensor):
        """Exhaustive draw (荒牌流局) settlement — VECTORIZED.

        JAX _abortive_draw_normal: tenpai players split 3000 (30×100) pool,
        noten players pay proportionally. Sets terminated_round."""
        if m_idx.shape[0] == 0:
            return bs
        M = m_idx.shape[0]
        P = 4

        # ── 1. Tenpai check from cached can_win ──
        can_win_M = bs.players.can_win[m_idx]  # (M, 4, 34)
        is_tenpai = can_win_M.any(dim=-1)  # (M, 4)
        n_tenpai = is_tenpai.sum(dim=1)  # (M,)
        n_noten = P - n_tenpai  # (M,)

        # ── 2. Point exchange (only when 0 < n_tenpai < 4) ──
        need_exchange = (n_tenpai > 0) & (n_tenpai < 4)  # (M,)
        if need_exchange.any():
            ex_idx = m_idx[need_exchange]  # (E,)
            ex_tenpai = is_tenpai[need_exchange]  # (E, 4)
            ex_n_tenpai = n_tenpai[need_exchange]  # (E,)
            ex_n_noten = n_noten[need_exchange]  # (E,)
            E = ex_idx.shape[0]

            tenpai_gain = (30 // ex_n_tenpai).to(torch.int32)  # (E,)
            noten_loss = (30 // ex_n_noten).to(torch.int32)  # (E,)

            # Apply to tenpai players (broadcasting, no per-player loop)
            tenpai_gain_exp = tenpai_gain.unsqueeze(1).expand(E, P)  # (E, 4)
            bs.round_state.score[ex_idx] += tenpai_gain_exp * ex_tenpai
            bs.rewards[ex_idx] += tenpai_gain_exp.float() * ex_tenpai

            # Apply to noten players (broadcasting, no per-player loop)
            is_noten = ~ex_tenpai  # (E, 4)
            noten_loss_exp = noten_loss.unsqueeze(1).expand(E, P)  # (E, 4)
            bs.round_state.score[ex_idx] -= noten_loss_exp * is_noten
            bs.rewards[ex_idx] -= noten_loss_exp.float() * is_noten

        # ── 3. Terminate round (all envs) ──
        bs.round_state.terminated_round[m_idx] = True
        bs.round_state.draw_next[m_idx] = False

        return bs

    def _advance_round_batch(self, bs: BatchState, mask: torch.Tensor):
        """Advance to next round for masked envs."""
        if not mask.any():
            return bs
        for idx in mask.nonzero(as_tuple=False).flatten().cpu().numpy():
            idx = int(idx)
            s = unstack_state(bs, idx)
            s = self._serial._advance_to_next_round_auto(s)
            self._copy_state_into_batch(bs, idx, s)
        return bs

    def _copy_state_into_batch(self, bs: BatchState, idx: int, s: EnvState):
        """Copy a single EnvState into a BatchState at position idx.

        Uses stack_states + _copy_dataclass_row to avoid manual field enumeration.
        """
        single_bs = stack_states([s])
        _copy_dataclass_row(bs, idx, single_bs, 0)
