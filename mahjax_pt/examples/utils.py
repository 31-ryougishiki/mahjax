"""Evaluation utilities for MahJax PyTorch agents.

Ported from examples/utils.py (JAX → PyTorch eager).
"""
import torch
import torch.nn.functional as F
from mahjax_pt.red_mahjong.hand import Hand

STARTING_SCORE = 250  # init score in mahjax (1 unit = 100 points)


def make_eval_fn(env, num_eval_envs, max_steps=200):
    """Build an evaluator that pits two policies in a 1-vs-3 setup.

    actor_fn_i(state, rng) -> action_idx (int)
    """
    NUM_PLAYERS = env.num_players

    def one_vs_three(actor_fn_1, actor_fn_2):
        def play_one(init_state, seat_is_a1, gen):
            events = {
                "hand_ended": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "hora_hand": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "ryuu_hand": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "won": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "dealt_in": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "riichi": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "melded": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
                "tenpai_ryuu": torch.zeros(NUM_PLAYERS, dtype=torch.int32),
            }
            state = init_state
            prev_terminated_round = False
            for _ in range(max_steps):
                if state.terminated or state.truncated:
                    break
                cp = state.current_player
                a1 = actor_fn_1(state, gen)
                a2 = actor_fn_2(state, gen)
                action = a1 if seat_is_a1[cp] else a2

                prev_terminated_round = bool(state.round_state.terminated_round)
                state = env.step(state, action)

                # Detect round end rising edge
                hand_ended = bool(state.round_state.terminated_round) and not prev_terminated_round
                if hand_ended:
                    events["hand_ended"][cp] += 1
                    p = state.players
                    won = p.has_won.to(torch.int32)
                    is_hora = won.sum() > 0
                    rewards = state.rewards
                    is_ron = is_hora and (rewards < 0).sum() == 1
                    is_ryuukyoku = not is_hora
                    last_player = int(state.round_state.last_player)
                    dealt_in_mask = torch.zeros(NUM_PLAYERS, dtype=torch.bool)
                    if is_ron:
                        dealt_in_mask[last_player] = True

                    if is_hora:
                        events["hora_hand"][cp] += 1
                    if is_ryuukyoku:
                        events["ryuu_hand"][cp] += 1
                    events["won"] += (hand_ended and p.has_won).to(torch.int32)
                    events["dealt_in"] += (hand_ended and dealt_in_mask).to(torch.int32)
                    events["riichi"] += (hand_ended and p.riichi).to(torch.int32)
                    events["melded"] += (hand_ended and (p.meld_counts > 0)).to(torch.int32)
                    tenpai = torch.tensor([bool(Hand.is_tenpai(Hand.to_34(p.hand_with_red[i]))) for i in range(NUM_PLAYERS)])
                    events["tenpai_ryuu"] += (hand_ended and is_ryuukyoku and tenpai).to(torch.int32)

                    if env.one_round:
                        break

            # Final stats
            score = state.round_state.score.float()
            higher = (score.unsqueeze(0) > score.unsqueeze(1)).sum(dim=1)
            ties_inc_self = (score.unsqueeze(0) == score.unsqueeze(1)).sum(dim=1)
            rank = 1.0 + higher.float() + 0.5 * (ties_inc_self.float() - 1.0)
            gain = score - float(STARTING_SCORE)
            return events, rank, gain

        def eval_fn(key=None):
            # Seed each eval env with a different seat assignment
            results = {"hand_ended": [], "hora_hand": [], "ryuu_hand": [],
                       "won": [], "dealt_in": [], "riichi": [], "melded": [],
                       "tenpai_ryuu": [], "ranks": [], "gains": [], "seat_is_a1": []}
            for i in range(num_eval_envs):
                gen = torch.Generator().manual_seed(i if key is None else (hash(key) + i) % (2 ** 31))
                init_state = env.init(gen)
                agent_seat = int(torch.randint(0, NUM_PLAYERS, (1,), generator=gen).item())
                seat_is_a1 = torch.tensor([s == agent_seat for s in range(NUM_PLAYERS)])
                events, ranks, gains = play_one(init_state, seat_is_a1, gen)

                for k in ["hand_ended", "hora_hand", "ryuu_hand", "won", "dealt_in",
                          "riichi", "melded", "tenpai_ryuu"]:
                    results[k].append(events[k])
                results["ranks"].append(ranks)
                results["gains"].append(gains)
                results["seat_is_a1"].append(seat_is_a1)

            # Aggregate
            seat_is_a1_stack = torch.stack(results["seat_is_a1"])  # (N, P)
            n_hands = torch.stack(results["hand_ended"]).float().sum(dim=1)  # (N,)
            total_hands = n_hands.sum()

            def safe_div(num, den):
                return num / max(den, 1.0)

            def per_view(prefix, mask):
                mf = mask.float()
                player_hands = (n_hands.unsqueeze(1) * mf).sum()
                n_seats = mf.sum()
                w = lambda k: (torch.stack(results[k]).float() * mf).sum()
                return {
                    f"{prefix}/hora_rate": safe_div(w("won"), player_hands),
                    f"{prefix}/deal_in_rate": safe_div(w("dealt_in"), player_hands),
                    f"{prefix}/riichi_rate": safe_div(w("riichi"), player_hands),
                    f"{prefix}/meld_rate": safe_div(w("melded"), player_hands),
                    f"{prefix}/tenpai_at_ryuu_rate": safe_div(w("tenpai_ryuu"),
                        (torch.stack(results["ryuu_hand"]).float() * mf).sum()),
                    f"{prefix}/avg_rank": safe_div((torch.stack(results["ranks"]) * mf).sum(), n_seats),
                    f"{prefix}/avg_gain": safe_div((torch.stack(results["gains"]) * mf).sum(), n_seats),
                }

            metrics = {
                "hand/hora_finish_rate": safe_div(torch.stack(results["hora_hand"]).float().sum(), total_hands),
                "hand/ryuukyoku_rate": safe_div(torch.stack(results["ryuu_hand"]).float().sum(), total_hands),
                "hand/total": total_hands,
            }
            metrics.update(per_view("agent", seat_is_a1_stack))
            metrics.update(per_view("baseline", ~seat_is_a1_stack))
            return metrics

        return eval_fn
    return one_vs_three


def make_policy_fn(network, params):
    """Return actor_fn(state, rng) -> action_idx using a trained network."""
    @torch.no_grad()
    def actor(state, rng=None):
        del rng
        obs = network._env.observe(state) if hasattr(network, '_env') else state
        # Convert observation to tensor dict
        logits, _ = network(obs)
        mask = state.legal_action_mask
        logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        return int(torch.argmax(logits).item())
    return actor
