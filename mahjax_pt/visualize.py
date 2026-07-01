#!/usr/bin/env python3
"""Mahjong game visualization.

Usage:
  # Text board in terminal
  python mahjax_pt/visualize.py --seed 42 --max_steps 50 --render text

  # HTML export (open in browser)
  python mahjax_pt/visualize.py --seed 42 --max_steps 200 --render html

  # With a trained agent
  python mahjax_pt/visualize.py --model params/red_mahjong-seed=0.pt --seed 42 --render html
"""

import torch
import argparse

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.tile import Tile, River
from mahjax_pt.red_mahjong.meld import Meld
from mahjax_pt.red_mahjong.action import Action
from mahjax_pt.red_mahjong.players import random_player, rule_based_player
from mahjax_pt.examples.networks.red_network import ACNet

# ── Tile names ─────────────────────────────────────────────────
_TILE_NAMES = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "E","S","W","N",
    "Wh","Gr","Rd",
    "5mr","5pr","5sr",
]

def _tn(t):
    """Tile index -> readable name."""
    t = int(t) if isinstance(t, torch.Tensor) else t
    if t < 0: return "."
    if t < len(_TILE_NAMES): return _TILE_NAMES[t]
    return f"?{t}"


# ═══════════════════════════════════════════════════════════════
# TEXT BOARD
# ═══════════════════════════════════════════════════════════════

class TextBoard:
    @staticmethod
    def render(state):
        cp = state.current_player
        rs = state.round_state
        wind_label = ("East", "South")[int(rs.round) // 4]
        seat_labels = ["E", "S", "W", "N"]
        lines = []

        lines.append(f"{'='*70}")
        lines.append(f"  {wind_label} {int(rs.round)%4 + 1} | honba={int(rs.honba)} | kyotaku={int(rs.kyotaku)} | step={state.step_count}")
        dora_str = " ".join(_tn(int(d)) for d in rs.dora_indicators if int(d) >= 0)
        lines.append(f"  Dora: {dora_str if dora_str else 'none'}")
        scores = "  ".join(f"{seat_labels[i]}:{int(rs.score[i].item())//10}k" for i in range(4))
        lines.append(f"  Scores: {scores}")
        lines.append(f"{'-'*70}")

        for p in range(4):
            marker = ">>" if p == cp else "  "
            riichi = " [RIICHI]" if bool(state.players.riichi[p]) else ""
            lines.append(f"  {marker} P{p} ({seat_labels[p]}){riichi}")

            # Hand
            hand_tiles = []
            h = state.players.hand_with_red[p]
            for t in range(37):
                cnt = int(h[t].item())
                hand_tiles.extend([_tn(t)] * cnt)
            lines.append(f"     Hand:  {' '.join(hand_tiles) if hand_tiles else '(none)'}")

            # Melds
            nm = int(state.players.meld_counts[p].item())
            if nm > 0:
                meld_strs = []
                for i in range(nm):
                    mv = int(state.players.melds[p, i].item())
                    if mv == 0xFFFF: continue
                    act = Meld.action(mv)
                    tgt = Meld.target(mv)
                    if Meld.is_pon(mv):       meld_strs.append(f"Pon({_tn(tgt)})")
                    elif Meld.is_kan(mv):      meld_strs.append(f"Kan({_tn(tgt)})")
                    elif Meld.is_chi(mv):
                        ci = Meld._chi_index(act)
                        meld_strs.append(f"Chi({_tn(tgt-ci)}{_tn(tgt-ci+1)}{_tn(tgt-ci+2)})")
                    else:                      meld_strs.append(f"?")
                lines.append(f"     Melds: {' | '.join(meld_strs)}")

            # River
            rd = River.decode_tile(state.players.river[p])
            rv = [_tn(int(rd[i].item())) for i in range(24) if int(rd[i].item()) >= 0]
            if rv:
                lines.append(f"     River: {' '.join(rv)}")

            if p < 3:
                lines.append(f"  {'-'*66}")

        lines.append(f"{'='*70}")
        if state.terminated:
            lines.append(f"  GAME OVER | scores: {rs.score.tolist()} | rewards: {state.rewards.tolist()}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# HTML EXPORT
# ═══════════════════════════════════════════════════════════════

class HTMLRecorder:
    def __init__(self):
        self.steps = []

    def record(self, state, action, reward_text=""):
        board = TextBoard.render(state).replace("\n", "<br>")
        aname = _action_name(action) if action is not None else "init"
        self.steps.append({
            "step": state.step_count, "player": state.current_player,
            "action": aname, "reward": reward_text, "board": board,
        })

    def export(self, path="game_replay.html"):
        html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Mahjong Replay</title>
<style>
body{font-family:Consolas,monospace;background:#1a1a2e;color:#ccc;padding:20px}
.step{border:1px solid #333;margin:8px 0;padding:12px;border-radius:6px;background:#16213e}
.step-hdr{font-size:13px;color:#888;margin-bottom:6px}
.step-act{font-weight:bold;color:#e74c3c}
.step-board{font-size:12px;line-height:1.3;white-space:pre-wrap;background:#0f0f23;padding:8px;border-radius:4px}
.nav{position:sticky;top:10px;background:#16213e;padding:8px;border-radius:6px;margin-bottom:16px}
.nav button{margin:2px;padding:4px 12px;background:#2980b9;color:#fff;border:none;border-radius:4px;cursor:pointer}
.nav button:hover{background:#3498db}
</style></head><body><div class="nav">
<button onclick="showAll()">Show All</button>
<button onclick="showLast()">Last Step</button>
<span id="info"></span>
</div><div id="content">"""
        for i, s in enumerate(self.steps):
            html += f"""<div class="step" id="s{i}">
<div class="step-hdr">Step {s['step']} | P{s['player']} | <span class="step-act">{s['action']}</span> {s['reward']}</div>
<div class="step-board">{s['board']}</div>
</div>"""
        html += """</div>
<script>
document.getElementById('info').textContent=' | Steps: '+document.querySelectorAll('.step').length;
function showAll(){document.querySelectorAll('.step').forEach(s=>s.style.display='block')}
function showLast(){document.querySelectorAll('.step').forEach(s=>s.style.display='none');
var ss=document.querySelectorAll('.step');if(ss.length)ss[ss.length-1].style.display='block';ss[ss.length-1].scrollIntoView()}
</script></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML saved: {path}")


def _action_name(a):
    a = int(a) if isinstance(a, torch.Tensor) else a
    names = {71:"Tsumogiri",72:"Riichi",73:"Tsumo",74:"RON",75:"Pon",76:"Pon(red)",
             77:"OpenKan",84:"Pass",85:"Kyuushu",86:"Dummy"}
    if a in names: return names[a]
    if 0 <= a < 37: return f"Discard({_tn(a)})"
    if 37 <= a < 71: return f"Kan({_tn(a-37)})"
    if 78 <= a <= 83: return f"Chi({a})"
    return f"Action({a})"


# ═══════════════════════════════════════════════════════════════
# PLAY + VISUALIZE
# ═══════════════════════════════════════════════════════════════

def play_and_visualize(model_path=None, opponent="random", seed=42,
                       max_steps=500, render="text", out_html=None, device="cpu"):
    env = make_env("red_mahjong", round_mode="single", observe_type="dict")
    recorder = HTMLRecorder() if render == "html" else None

    agent_seat = seed % 4 if model_path else -1
    if model_path:
        net = ACNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        def agent_actor(state, rng=None, sample=False):
            obs = env.observe(state)
            obs_dev = {k: v.to(device) for k, v in obs.items()}
            with torch.no_grad():
                logits, _ = net(obs_dev)
                mask = state.legal_action_mask.to(device)
                logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
                if sample:
                    probs = torch.softmax(logits, dim=-1)
                    return int(torch.multinomial(probs.squeeze(0), 1).item())
                return int(torch.argmax(logits).item())
    else:
        agent_actor = None

    baseline = rule_based_player if opponent == "rule_based" else random_player
    gen = torch.Generator().manual_seed(seed)
    state = env.init(gen)
    if recorder: recorder.record(state, None)

    step = 0
    while not state.terminated and step < max_steps:
        cp = state.current_player
        action = agent_actor(state, gen) if cp == agent_seat else baseline(state, gen)
        state = env.step(state, action)

        rtext = ""
        if state.rewards.abs().sum() > 0:
            rtext = f"| reward={state.rewards.tolist()}"

        if render == "text":
            print(f"\n--- Step {state.step_count} | P{cp}: {_action_name(action)} {rtext} ---")
            print(TextBoard.render(state))

        if recorder: recorder.record(state, action, rtext)
        step += 1

    if recorder:
        path = out_html or f"game_replay_s{seed}.html"
        recorder.export(path)
    else:
        print(f"\nDONE | {state.step_count} steps | scores: {state.round_state.score.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mahjong game visualization")
    parser.add_argument("--model", default=None, dest="model_path", help="Path to trained .pt model")
    parser.add_argument("--opponent", default="random", choices=["random", "rule_based"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--render", default="text", choices=["text", "html"])
    parser.add_argument("--out_html", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    play_and_visualize(**vars(args))
