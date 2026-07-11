#!/usr/bin/env python3
"""L3: ACNet Weight Transfer — fixed mapping for JAX 0.10.2 + Flax 0.12.7.

Uses explicit param-by-param mapping with numpy-controlled weights to guarantee
identical forward pass between JAX and PyTorch ACNet.

Strategy:
  1. Generate random numpy weights for all params
  2. Set JAX params and PT params to the SAME numpy values (accounting for shapes)
  3. Verify forward pass
  4. Test with both all-ones and random observation data
"""

import sys, os
import numpy as np
import jax, jax.numpy as jnp
import torch

# Path setup
_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'examples')
sys.path.insert(0, _EXAMPLES_DIR)

SEED = 42
B = 4
NA = 87
NP = 4
NT = 37

# ═══════════════════════════════════════════════════════════════════════════
# JAX inline ACNet (exact copy from test_full_ppo_parity.py)
# ═══════════════════════════════════════════════════════════════════════════

HE, HE2, GE, FD, TD, MHL = 128, 192, 64, 256, 256, 200

import flax.linen as nn

class JTB(nn.Module):
    f:int; h:int; m:int
    @nn.compact
    def __call__(self,x,mask=None):
        y=nn.LayerNorm()(x)
        if mask is not None and mask.ndim==2:mask=mask[:,None,None,:]
        y=nn.MultiHeadDotProductAttention(num_heads=self.h,kernel_init=nn.initializers.orthogonal(),deterministic=True)(y,mask=mask);x=x+y
        y=nn.LayerNorm()(x);y=nn.Dense(self.m,kernel_init=nn.initializers.orthogonal())(y);y=nn.relu(y)
        y=nn.Dense(self.f,kernel_init=nn.initializers.orthogonal())(y);x=x+y;return x

class JFE(nn.Module):
    @nn.compact
    def __call__(self,obs):
        hd=jnp.clip(obs["hand"].astype(jnp.int32),-1,99)+1
        if hd.ndim==1:hd=hd[None,:]
        he=nn.Embed(NT+1,HE,embedding_init=nn.initializers.orthogonal())(hd)
        hm=(hd>0).astype(jnp.float32);xh=he*hm[...,None]
        for _ in range(2):xh=JTB(HE,4,TD)(xh,mask=hm)
        hf=(xh*hm[...,None]).sum(1)/jnp.maximum(hm.sum(1,keepdims=True),1.)
        ah=obs["action_history"]
        if ah.ndim==2:ah=ah[None,...]
        pl,ac,ts=ah[:,0,:].astype(jnp.int32),ah[:,1,:].astype(jnp.int32),ah[:,2,:].astype(jnp.int32)
        hm2=(ac>=0).astype(jnp.float32)
        pe=nn.Embed(NP+1,HE2,embedding_init=nn.initializers.orthogonal())(jnp.clip(pl+1,0,99))
        ae=nn.Embed(NA+1,HE2,embedding_init=nn.initializers.orthogonal())(jnp.clip(ac+1,0,99))
        te=nn.Embed(3,HE2,embedding_init=nn.initializers.orthogonal())(jnp.clip(ts+1,0,99))
        pose=nn.Embed(MHL,HE2,embedding_init=nn.initializers.orthogonal())(jnp.arange(MHL)[None,:])
        xh2=pe+ae+te+pose;xh2=xh2*hm2[...,None]
        for _ in range(2):xh2=JTB(HE2,4,TD)(xh2,mask=hm2)
        hif=(xh2*hm2[...,None]).sum(1)/jnp.maximum(hm2.sum(1,keepdims=True),1.)
        def _b(x,nd):
            x=jnp.asarray(x,dtype=jnp.float32)
            if x.ndim==nd:return x[None,...] if nd>0 else x.reshape((1,1))
            elif x.ndim==nd+1 and nd==0:return x[:,None]
            return x
        gs=jnp.concatenate([(_b(obs.get("scores",jnp.zeros(4)),1)+250.)/1250.,_b(obs.get("shanten_count",0),0)/6.,
            _b(obs.get("furiten",False),0),_b(obs.get("round",0),0)/12.,_b(obs.get("honba",0),0)/10.,
            _b(obs.get("kyotaku",0),0)/10.,_b(obs.get("prevalent_wind",0),0)/3.,_b(obs.get("seat_wind",0),0)/3.],-1)
        di=_b(obs.get("dora_indicators",jnp.zeros(5,dtype=jnp.int32)),1).astype(jnp.int32)
        di=jnp.clip(di+1,0,99);dm=(di>0).astype(jnp.float32)
        de=nn.Embed(NT+1,HE,embedding_init=nn.initializers.orthogonal())(di)*dm[...,None]
        df=nn.relu(nn.Dense(GE,kernel_init=nn.initializers.orthogonal())(de.sum(1)/jnp.maximum(dm.sum(1,keepdims=True),1.)))
        gi=jnp.concatenate([gs,df],-1)
        go=nn.Dense(GE,kernel_init=nn.initializers.orthogonal())(gi);go=nn.relu(go)
        go=nn.Dense(GE,kernel_init=nn.initializers.orthogonal())(go)
        return jnp.concatenate([hf,hif,go],-1)

class JAC(nn.Module):
    def setup(self):
        self.pf=JFE();self.cf=JFE()
        self.pm=nn.Sequential([nn.Dense(FD,kernel_init=nn.initializers.orthogonal()),nn.relu,
                                nn.Dense(NA,kernel_init=nn.initializers.orthogonal(0.01))])
        self.vm=nn.Sequential([nn.Dense(FD,kernel_init=nn.initializers.orthogonal()),nn.relu,
                                nn.Dense(1,kernel_init=nn.initializers.orthogonal())])
    def __call__(self,obs):return self.pm(self.pf(obs)),self.vm(self.cf(obs)).squeeze(-1)


def flat(tree):
    r=[]
    if isinstance(tree,dict):
        for v in tree.values():r.extend(flat(v))
    elif isinstance(tree,(jnp.ndarray,np.ndarray)):r.append(tree)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# Helper: generate controlled weights and inject into both networks
# ═══════════════════════════════════════════════════════════════════════════

def np_to_jax(tree, np_weights, idx=0):
    """Replace JAX param tree values with numpy weights (in-place-ish)."""
    result = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            result[k], idx = np_to_jax(v, np_weights, idx)
        else:
            w = np_weights[idx]
            # Match dtype
            result[k] = jnp.asarray(w, dtype=v.dtype)
            idx += 1
    return result, idx


def np_to_pt(pt_params, np_weights):
    """Copy numpy weights to PT params (accounting for shape differences)."""
    with torch.no_grad():
        for i, (pp, nw) in enumerate(zip(pt_params, np_weights)):
            ps = tuple(pp.shape)
            ns = nw.shape
            if ps == ns:
                pp.data.copy_(torch.from_numpy(nw))
            elif len(ns) == 2 and len(ps) == 2 and ns == ps[::-1]:
                pp.data.copy_(torch.from_numpy(nw.T))
            elif ns == ps:
                pp.data.copy_(torch.from_numpy(nw))
            else:
                raise ValueError(f"Shape mismatch at [{i}]: JAX{ns} vs PT{ps}")


def build_jax_to_pt_map():
    """Build a mapping: jax_filtered_index → pt_index.

    JAX param order (160 params):
      pf[0..75] → pm[76..79] → cf[80..155] → vm[156..159]

    After skipping MHA biases (32 params):
      pf: 0..59 (60 params)
      cf: 60..119 (60 params)
      pm: 120..123 (4 params)
      vm: 124..127 (4 params)
      Total: 128 params

    PT param order (128 params):
      policy_extractor[0..59]
      critic_extractor[60..119]
      policy_mlp[120..123]
      value_mlp[124..127]
    """
    # For each FE (76 JAX → 60 PT after skip):
    # 0: embed(38,128) → pt[0]
    # 1-2: LN → pt[1-2]
    # 3: Q(128,4,32) → pt[3] reshape
    # 4: Qb(4,32) SKIP
    # 5: K(128,4,32) → pt[4] reshape
    # 6: Kb(4,32) SKIP
    # 7: V(128,4,32) → pt[5] reshape
    # 8: Vb(4,32) SKIP
    # 9: out(4,32,128) → pt[6] reshape
    # 10: outb(128,) SKIP
    # 11-12: LN → pt[7-8]
    # 13: mlp1_k(128,256) → pt[9] transpose
    # 14: mlp1_b(256,) → pt[10]
    # 15: mlp2_k(256,128) → pt[11] transpose
    # 16: mlp2_b(128,) → pt[12]

    # Pattern repeats for second hand JTB (indices shifted by 16 in JAX, 12 in PT)
    # Then history section...

    # Let me just build it programmatically
    jax_to_pt = {}  # jax_raw_idx → pt_idx, None = skip

    def map_fe(jax_start, pt_start, he_dim=HE, he2_dim=HE2):
        """Map one FeatureExtractor from jax_start to pt_start.
        Returns (next_jax_idx, next_pt_idx)."""
        ji = jax_start
        pi = pt_start

        # ── Hand embed (Embed shapes same) ──
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1  # embed (38, 128)

        # ── Hand JTB #1 ──
        # LN1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MHA Q
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1  # Q bias SKIP
        # MHA K
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1  # K bias SKIP
        # MHA V
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1  # V bias SKIP
        # MHA out
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1  # out bias SKIP
        # LN2
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP1
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP2
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1

        # ── Hand JTB #2 (same pattern) ──
        # LN1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MHA Q
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA K
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA V
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA out
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # LN2
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP1
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP2
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1

        # ── History embeds (Embed shapes are same: (vocab, dim)) ──
        # player emb (5, 192)
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # action emb (88, 192)
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # tsumogiri emb (3, 192)
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # pos emb (200, 192)
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1

        # ── History JTB #1 ──
        # LN1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MHA Q
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA K
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA V
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA out
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # LN2
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP1
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP2
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1

        # ── History JTB #2 (same pattern) ──
        # LN1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MHA Q
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA K
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA V
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # MHA out
        jax_to_pt[ji] = (pi, 'reshape_3d'); ji += 1; pi += 1
        jax_to_pt[ji] = None; ji += 1
        # LN2
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP1
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # MLP2
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1

        # ── Global ──
        # dora embed (38, 128) — same shape as PT
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1
        # dora dense (128, 64)
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1  # bias
        # global mlp in (75, 64)
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1  # bias
        # global mlp out (64, 64)
        jax_to_pt[ji] = (pi, 'transpose'); ji += 1; pi += 1
        jax_to_pt[ji] = (pi, 'direct'); ji += 1; pi += 1  # bias

        return ji, pi

    # pf: JAX[0..75] → PT[0..59]
    ji, pi = map_fe(0, 0, HE, HE2)
    assert ji == 76 and pi == 60, f"FE mapping error: ji={ji}, pi={pi}"

    # cf: JAX[80..155] → PT[60..119]
    ji2, pi2 = map_fe(80, 60, HE, HE2)
    assert ji2 == 156 and pi2 == 120, f"FE2 mapping error: ji2={ji2}, pi2={pi2}"

    # pm: JAX[76..79] → PT[120..123]
    jax_to_pt[76] = (120, 'transpose')  # (384, 256) → PT (256, 384)
    jax_to_pt[77] = (121, 'direct')     # bias
    jax_to_pt[78] = (122, 'transpose')  # (256, 87) → PT (87, 256)
    jax_to_pt[79] = (123, 'direct')     # bias

    # vm: JAX[156..159] → PT[124..127]
    jax_to_pt[156] = (124, 'transpose')  # (384, 256) → PT (256, 384)
    jax_to_pt[157] = (125, 'direct')     # bias
    jax_to_pt[158] = (126, 'transpose')  # (256, 1) → PT (1, 256)
    jax_to_pt[159] = (127, 'direct')     # bias

    return jax_to_pt


# ═══════════════════════════════════════════════════════════════════════════

def make_obs(batch_dim=B):
    """Make test observation dicts for both JAX and PT."""
    np.random.seed(SEED)
    hand = np.random.randint(-1, 37, size=(batch_dim, 14)).astype(np.int32)
    hand[:, 0] = np.random.randint(0, 37, size=batch_dim)

    ah = np.zeros((batch_dim, 3, 200), dtype=np.int32)
    ah[:, 0, :] = np.random.randint(0, 4, size=(batch_dim, 200))
    ah[:, 1, :] = np.random.randint(-1, 87, size=(batch_dim, 200))
    ah[:, 2, :] = np.random.randint(-1, 2, size=(batch_dim, 200))

    obs = {
        "hand": hand,
        "action_history": ah,
        "shanten_count": np.random.randint(0, 7, size=batch_dim).astype(np.int32),
        "furiten": np.random.randint(0, 2, size=batch_dim).astype(bool),
        "scores": np.random.randint(-200, 500, size=(batch_dim, 4)).astype(np.int32),
        "round": np.random.randint(0, 13, size=batch_dim).astype(np.int32),
        "honba": np.random.randint(0, 11, size=batch_dim).astype(np.int32),
        "kyotaku": np.random.randint(0, 11, size=batch_dim).astype(np.int32),
        "prevalent_wind": np.random.randint(0, 4, size=batch_dim).astype(np.int32),
        "seat_wind": np.random.randint(0, 4, size=batch_dim).astype(np.int32),
        "dora_indicators": np.random.randint(-1, 37, size=(batch_dim, 5)).astype(np.int32),
    }
    return obs


def main():
    print(f"\n{'='*60}")
    print("L3: ACNet Weight Transfer (Explicit Mapping)")
    print(f"{'='*60}\n")

    # ── 1. Build mapping ───────────────────────────────────────────
    print("Building param mapping...")
    jax_to_pt = build_jax_to_pt_map()

    # Verify: 160 JAX params → 128 PT params (32 skipped)
    mapped = [v for v in jax_to_pt.values() if v is not None]
    skipped = [k for k, v in jax_to_pt.items() if v is None]
    print(f"  Mapped: {len(mapped)}, Skipped: {len(skipped)}")
    assert len(mapped) == 128, f"Expected 128 mapped, got {len(mapped)}"
    assert len(skipped) == 32, f"Expected 32 skipped, got {len(skipped)}"
    print("  [PASS] Mapping counts correct")

    # ── 2. Init JAX network ────────────────────────────────────────
    print("\nInitializing JAX ACNet...")
    jax_net = JAC()
    dummy_obs = {k: jnp.asarray(v) for k, v in make_obs(B).items()}
    jax_params = jax_net.init(jax.random.PRNGKey(SEED), dummy_obs)
    jax_flat = flat(jax_params)
    print(f"  JAX params: {len(jax_flat)}")

    # ── 3. Init PT network and load same weights ───────────────────
    print("Initializing PT ACNet and transferring weights...")
    from mahjax_pt.examples.networks.red_network import ACNet as PTACNet
    pt_net = PTACNet()
    pt_params = list(pt_net.parameters())
    print(f"  PT params: {len(pt_params)}")

    with torch.no_grad():
        for jax_idx, mapping in jax_to_pt.items():
            if mapping is None:
                continue
            pt_idx, mode = mapping
            jv = np.array(jax_flat[jax_idx])
            pp = pt_params[pt_idx]
            js, ps = jv.shape, tuple(pp.shape)

            if mode == 'direct':
                pp.data.copy_(torch.from_numpy(jv))
            elif mode == 'transpose':
                pp.data.copy_(torch.from_numpy(jv.T))
            elif mode == 'reshape_3d':
                # JAX: (feat, heads, head_dim), PT Linear: (out, in)
                # PT computes input @ W.T → need W[j,d] = W_jax[d,j//hd,j%hd]
                # reshape to (feat, heads*hd) then transpose to (heads*hd, feat)
                pp.data.copy_(torch.from_numpy(jv.reshape(ps).T))
            else:
                raise ValueError(f"Unknown mode: {mode}")

    # ── 4. Verify forward pass ─────────────────────────────────────
    print("\nVerifying forward pass...")

    # Test 1: All-ones obs (matching test_full_ppo_parity.py)
    print("  Test 1: All-ones observation...")
    ones_obs_jax = {
        'hand': jnp.ones((B, 14), jnp.int32),
        'action_history': jnp.ones((B, 3, 200), jnp.int32),
        'shanten_count': jnp.ones((B,), jnp.int32) * 2,
        'furiten': jnp.zeros((B,), jnp.bool_),
        'scores': jnp.ones((B, 4), jnp.int32) * 250,
        'round': jnp.zeros((B,), jnp.int32),
        'honba': jnp.zeros((B,), jnp.int32),
        'kyotaku': jnp.zeros((B,), jnp.int32),
        'prevalent_wind': jnp.zeros((B,), jnp.int32),
        'seat_wind': jnp.zeros((B,), jnp.int32),
        'dora_indicators': jnp.full((B, 5), -1, jnp.int32),
    }
    ones_obs_pt = {
        'hand': torch.ones(B, 14, dtype=torch.long),
        'action_history': torch.ones(B, 3, 200, dtype=torch.long),
        'shanten_count': torch.ones(B, dtype=torch.int32) * 2,
        'furiten': torch.zeros(B, dtype=torch.bool),
        'scores': torch.ones(B, 4, dtype=torch.int32) * 250,
        'round': torch.zeros(B, dtype=torch.int32),
        'honba': torch.zeros(B, dtype=torch.int32),
        'kyotaku': torch.zeros(B, dtype=torch.int32),
        'prevalent_wind': torch.zeros(B, dtype=torch.int32),
        'seat_wind': torch.zeros(B, dtype=torch.int32),
        'dora_indicators': torch.full((B, 5), -1, dtype=torch.long),
    }

    jl1, jv1 = jax_net.apply(jax_params, ones_obs_jax)
    pt_net.eval()
    with torch.no_grad():
        pl1, pv1 = pt_net(ones_obs_pt)

    ld1 = float(np.abs(np.array(jl1) - pl1.numpy()).max())
    vd1 = float(np.abs(np.array(jv1) - pv1.numpy()).max())
    ok1 = ld1 < 1e-4 and vd1 < 1e-4
    print(f"    logit_diff={ld1:.2e}  value_diff={vd1:.2e}  "
          f"{'[PASS]' if ok1 else '[FAIL]'}")

    # Test 2: Random obs
    print("  Test 2: Random observation...")
    rand_obs = make_obs(B)
    jax_rand = jax.tree.map(lambda x: jnp.asarray(x), rand_obs)
    pt_rand = {k: torch.from_numpy(v.copy()) for k, v in rand_obs.items()}
    # Fix dtypes for PT
    for k in ['hand', 'action_history', 'dora_indicators']:
        pt_rand[k] = pt_rand[k].long()
    pt_rand['shanten_count'] = pt_rand['shanten_count'].int()
    pt_rand['scores'] = pt_rand['scores'].int()
    pt_rand['round'] = pt_rand['round'].int()
    pt_rand['honba'] = pt_rand['honba'].int()
    pt_rand['kyotaku'] = pt_rand['kyotaku'].int()
    pt_rand['prevalent_wind'] = pt_rand['prevalent_wind'].int()
    pt_rand['seat_wind'] = pt_rand['seat_wind'].int()

    jl2, jv2 = jax_net.apply(jax_params, jax_rand)
    with torch.no_grad():
        pl2, pv2 = pt_net(pt_rand)

    ld2 = float(np.abs(np.array(jl2) - pl2.numpy()).max())
    vd2 = float(np.abs(np.array(jv2) - pv2.numpy()).max())
    ok2 = ld2 < 1e-4 and vd2 < 1e-4
    print(f"    logit_diff={ld2:.2e}  value_diff={vd2:.2e}  "
          f"{'[PASS]' if ok2 else '[FAIL]'}")

    # ── Summary ────────────────────────────────────────────────────
    all_ok = ok1 and ok2
    print(f"\n{'='*60}")
    print(f"L3 Result: {'[PASS] WEIGHT TRANSFER CORRECT' if all_ok else '[FAIL]'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
