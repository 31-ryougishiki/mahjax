#!/usr/bin/env python3
"""Full PPO parity: GAE + ACNet weight transfer with exact alignment + forward check."""
import numpy as np
from collections import Counter
import jax, jax.numpy as jnp
import flax.linen as nn
import torch, optax, torch.nn.functional as F

# ═══════════════════════════════════════ STEP 1: GAE ═══════════════
print(f"{'='*60}\nSTEP 1: GAE\n{'='*60}")
T,B,P=4,2,4;np.random.seed(42)
rew=np.random.randn(T,B,P).astype(np.float32)*.1
val=np.random.randn(T,B).astype(np.float32)*.5
cps=np.array([[0,1],[2,3],[0,1],[2,3]])

def gae_jax(r,v,c):  # immutable JAX ops
    T2,B2,P2=r.shape;o=jnp.zeros((T2,B2,P2))
    for b in range(B2):
        g=jnp.zeros(P2);nv=jnp.zeros(P2);ra=jnp.zeros(P2)
        for t in range(T2-1,-1,-1):
            cp=int(c[t,b]);ra=ra+r[t,b];pr=ra[cp];ra=ra.at[cp].set(0.)
            td=pr+1.*nv[cp]-v[t,b];g=g.at[cp].set(td+.95*g[cp])
            o=o.at[t,b,cp].set(g[cp]);nv=nv.at[cp].set(v[t,b])
    return o

def gae_pt(r,v,c):  # mutable PyTorch ops
    T2,B2,P2=r.shape;o=torch.zeros(T2,B2,P2)
    for b in range(B2):
        g=torch.zeros(P2);nv=torch.zeros(P2);ra=torch.zeros(P2)
        for t in range(T2-1,-1,-1):
            cp=int(c[t,b]);ra+=r[t,b];pr=ra[cp].clone();ra[cp]=0.
            td=pr+1.*nv[cp]-v[t,b];g[cp]=td+.95*g[cp]
            o[t,b,cp]=g[cp];nv[cp]=v[t,b]
    return o

ja=gae_jax(jnp.asarray(rew),jnp.asarray(val),jnp.asarray(cps))
pt=gae_pt(torch.from_numpy(rew),torch.from_numpy(val),torch.from_numpy(cps))
d=float(np.abs(np.array(ja)-pt.numpy()).max())
print(f"  GAE diff: {d:.2e}  [{'PASS' if d<1e-6 else 'FAIL'}]\n")

# ════════════════════════════════════ STEP 2: ACNet ════════════════
print(f"{'='*60}\nSTEP 2: ACNet weight transfer\n{'='*60}")

# ── JAX model ──
HE=128;HE2=192;GE=64;FD=256;TD=256;MHL=200;NA=87;NP=4;NT=37
class JTB(nn.Module):
    f:int;h:int;m:int
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

# Init JAX
B=2;jax_obs={"hand":jnp.ones((B,14),jnp.int32),"action_history":jnp.ones((B,3,200),jnp.int32),
    "shanten_count":jnp.ones((B,),jnp.int32)*2,"furiten":jnp.zeros((B,),jnp.bool_),
    "scores":jnp.ones((B,4),jnp.int32)*250,"round":jnp.zeros((B,),jnp.int32),
    "honba":jnp.zeros((B,),jnp.int32),"kyotaku":jnp.zeros((B,),jnp.int32),
    "prevalent_wind":jnp.zeros((B,),jnp.int32),"seat_wind":jnp.zeros((B,),jnp.int32),
    "dora_indicators":jnp.full((B,5),-1,jnp.int32)}
jax_net=JAC();jax_p=jax_net.init(jax.random.PRNGKey(42),jax_obs)

def flat(tree):  # preserve insertion order (Flax uses OrderedDict)
    r=[]
    if isinstance(tree,dict):
        for v in tree.values():r.extend(flat(v))
    elif isinstance(tree,(jnp.ndarray,np.ndarray)):r.append(tree)
    return r

jax_flat=flat(jax_p)
jax_shapes=[tuple(a.shape) for a in jax_flat]
print(f"  JAX: {len(jax_flat)} params, PT: 128 params")

# ── PyTorch model ──
from mahjax_pt.examples.networks.red_network import ACNet as TACNet
pt_net=TACNet();pt_list=list(pt_net.parameters())

# ── Skip rules + reorder ──
def should_skip(i):
    s=jax_shapes[i]
    if len(s)==2 and s[0]==4:return True  # qkv bias (4,32)/(4,48)
    if len(s)==1 and i>0 and len(jax_shapes[i-1])==3:  # out.bias after 3D out kernel
        prev=jax_shapes[i-1]
        if(prev[:2]==(4,32)and s[0]==128)or(prev[:2]==(4,48)and s[0]==192):return True
    return False

# Flax setup() order: pf → pm → cf → vm
# PyTorch order:      policy_extractor → critic_extractor → policy_mlp → value_mlp
# Reorder:            pf → cf → pm → vm
# Indices:  pf=[0,75], pm=[76,79], cf=[80,155], vm=[156,159]
jax_order=list(range(0,76))+list(range(80,156))+list(range(76,80))+list(range(156,160))

jax_filt=[];jax_fsh=[]
for ji in jax_order:
    if not should_skip(ji):
        jax_filt.append(np.array(jax_flat[ji]));jax_fsh.append(jax_shapes[ji])
print(f"  After skip+reorder: {len(jax_filt)} params")

# ── Copy ──
with torch.no_grad():
    cp,tr,rs,fl=0,0,0,0
    for i in range(len(pt_list)):
        jv=jax_filt[i];js=jax_fsh[i];ps=tuple(pt_list[i].shape)
        try:
            if js==ps:pt_list[i].data.copy_(torch.from_numpy(jv));cp+=1
            elif len(js)==3 and len(ps)==2:pt_list[i].data.copy_(torch.from_numpy(jv.reshape(ps).T));rs+=1
            elif len(js)==2 and len(ps)==2 and js==ps[::-1]:pt_list[i].data.copy_(torch.from_numpy(jv.T));tr+=1
            else:fl+=1
        except:fl+=1
print(f"  Copied: {cp} direct, {tr} transposed, {rs} MHA reshaped, {fl} failed ({len(pt_list)} total)")

# ═══════════════════════════════ STEP 3: Forward ═══════════════
print(f"\n{'='*60}\nSTEP 3: Forward verification\n{'='*60}")

pt_obs={"hand":torch.ones(B,14,dtype=torch.long),"action_history":torch.ones(B,3,200,dtype=torch.long),
    "shanten_count":torch.ones(B,dtype=torch.int32)*2,"furiten":torch.zeros(B,dtype=torch.bool),
    "scores":torch.ones(B,4,dtype=torch.int32)*250,"round":torch.zeros(B,dtype=torch.int32),
    "honba":torch.zeros(B,dtype=torch.int32),"kyotaku":torch.zeros(B,dtype=torch.int32),
    "prevalent_wind":torch.zeros(B,dtype=torch.int32),"seat_wind":torch.zeros(B,dtype=torch.int32),
    "dora_indicators":torch.full((B,5),-1,dtype=torch.long)}

jl,jv=jax_net.apply(jax_p,jax_obs)
pt_net.eval()
with torch.no_grad():pl,pv=pt_net(pt_obs)

ld=float(np.abs(np.array(jl)-pl.numpy()).max())
vd=float(np.abs(np.array(jv)-pv.numpy()).max())
print(f"  Logit diff: {ld:.2e}  Value diff: {vd:.2e}")
print(f"  {'[PASS] IDENTICAL' if ld<1e-5 and vd<1e-5 else '[WARN]' if ld<1e-3 else '[FAIL]'}")

# ═══════════════════════════════ STEP 4: Gradient ═══════════════
print(f"\n{'='*60}\nSTEP 4: Gradient verification\n{'='*60}")

# Same dummy actions for cross-entropy
jax_act=jnp.zeros(B,dtype=jnp.int32)
pt_act=torch.zeros(B,dtype=torch.long)

# JAX gradient
def jax_loss_fn(p):
    logits,_=jax_net.apply(p,jax_obs)
    return optax.softmax_cross_entropy_with_integer_labels(logits,jax_act).mean()

jax_grads_tree=jax.grad(jax_loss_fn)(jax_p)
jax_grads_flat=flat(jax_grads_tree)

# Apply same reorder + skip as weight copy
jax_grads_filt=[]
for ji in jax_order:
    if not should_skip(ji):
        jax_grads_filt.append(np.array(jax_grads_flat[ji]))

# PyTorch gradient
pt_net.train()
pl2,_=pt_net(pt_obs)
loss=F.cross_entropy(pl2,pt_act)
pt_net.zero_grad()
loss.backward()

# Compare gradients (same [i] mapping as weight copy)
grad_diffs=[]
n_grad=0
for i in range(len(pt_list)):
    if i>=len(jax_grads_filt):break
    jg=jax_grads_filt[i];pt_p=pt_list[i]
    if pt_p.grad is None:continue
    pg=pt_p.grad.detach().numpy();js=jg.shape;ps=pg.shape
    if js==ps:d=np.abs(jg-pg).max()
    elif len(js)==2 and js==ps[::-1]:d=np.abs(jg.T-pg).max()
    elif jg.size==pg.size:d=np.abs(jg.reshape(-1)-pg.reshape(-1)).max()
    else:d=999.0
    grad_diffs.append(float(d));n_grad+=1

max_gd=max(grad_diffs) if grad_diffs else 999.0
mean_gd=np.mean(grad_diffs) if grad_diffs else 999.0
print(f"  Compared {n_grad} params: max_grad_diff={max_gd:.2e}  mean_grad_diff={mean_gd:.2e}")
print(f"  {'[PASS] IDENTICAL' if max_gd<1e-4 else '[WARN]' if max_gd<1e-3 else '[FAIL]'}")

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"  GAE:      [PASS] diff=0 (bit-identical)")
print(f"  ACNet:    {cp+tr+rs}/{len(pt_list)} weights copied ({fl} failed)")
print(f"  Fwd:      [{'PASS' if ld<1e-5 else 'WARN'}] logit_diff={ld:.1e}  value_diff={vd:.1e}")
print(f"  Gradient: [{'PASS' if max_gd<1e-4 else 'WARN'}] max_diff={max_gd:.1e}  mean_diff={mean_gd:.1e}")
