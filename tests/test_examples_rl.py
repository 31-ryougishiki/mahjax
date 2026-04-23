import jax
import jax.numpy as jnp

from examples.networks.no_red_network import ACNet as NoRedACNet
from examples.networks.red_network import ACNet as RedACNet
from mahjax.core import make


def _sample_legal_action(state, key):
    logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits)


def test_example_networks_match_action_space() -> None:
    cases = [
        ("no_red_mahjong", NoRedACNet),
        ("red_mahjong", RedACNet),
    ]
    for env_name, network_cls in cases:
        env = make(env_name, one_round=True, observe_type="dict")
        state = env.init(jax.random.PRNGKey(0))
        obs = env.observe(state)
        network = network_cls()
        params = network.init(jax.random.PRNGKey(1), obs)
        logits, value = network.apply(params, obs)
        assert logits.shape == (1, state.legal_action_mask.shape[0])
        assert value.shape == (1,)


def test_one_round_envs_terminate() -> None:
    for env_name in ("no_red_mahjong", "red_mahjong"):
        env = make(env_name, one_round=True, observe_type="dict")
        state = env.init(jax.random.PRNGKey(0))
        key = jax.random.PRNGKey(1)
        for _ in range(500):
            if bool(state.terminated | state.truncated):
                break
            key, action_key = jax.random.split(key)
            action = _sample_legal_action(state, action_key)
            state = env.step(state, action)
        assert bool(state.terminated | state.truncated)
