source /home/l30054496/cann/Ascend/cann-9.0.0/set_env.sh
source /home/l30054496/cann/Ascend/nnal/atb/9.0.0/atb/set_env.sh

export PYTHONPATH="/home/z30055003/mahjax:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
python /home/z30055003/mahjax/mahjax_pt/examples/ppo_with_reg.py \
    --num_envs 12 \
    --num_steps 128 \
    --total_timesteps 50000