source /home/l30054496/cann/Ascend/cann-9.0.0/set_env.sh
source /home/l30054496/cann/Ascend/nnal/atb/9.0.0/atb/set_env.sh

export PYTHONPATH="/home/z30055003/mahjax:${PYTHONPATH}"

python /home/z30055003/mahjax/mahjax_pt/examples/collect_offline_data.py \
    --num_samples 2000 --num_envs 4