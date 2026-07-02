source /home/l30054496/cann/Ascend/cann-9.0.0/set_env.sh
source /home/l30054496/cann/Ascend/nnal/atb/9.0.0/atb/set_env.sh

export PYTHONPATH="/home/z30055003/mahjax:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
python /home/z30055003/mahjax/mahjax_pt/examples/bc.py \
    --num_epochs 5 \
    --batch_size 1024 \
    --device npu:0 \
    --dataset_path /home/z30055003/mahjax/mahjax_pt/examples/offline_data/red_mahjong_offline_data.pkl