PYTHONHASHSEED=42 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 \
torchrun --nproc_per_node 2 main_ipod.py \
--eval_batch_size 1 \
--num_eval_workers 2 \
--exp_name IPoD_eval \
--co3d_path ~/research/IPoD/dataset \
--one_class chair \
--n_queries 2048 \
--n_query_udf 16000 \
--run_val \
--resume ~/research/IPoD/ckpts/ipod_transformer_co3d.pth \
--save_pc
