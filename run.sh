export NGPU=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --config-file ./configs/mobilenext_ssd320_voc0712.yaml \
    --eval_step 10000 \
    SOLVER.WARMUP_FACTOR 0.1333 \
    SOLVER.WARMUP_ITERS 1000
