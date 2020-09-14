export NGPU=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --config-file ./configs/mobilenext_ssd320_coco.yaml \
    SOLVER.WARMUP_FACTOR 0.1333 \
    SOLVER.WARMUP_ITERS 2000
