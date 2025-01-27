python train_net.py \
--num-gpus 1 \
--config configs/code_release/dhd_traffic_baseline.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR ./output/tju_baseline \
SOLVER.BASE_LR 0.005 SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 TEST.VAL_LOSS False