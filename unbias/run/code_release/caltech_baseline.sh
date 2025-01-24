python train_net.py \
--num-gpus 1 \
--config configs/code_release/caltech_baseline.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR ./output/caltech_baseline \
SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 1 SOLVER.IMG_PER_BATCH_UNLABEL 1 TEST.VAL_LOSS False