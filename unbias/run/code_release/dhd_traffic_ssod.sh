python train_net.py \
  --num-gpus 1 \
  --config configs/code_release/dhd_traffic_ssod.yaml \
  --dist-url tcp://0.0.0.0:21197 \
  MODEL.WEIGHTS ./output/tju_baseline/model_0007999.pth \
  OUTPUT_DIR output/tju_ssod \
  SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 DATALOADER.SUP_PERCENT 16.03864 TEST.VAL_LOSS False