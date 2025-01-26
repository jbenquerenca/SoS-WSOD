python train_net.py \
  --num-gpus 1 \
  --config configs/code_release/caltech_ssod.yaml \
  --dist-url tcp://0.0.0.0:21197 \
  MODEL.WEIGHTS ./output/caltech_baseline/model_0009999.pth \
  OUTPUT_DIR output/caltech_ssod/ \
  SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 DATALOADER.SUP_PERCENT 26.76254 TEST.VAL_LOSS False