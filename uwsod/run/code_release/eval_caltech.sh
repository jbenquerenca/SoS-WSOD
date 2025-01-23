MODEL_PATH=output/Caltech_Pedestrians/model_0089999.pth
OUTPUT_DIR=output/sos_release_oicr_plus_caltech/validation

python3 projects/WSL/tools/train_net_multi.py \
--num-gpus 1 \
--config-file projects/WSL/configs/Detection/code_release/caltech_oicr_plus.yaml \
--dist-url tcp://0.0.0.0:17346 --eval-only \
MODEL.WEIGHTS ${MODEL_PATH} \
OUTPUT_DIR ${OUTPUT_DIR} TEST.AUG.ENABLED False