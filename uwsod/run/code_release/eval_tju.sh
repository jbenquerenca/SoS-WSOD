MODEL_PATH=output/TJU-Pedestrian-Traffic/model_final.pth
OUTPUT_DIR=output/sos_release_oicr_plus_tju/validation

python3 projects/WSL/tools/train_net_multi.py \
--num-gpus 1 \
--config-file projects/WSL/configs/Detection/code_release/dhd_traffic_oicr_plus.yaml \
--dist-url tcp://0.0.0.0:17346 --eval-only \
MODEL.WEIGHTS ${MODEL_PATH} \
OUTPUT_DIR ${OUTPUT_DIR} TEST.AUG.ENABLED False