CUDA_VISIBLE_DEVICES=0 python3 projects/WSL/tools/train_net_multi.py --num-gpus 1 \
	--config-file projects/WSL/configs/Detection/code_release/detection_result_test_dhd_traffic.yaml \
	--dist-url tcp://0.0.0.0:8723 --resume --eval-only \
	OUTPUT_DIR output/sos_release_oicr_plus_dhd_traffic/