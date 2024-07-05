
BATCH_SIZE=16
DATA_ROOT=/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release/data
OUTPUT_DIR=./outputs/def-detr-base/sim2city/cross_domain_mae


CUDA_VISIBLE_DEVICES=7 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume /home/tajamul/scratch/def-detr-base-sim2city-teaching-params.pth

