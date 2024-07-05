BATCH_SIZE=16
DATA_ROOT=/Path/to/Dataset/Root/Dir
OUTPUT_DIR=/Path/to/output/dir

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--data_root ${DATA_ROOT} \
--source_dataset aiims \
--target_dataset cview \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume /Path/to/best teacher wts \
--csv True