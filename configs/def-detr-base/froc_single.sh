BATCH_SIZE=64
DATA_ROOT=/home/tajamul/scratch/Domain_Adaptation/MRT/DATA

CUDA_VISIBLE_DEVICES=7 python -u froc_metric.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--data_root ${DATA_ROOT} \
--source_dataset ddsm_full \
--target_dataset inbreast_split_full \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--resume /home/tajamul/scratch/Domain_Adaptation/MICCAI_2024/data_server/D_MASTER/MRT/ddsm_full/teaching/model_best.pth \

