#!/bin/bash

BATCH_SIZE=64
DATA_ROOT=/home/tajamul/scratch/Domain_Adaptation/MRT/DATA
# OUTPUT_DIR_BASE=/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-froc/outputs/def-detr-base/Psuedo_labels/epoch_
LOG_FILE=/home/tajamul/scratch/Domain_Adaptation/MICCAI_2024/data_server/ddsm_full/froc_ddsm_val_full.txt  # Update this with the actual path for the log file

# Loop over checkpoint files from 0 to 100
for checkpoint in {35..100}
do
    OUTPUT_DIR=${OUTPUT_DIR_BASE}${checkpoint}

    CUDA_VISIBLE_DEVICES=6 python -u froc_mrt.py \
    --backbone resnet50 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_classes 2 \
    --data_root ${DATA_ROOT} \
    --source_dataset ddsm_full \
    --target_dataset ddsm_full \
    --eval_batch_size ${BATCH_SIZE} \
    --mode eval \
    --output_dir ${OUTPUT_DIR} \
    --resume /home/tajamul/scratch/Domain_Adaptation/MICCAI_2024/data_server/rsna_full/model_epoch_${checkpoint}.pth \
    >> ${LOG_FILE} 2>&1

    # Update the log file after each epoch
    echo "Epoch ${checkpoint} completed. Checkpoint results are in ${OUTPUT_DIR}" >> ${LOG_FILE}

done
