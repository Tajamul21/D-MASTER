N_GPUS=4
BATCH_SIZE=8
DATA_ROOT=/Path/to/Dataset/Root/Dir
OUTPUT_DIR=/Path/to/output/dir
CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26508 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset aiims \
--target_dataset cview \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-5 \
--lr_backbone 2e-6 \
--lr_linear_proj 2e-6 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode teaching \
--output_dir ${OUTPUT_DIR} \
--resume /path/to/cross_domain_mae model wts \
--epoch_retrain 10 \
--epoch_mae_decay 5 \
--alpha_dt 0.9 \
--max_dt 0.6 \
--teach_box_loss True

