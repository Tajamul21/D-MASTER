N_GPUS=4
BATCH_SIZE=16
DATA_ROOT=/Path/to/Dataset/Root/Dir
OUTPUT_DIR=/Path/to/output/dir

 CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--dropout 0.1 \
--data_root ${DATA_ROOT} \
--source_dataset gbcnet \
--target_dataset gbcnet \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 100 \
--epoch_lr_drop 40 \
--mode single_domain \
--output_dir ${OUTPUT_DIR}
