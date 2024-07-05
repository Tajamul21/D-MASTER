N_GPUS=6
BATCH_SIZE=18
DATA_ROOT=/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release/data
OUTPUT_DIR=./outputs/def-detr-base/sim2city/sim2city_check

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26504 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode cross_domain_mae \
--output_dir ${OUTPUT_DIR} \
--resume /home/tajamul/scratch/Domain_Adaptation/DA_expts/MRT/Sim2city/Source_only/model_best.pth \
