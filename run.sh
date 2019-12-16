resnet(){
PRETRAINED_RESNET=/hdd1/hdd_B/ycq/pretrained/resnet_56.pt
MIU=1
LAMBDA=0.6
python main.py \
--teacher_dir $PRETRAINED_RESNET \
--arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU'_test' --gpus 2 \
--data_dir /hdd1/hdd_A/data/cifar10
}

vgg(){
PRETRAINED_VGG=[pre-trained model dir]
MIU=1e-1
LAMBDA=1e-3
python main.py \
--teacher_dir $PRETRAINED_VGG \
--arch vgg --teacher_model vgg_16_bn --student_model vgg_16_bn_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/vgg/lambda_'$LAMBDA'_miu_'$MIU 
} 

googlenet(){
PRETRAINED_GOOGLENET=[pre-trained model dir]
MIU=1e-1
LAMBDA=1e-2
PRINT=200
python main.py \
--teacher_dir $PRETRAINED_GOOGLENET \
--arch googlenet --teacher_model googlenet --student_model googlenet_sparse \
--lambda $LAMBDA --miu $MIU  \
--job_dir 'experiment/googlenet/lambda_'$LAMBDA'_miu_'$MIU \
--train_batch_size 64 --gpus 1
} 

densenet(){
PRETRAINED_DENSENET=[pre-trained model dir]
MIU=1e-1
LAMBDA=1e-2
python main.py \
--teacher_dir $PRETRAINED_DENSENET \
--arch densenet --teacher_model densenet_40 --student_model densenet_40_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/densenet/lambda_'$LAMBDA'_miu_'$MIU \
--train_batch_size 64 --gpus 2 
} 

finetune(){
ARCH=resnet
python finetune.py \
--arch $ARCH --lr 1e-5 \
--refine experiment/$ARCH/lambda_0.8_miu_1/resnet_pruned_11.pt \
--job_dir experiment/$ARCH/ft_lambda_0.8_miu_1_lr_1e-5/ \
--pruned 
}


# Training
# vgg;
resnet;
# googlenet;
# densenet;
# Fine-tuning
# finetune;