#cifar-100有标签 tiny-ImageNet无标签-以教师模型为resnet110,学生模型为resnet32为例
python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill kd -r 0.1 -a 0.9 -b 0 --ood tin

#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill hint -a 0 -b 100 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill attention -a 0 -b 1000 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill fsp -a 0 -b 50 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill nst -a 0 -b 50 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill factor  -a 0 -b 200 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill similarity -a 0 -b 3000 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill vid -a 0 -b 1 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill abound -a 0 -b 1 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill rkd -a 0 -b 1 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill pkt -a 0 -b 1 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill mlld -a 0 -b 1 --ood tin
#
##CTKD
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill kd -r 0.1 -a 0.9 -b 0 --have_mlp 1 --mlp_name 'global' --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill fam -a 0 -b 1 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill nkd -a 0 -b 1 --ood tin
#
#python train_student_contrast.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth  --model_s resnet32 --distill srd -a 0 -b 1 --ood tin