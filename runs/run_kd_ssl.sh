

#cifar-100有标签 tiny-ImageNet无标签
python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_a ./save/models/resnet56_cifar100/cifar100_resnet56-f2eff4c8.pt --model_s resnet32 --distill srd_kd_dist -a 0 -b 1 --ood tin

#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_a ./save/models/resnet56_cifar100/cifar100_resnet56-f2eff4c8.pt --model_s resnet50 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
#python train_student.py --path_t ./save/models/resnet56_cifar100/cifar100_resnet56-f2eff4c8.pt --path_a ./save/models/resnet44_cifar100/cifar100_resnet44-ffe32858.pt --model_s resnet8 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
#python train_student.py --path_t ./save/models/vgg19_cifar100/cifar100_vgg19_bn-b98f7bd7.pt --path_a ./save/models/vgg13_cifar100/cifar100_vgg13_bn-5ebe5778.pt --model_s vgg11 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
#python train_student.py --path_t ./save/models/vgg16_cifar100/cifar100_vgg16_bn-7d8c4031.pt --path_a ./save/models/vgg13_cifar100/cifar100_vgg13_bn-5ebe5778.pt --model_s vgg8 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
#python train_student.py --path_t ./save/models/repvgg_a2_ciafr100/cifar100_repvgg_a2-8e71b1f8.pt --path_a ./save/models/repvgg_a1_cifar100/cifar100_repvgg_a1-c06b21a7.pt --model_s repvgg_a0 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_a ./save/models/resnet56_cifar100/cifar100_resnet56-f2eff4c8.pt --model_s mobilenetv2_x0_5 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
#python train_student.py --path_t ./save/models/resnet56_cifar100/cifar100_resnet56-f2eff4c8.pt --path_a ./save/models/resnet44_cifar100/cifar100_resnet44-ffe32858.pt --model_s wrn_16_1 --distill srd_kd_dist -a 0 -b 1 --ood tin
#
##cifar-10有标签 cifar-100无标签
#python train_student.py --path_t ./save/models/resnet44_cifar10/cifar10_resnet44-2a3cabcb.pt --path_a ./save/models/resnet32_cifar10/cifar10_resnet32-ef93fc4d.pt --model_s resnet20 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10
#
#python train_student.py --path_t ./save/models/resnet56_cifar10/cifar10_resnet56-187c023a.pt --path_a ./save/models/resnet32_cifar10/cifar10_resnet32-ef93fc4d.pt --model_s resnet8 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10
#
#python train_student.py --path_t ./save/models/resnet56_cifar10/cifar10_resnet56-187c023a.pt --path_a ./save/models/resnet44_cifar10/cifar10_resnet44-2a3cabcb.pt --model_s resnet20 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10
#
#python train_student.py --path_t ./save/models/resnet56_cifar10/cifar10_resnet56-187c023a.pt --path_a ./save/models/resnet44_cifar10/cifar10_resnet44-2a3cabcb.pt --model_s wrn_16_1 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10
#
#python train_student.py --path_t ./save/models/mobilenetv2_x1_4_cifar10/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt --path_a ./save/models/mobilenetv2_x1_0_cifar10/cifar10_mobilenetv2_x1_0-fe6a5b48.pt --model_s mobilenetv2_x0_5 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10
#
#python train_student.py --path_t ./save/models/vgg19_cifar10/cifar10_vgg19_bn-57191229.pt --path_a ./save/models/vgg16_cifar10/cifar10_vgg16_bn-6ee7ea24.pt --model_s vgg11 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10
#
#python train_student.py --path_t ./save/models/vgg16_cifar10/cifar10_vgg16_bn-6ee7ea24.pt --path_a ./save/models/vgg13_cifar10/cifar10_vgg13_bn-c01e4a43.pt --model_s vgg8 --distill srd_kd_dist -a 0 -b 1 --ood cifar100 --dataset cifar10