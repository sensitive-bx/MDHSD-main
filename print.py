import torch

# 替换为你自己的模型路径
ckpt_path = '/home/cbx/SRD_5_nanguo_ls_dist_repvgg/SRD_ossl-main-5_难过_ls_dist_repvgg/Results/kdssl_False/student_model/O:tinS:repvgg_a0_A:repvgg_a1_T:repvgg_a2_cifar100_srd_kd_dist_r:1_a:0.0_b:1.0e:pytorch_Mon Nov 11 10:33:27 2024/repvgg_a0_best.pth'

# 加载 checkpoint
checkpoint = torch.load(ckpt_path, map_location='gpu')

# 打印所有键，看看有没有 'best_acc' 或类似的
print("Checkpoint keys:", checkpoint.keys())

# 如果你看到有 'best_acc' 或类似字段：
if 'best_acc' in checkpoint:
    print("Best Accuracy:", checkpoint['best_acc'])
elif 'acc' in checkpoint:
    print("Best Accuracy (acc):", checkpoint['acc'])
else:
    print("没有找到最佳准确率字段，你可能需要根据实际字段名修改代码")
