# MDHSD-main


### Dataset
下载数据集到此路径下：
```python
/data
     /cifar
          /cifar-100-python
          /cifar-10-batches-py
     /tinyImageNet200
```


### Train
若训练本文方法MDHSD，则运行下面这条指令：
`sh ./runs/run_kd_ssl.sh`<br>
若训练对比算法，则运行下面这条指令：
`sh ./runs/run_kd_ssl_contrast.sh`
