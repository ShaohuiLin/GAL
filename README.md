# Towards Optimal Structured CNN Pruning via Generative Adversarial Learning(GAL)

PyTorch implementation for GAL.



![GAL-framework](https://user-images.githubusercontent.com/47294246/54805147-021eb500-4cb1-11e9-85ac-861ecbada3e1.png)

An illustration of GAL. Blue solid block, branch and channel elements are active, while red dotted elements are inactive and can be pruned since their corresponding scaling factors in the soft mask are 0.



## Abstract

Structured pruning of filters or neurons has received increased focus for compressing convolutional neural networks. Most existing methods rely on multi-stage optimizations in a layer-wise manner for iteratively pruning and retraining which may not be optimal and may be computation intensive. Besides, these methods are designed for pruning a specific structure, such as filter or block structures  without jointly pruning heterogeneous structures. In this paper, we propose an effective structured pruning approach that jointly prunes filters as well as other structures in an end-to-end manner. To accomplish this, we first introduce a soft mask to scale the output of these structures by defining a new objective function with sparsity regularization to align the output of baseline and network with this mask. We then effectively solve the optimization problem by generative adversarial learning (GAL), which learns a sparse soft mask in a label-free and an end-to-end manner. By forcing more scaling factors in the soft mask to zero, the fast iterative shrinkage-thresholding algorithm (FISTA) can be leveraged to fast and reliably remove the corresponding structures. Extensive experiments demonstrate the effectiveness of GAL on different datasets, including MNIST, CIFAR-10 and ImageNet ILSVRC 2012. For example, on ImageNet ILSVRC 2012, the pruned ResNet-50 achieves 10.88% Top-5 error and results in a factor of 3.7x speedup. This significantly outperforms state-of-the-art methods.



## Citation
If you find GAL useful in your research, please consider citing:

```
@inproceedings{lin2019towards,
  title     = {Towards Optimal Structured CNN Pruning via Generative Adversarial Learning},
  author    = {Lin, Shaohui and Ji, Rongrong and Yan, Chenqian and Zhang, Baochang and Cao, Liujuan and Ye, Qixiang and Huang, Feiyue and Doermann, David},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```



## Running Code

In this code, you can run our models on CIFAR10 dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.



### Run examples

The scripts of training and fine-tuning are provided  in the `run.sh`, please kindly uncomment the appropriate line in `run.sh` to execute the training and fine-tuning.

```shell
bash run.sh
```



**For training**, change the `teacher_dir` to the place where the pretrained model is located. 

```shell
# ResNet-56
MIU=1
LAMBDA=0.8
python main.py \
--teacher_dir [pre-trained model dir] \
--arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU
```

After training, checkpoints and loggers can be found in the `job_dir`, The pruned model of best performance will be named `[arch]_pruned_[pruned_num].pt`. For example: `resnet_pruned_11.pt`



**For fine-tuning**, change the `refine` to the place where the pruned model is allowed to be fine-tuned. 

```shell
# ResNet-56
python finetune.py \
--arch resnet --lr 1e-4 \
--refine experiment/resnet/lambda_0.8_miu_1/resnet_pruned_11.pt \
--job_dir experiment/resnet/ft_lambda_0.8_miu_1/ \
--pruned 
```

You can set `--pruned` to reuse the pruned model. 



**Evaluate our ImageNet model**

```bash
python test.py \
--target_model gal_05 \
--student_model resnet_50_sparse \
--dataset imagenet --data_dir $DATA_DIR \
--eval_batch_size 256 --job_dir ./checkpoints/gal_05.pth.tar
--test_only 
```



We also provide our baseline models below. Enjoy your training and testing!

Cifar10: | [ResNet56](https://drive.google.com/open?id=1XHNxyFklGjvzNpTjzlkjpKc61-LLjt5T) | [Vgg-16](https://drive.google.com/open?id=1pnMmLEWAUjVfqFUHanFlps6fSu10UYc1) | [DenseNet-40](https://drive.google.com/open?id=1Ev0SH14lWB5QuyPWLbbUEwGhVJ68tPkb) | [GoogleNet](https://drive.google.com/open?id=1tLZHnycQc4oAJhZ4JNYET_xHwR9mcdZX) |


ImageNet: [Baidu Wangpan](https://pan.baidu.com/s/1vvXcpEltrviWcKLN9kf5-w) (password:77zk) | [Google Drive](https://drive.google.com/drive/folders/1RB59_UI9fn-L3gqdROyJTVWtpTVTJk8s?usp=sharing)



## Tips

If you find any problems, please feel free to contact to the authors (shaohuilin007@gmail.com or Im.cqyan@gmail.com).