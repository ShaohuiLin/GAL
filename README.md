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

In this code, you can run our Resnet-56 model on CIFAR10 dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.



### Run examples

The scripts of training and fine-tuning are provided  in the `run.sh`, please kindly uncomment the appropriate line in `run.sh` to execute the training and fine-tuning.

```shell
sh run.sh
```



**For training**, change the `teacher_dir` to the place where the pretrained model is located. 

```shell
# run.sh
python main.py --teacher_dir [pre-trained model dir]
```

The pruned model will be named `pruned.pt`



**For fine-tuning**, change the `refine` to the place where the pruned model is allowed to be fine-tuned. 

```shell
# run.sh
python finetune.py --refine [pruned model dir] 
```

You can set `--pruned` to reuse the `pruned.pt`. If you want to initiate weights randomly, just set  `--random`.



We also provide our [baseline](https://drive.google.com/open?id=1XHNxyFklGjvzNpTjzlkjpKc61-LLjt5T) model. Enjoy your training and testing!



## Tips

If you find any problems, please feel free to contact to the authors (shaohuilin007@gmail.com or Im.cqyan@gmail.com).