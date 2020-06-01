# Dynamic Model Pruning with Feedback

Paper Link : [**Dynamic Model Pruning with Feedback**](https://openreview.net/pdf?id=SJem8lSFwB) - ICLR2020

If you want to get information of hyperparameters, you should read appendix part of this paper

## Abstract

(1) Allowing dynamic allocation of the sparsity pattern

(2) Incorporating feedback signal to reactivate prematurely pruned weights

## Method

![Alt text](./resource/method.jpg)

![Alt text](./resource/figure.jpg)



## Run

```
python train.py --data cifar10 --datapath DATAPATH --model resnet20 --expname DPFTest
```

## Summary

Summary will be updated...