python main.py cifar10 -a resnet --layers 56 -C -g 0 --save prune.pth \
-P --prune-type unstructured --prune-freq 16 --prune-rate 0.9 --prune-imp L2 \
--epochs 300 --batch-size 128  --lr 0.2 --wd 1e-4 --nesterov --scheduler multistep --milestones 150 225 --gamma 0.1