train(){
  python main_resnet.py \
  --data_path ./cifar-10 \
  --save ./checkpoint/ResNet56 \
  --use_function piecewise
}

train;