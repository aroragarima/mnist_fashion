echo "Training with best model : \n \tLearning rate 0.01 \n \tOptimiser RMSProp"

python3 train.py -lr 0.01 -s 128 -opt 2 --save_dir saved_models/ -aug 2 -train
