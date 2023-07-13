python3 fixmatch/train_fixmatch.py --dataset cifar10 \
                          --batch_size 64 \
                          --num_labels 250 \
                          --epochs 512 \
                          --eval_steps 128 \
                          --wd 0.0005  \
                          --seed 41 \
                          --save fixmatch/save \
                          --root data \