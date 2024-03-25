python fine_tunning.py \
--train_dataset /mnt/nasquatch/data/2p/hakan/imgs_totrain.txt \
--validation_dataset /mnt/nasquatch/data/2p/hakan/imgs_totest.txt \
--lr 0.01 \
--epochs 100 \
--num_class 2 \
--lr_scheduler plateau \
--model_name resnet \
--dropout 0.0
