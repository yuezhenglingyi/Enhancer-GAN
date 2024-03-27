

CUDA_VISIBLE_DEVICES="1" python b_Feedback_GAN.py  \
--lr 0.00001 \
--input_rate 0.7 \
--output_rate 0.3 \
--num_epochs 1000 \
--train_dir data/real_Sequence_test.txt \
--val_dir data/real_Sequence_val.txt \
--d_steps 5 \
--lamda 8 \
--gumbel 0 \
--retrain 1 \
--load_dir checkpoint/Vanilla-GAN/1689393684/model_path \
--iteration 200 \
--time "1689393684_5" \
--preds_cutoff 2.5

