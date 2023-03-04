#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib/
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
mkdir -p /opt/ml/input/data/
mkdir -p /opt/ml/model/samples
python train_dreambooth.py  --attention xformers      \
            --class_data_dir /opt/ml/input/data/class_images/   \
            --class_prompt "photo of a man"    \
            --gradient_accumulation_steps 1  \
            --gradient_checkpointing True      \
            --instance_data_dir /opt/ml/input/data/images/    \
            --instance_prompt "photo of a zwx  man"    \
            --learning_rate 2e-06       \
            --lr_scheduler constant        \
            --lr_warmup_steps 0     \
            --max_train_steps 500   \
            --mixed_precision fp16       \
            --model_name  aws-trained-dreambooth-model    \
            --models_path /opt/ml/model/     \
            --not_cache_latents True   \
            --num_class_images 50       \
            --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5    \
            --prior_loss_weight 0.5    \
            --resolution 512    \
            --sample_batch_size 1    \
            --save_steps 500      \
            --save_use_epochs False   \
            --train_batch_size 1       \
            --train_text_encoder False    \
            --use_8bit_adam True    \
            --use_ema True   \
            --with_prior_preservation True