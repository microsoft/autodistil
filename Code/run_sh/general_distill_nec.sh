echo 'begin'

CORPUS_JSON=/home/dsi/dxu/Backups/Research_Server/Working/Pretrained-Language-Model/TinyBERT/output_wiki/corpus_jsonfile_for_general_KD/
TASK_NAME=MRPC
TASK_DIR=/home/dsi/dxu/Backups/Research_Server/Working/transformers/glue_data/MRPC
BERT_BASE_DIR=/home/dsi/dxu/Backups/Research_Server/Working/Pretrained-Language-Model/TinyBERT/pretrained_BERTs/BERT_base_uncased
SUPERNET_DIR=/home/dsi/dxu/Backups/Research_Server/Working/MSR/output_debug_KDNAS_2/
Depth_Mult_List=0.25,0.33333,0.5
Width_Mult_List=0.5,0.66667,1.0
Hidden_Mult_List=0.25,0.5,0.75,1.0
General_KD_Cache=/home/dsi/dxu/Backups/Research_Server/Working/MSR/output_general_KD_cache/

CUDA_VISIBLE_DEVICES=2,0,3 nohup python -u -m torch.distributed.launch --nproc_per_node=3 /home/dsi/dxu/Backups/Research_Server/Working/MSR/KDNAS/Code/run_wiki_NoAssist_RndSampl.py --pregenerated_data $CORPUS_JSON \
                          --model_type bert \
                          --task_name $TASK_NAME \
                          --do_train \
                          --data_dir $TASK_DIR \
                          --model_dir $BERT_BASE_DIR \
                          --output_dir $SUPERNET_DIR \
                          --max_seq_length 128 \
                          --learning_rate 2e-5 \
                          --per_gpu_train_batch_size 8 \
                          --num_train_epochs 1 \
                          --depth_mult_list $Depth_Mult_List \
                          --width_mult_list $Width_Mult_List \
                          --hidden_mult_list $Hidden_Mult_List \
                          --depth_lambda1 1.0 \
                          --depth_lambda2 1.0 \
                          --depth_lambda3 0.001 \
                          --depth_lambda4 0.0 \
                          --training_phase dynabert \
                          --logging_steps 2000 \
                          --output_cache_dir $General_KD_Cache \
                          --weight_decay 0.01 \
                          --num_train_epochs_wholeset 2 \
                          > /home/dsi/dxu/Backups/Research_Server/Working/Pretrained-Language-Model/TinyBERT/output_wiki/logs/train_supernet_wiki_distill.log 2>&1 &

echo 'end'

 