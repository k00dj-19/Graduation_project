ckpt_path=/home/rlaehdwls120/project/TR-DETR/test-tr-detr/hl-video_tef-weighted_caption_bsz32-2024_10_10_11_51_00/model_best.ckpt
eval_split_name=test
eval_path=data/highlight_${eval_split_name}_release.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python tr_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
