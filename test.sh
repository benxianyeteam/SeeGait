# # **************** For SUSTECH1K/CCPG/CASIA-B Dataset ****************

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 "./opengait/main.py" --cfgs "./configs/SeeGait/Seegait_for_sustech1k.yaml"  --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=4 "./opengait/main.py" --cfgs "./configs/SeeGait/SeeGait_for_ccpg.yaml" --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=29502 --nproc_per_node=4 "./opengait/main.py" --cfgs "./configs/SeeGait/Seegait_for_casiab.yaml"  --phase test --log_to_file

