### VQA-MED
# CUDA_VISIBLE_DEVICES=1 nohup python predict.py --encoder_type sam_vit_h --ObjectAwareModel_path ../weight/ObjectAwareModel.pt --encoder_path ../weight/sam_vit_h.pt --output_dir ../VQA-Med_sam --img_path ../VQA-MED/ImageClef-2019-VQA-Med-Training/Train_images &

### AQUA

# SAM
# CUDA_VISIBLE_DEVICES=6 nohup python predict.py --encoder_type sam_vit_h --ObjectAwareModel_path ../weight/ObjectAwareModel.pt --encoder_path ../weight/sam_vit_h.pt --output_dir ../AQUA_sam --img_path ../AQUA/SemArt/Images &

# Mobile SAM
CUDA_VISIBLE_DEVICES=7 nohup python predict.py --encoder_type efficientvit_l2 --ObjectAwareModel_path ../weight/ObjectAwareModel.pt --encoder_path ../weight/l2.pt --output_dir ../AQUA_sam_m --img_path ../AQUA/SemArt/Images &