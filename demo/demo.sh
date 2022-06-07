# CUDA_VISIBLE_DEVICES=1 python demo.py     --config-file ../configs/CondInst/our_R50.yaml     --input /home/dlhuang/AdelaiDet-CondInst/inftime_stone    --output /home/dlhuang/AdelaiDet-CondInst/detection_results/MS_R_50_1x_FPN96delP6P7delconv_FCOScls2box2BN_iou   --opts MODEL.WEIGHTS ../our/MS_R_50_1x_FPN96delP6P7delconv_FCOScls2box2BN_iou/model_final.pth

# CUDA_VISIBLE_DEVICES=1 python demo.py     --config-file ../configs/CondInst/our96.yaml     --input /home/dlhuang/AdelaiDet-CondInst/inftime_stone    --output /home/dlhuang/AdelaiDet-CondInst/detection_results/our96   --opts MODEL.WEIGHTS ../our/mobilenetv3small_FPN96delP6P7delconv_FCOScls2box2BN_iou/model_final.pth

# CUDA_VISIBLE_DEVICES=1 python demo.py     --config-file ../configs/CondInst/BN.yaml     --input /home/dlhuang/AdelaiDet-CondInst/inftime_stone    --output /home/dlhuang/AdelaiDet-CondInst/detection_results/inftime   --opts MODEL.WEIGHTS ../CondInst_ablations/BN/model_final.pth

CUDA_VISIBLE_DEVICES=1 python demo.py     --config-file ../configs/CondInst/our_R50.yaml     --input /home/dlhuang/AdelaiDet-CondInst/featuremap    --output /home/dlhuang/AdelaiDet-CondInst/detection_results/featuremap   --opts MODEL.WEIGHTS ../our/MS_R_50_1x_FPN96delP6P7delconv_FCOScls2box2BN_iou/model_final.pth

# CUDA_VISIBLE_DEVICES=1 python demo.py     --config-file ../configs/CondInst/our96.yaml     --input /home/dlhuang/AdelaiDet-CondInst/featuremap    --output /home/dlhuang/AdelaiDet-CondInst/detection_results/featuremap   --opts MODEL.WEIGHTS ../our/mobilenetv3small_FPN96delP6P7delconv_FCOScls2box2BN_iou/model_final.pth

# CUDA_VISIBLE_DEVICES=1 python demo.py     --config-file ../configs/CondInst/cls_box2.yaml     --input /home/dlhuang/AdelaiDet-CondInst/featuremap    --output /home/dlhuang/AdelaiDet-CondInst/detection_results/featuremap   --opts MODEL.WEIGHTS ../CondInst_ablations/cls_box2/model_final.pth