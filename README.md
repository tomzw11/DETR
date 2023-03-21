## DETR in MindSpore

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko

[[`arXiv`](https://arxiv.org/abs/2005.12872)] [[`BibTeX`](#citing-detr)]

This is an implementation of DETR in MindSpore based on [facebookresearch DETR](https://github.com/facebookresearch/detr).

## Training
Training DETR model for 300 epochs:
```bash
bash scripts/run_distribute_train_ascend.sh hccl_8p.json]
```
By default, we use 8 NPUs with total batch size 32 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
python eval.py 
	--output_dir=outputs 
	--mindrecord_dir="coco_mindrecord/" 
	--device_id=0 
	--device_target=Ascend 
	--resume="outputs/detr_epoch_299.ckpt" 
	--max_size=960
```

## Citing DETR
```BibTex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```