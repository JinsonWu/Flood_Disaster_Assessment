echo "training seresnext50 classification model with seeds 0-2"
python train50_cls_cce.py 0
python train50_cls_cce.py 1
python train50_cls_cce.py 2
python tune50_cls_cce.py 0
python tune50_cls_cce.py 1
python tune50_cls_cce.py 2

echo "Model Training Finished!"