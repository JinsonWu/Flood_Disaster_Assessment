echo "training seresnext50 classification model with seeds 0-2"
python train50_cls_cce.py 0
python train50_cls_cce.py 1
python train50_cls_cce.py 2
python tune50_cls_cce.py 0
python tune50_cls_cce.py 1
python tune50_cls_cce.py 2

echo "training dpn92 classification model with seeds 0-2"
python train92_cls_cce.py 0
python train92_cls_cce.py 1
python train92_cls_cce.py 2
python tune92_cls_cce.py 0
python tune92_cls_cce.py 1
python tune92_cls_cce.py 2

echo "training resnet34 classification model with seeds 0-2"
python train34_cls.py 0
python train34_cls.py 1
python train34_cls.py 2
python tune34_cls.py 0
python tune34_cls.py 1
python tune34_cls.py 2

echo "training senet154 classification model with seeds 0-2"
python train154_cls_cce.py 0
python train154_cls_cce.py 1
python train154_cls_cce.py 2
python tune154_cls_cce.py 0
python tune154_cls_cce.py 1
python tune154_cls_cce.py 2

echo "All models trained!"