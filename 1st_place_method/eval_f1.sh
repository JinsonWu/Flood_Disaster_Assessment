#!/bin/bash
python predict34cls.py 0
python predict34cls.py 1
python predict34cls.py 2
python predict50cls.py 0
python predict50cls.py 1
python predict50cls.py 2
python predict92cls.py 0
python predict92cls.py 1
python predict92cls.py 2
python predict154cls.py 0
python predict154cls.py 1
python predict154cls.py 2
echo "F1-Score Evaluation is Completed!"