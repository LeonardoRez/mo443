#!/bin/bash

python code.py peppers.png input/i-1.txt 0 coded/result1.png
python uncode.py coded/result1.png 0 output/o-1.txt

python code.py baboon.png input/i-2.txt 0 coded/result2.png
python uncode.py coded/result2.png 0 output/o-2.txt

python code.py monalisa.png input/i-3.txt 2 coded/result3.png
python uncode.py coded/result3.png 2 output/o-3.txt

python code.py watch.png input/i-4.txt 1 coded/result4.png
python uncode.py coded/result4.png 1 output/o-4.txt

python code.py watch.png input/i-1.txt 6 coded/result5.png
python uncode.py coded/result5.png 6 output/o-5.txt


