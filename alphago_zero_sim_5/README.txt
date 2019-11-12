Set parameters in utils/dump/constants.py
One can use Reload path to finetune previous models
we used the following sequence of training updating on the previous step:
1. 200 simulations, lr=0.1 for 40 epochs    20min/epoch
2. 200 simulations, lr=0.01 for 20 epochs	20min/epoch
3. 400 simulations, lr=0.05 for 20 epochs	35min/epoch
4. 400 simulations, lr=0.01 for 30 epochs	35min/epoch
