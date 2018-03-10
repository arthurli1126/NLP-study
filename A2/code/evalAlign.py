from decode import *
import numpy as np
from math import log
from preprocess import *
from lm_train import *
from log_prob import *
from align_ibm1 import *


train_dir = "../Hansard/Training/"
test_dir = "../Hansard/Testing/"
fn_lme="./hansard_eng.pickle"
fn_lmf="./hansard_fre.pickle"
testF="../Hansard/Testing/Task5.f"