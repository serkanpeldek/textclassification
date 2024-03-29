import pandas as pd

from .hf_utils import train_distilbert
from .hf_utils import test_distilbert



def run_distilbert(df_train, df_test, data_col, target_col, epoch, save_dir, checkpoint_num, txt_save_dir):
    train_distilbert(df_train, data_col, target_col, save_dir, epoch)
    test_distilbert(df_test, data_col, target_col, save_dir, checkpoint_num, txt_save_dir)

