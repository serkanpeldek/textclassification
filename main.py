import pandas as pd
from sklearn.pipeline import Pipeline

from utils.preprocess import textProcessor1
from utils.params import PathParams
from utils.loaders import DFLoader
from utils.tfidf_clf import run_tfidfclf
from utils.tfidf_cluster import run_tfidfculster
from utils.distilbert_clf import run_distilbert

if __name__ == "__main__":
    """
    PLEASE SEE SOLUTION.md

    """
    base_dir = "C:/Users/serka/OneDrive/Belgeler/repos/SerkanPeldek"
    pp = PathParams(base_dir=base_dir)
    dfLoader = DFLoader(pp, textProcessor1)
    df_train, df_test = dfLoader.load("processed")

    print(df_train.info(verbose=True, show_counts=True))
    print(df_test.info(verbose=True, show_counts=True))
    
   
    print(df_train.shape, df_test.shape)
    print("train cols:", df_train.columns, "test cols:", df_test.columns)
    print("num of class:", len(df_train["category"].unique()))
    
    # TASK 1
    # run_tfidfclf(df_train, df_test, pp)
    
    run_distilbert(df_train, 
                   df_test, 
                   data_col="product_text", 
                   target_col="category", 
                   epoch=10, 
                   save_dir="runs/hfdistilbert", 
                   checkpoint_num=56980,
                   txt_save_dir="results")
    
    # TASK 2
    run_tfidfculster(df_train, 
                     data_col="product_text", 
                     target_col="category",  
                     num_categories=5, 
                     save_dir="results", 
                     filename="output_categories")
    
   

    

    