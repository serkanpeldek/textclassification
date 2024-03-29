import pandas as pd
class DFLoader:
    def __init__(self, pp, textProcessor):
        self.pp = pp
        self.textProcessor = textProcessor
    
    def load(self, name, processed=False):
        if name == "original":
            return self.load_original(processed)
        elif name == "processed":
            return self.load_processed()
        else:
            raise ValueError("Name should be one of 'original', 'processed'")
    
    def load_original(self, processed=False):
        df_train = pd.read_csv(self.pp.paths["csv_tr"], sep=chr(1))
        df_test = pd.read_csv(self.pp.paths["csv_test"], sep=chr(1))
        if processed:
            print("Train text processing start")
            df_train["product_text"] = df_train["product_text"].apply(self.textProcessor.process)
            print("Text text processing start")
            df_test["product_text"] = df_test["product_text"].apply(self.textProcessor.process)

            df_train.to_csv(self.pp.paths["csv_tr_p"], sep=chr(1))
            df_test.to_csv(self.pp.paths["csv_test_p"], sep=chr(1))
        
        df_train.dropna(subset=['product_text', 'category'], inplace=True)
        df_test.dropna(subset=['product_text', 'category'], inplace=True)
        return df_train, df_test

    
    def load_processed(self):
        df_train = pd.read_csv(self.pp.paths["csv_tr_p"], sep=chr(1))
        df_test = pd.read_csv(self.pp.paths["csv_test_p"], sep=chr(1))
        df_train.dropna(subset=['product_text', 'category'], inplace=True)
        df_test.dropna(subset=['product_text', 'category'], inplace=True)
        return df_train, df_test