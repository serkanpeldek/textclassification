class PathParams:
    def __init__(self, base_dir):
        self.paths = dict()
        self.set_paths(base_dir=base_dir)
    
    def set_paths(self, base_dir):
        self.paths["base_dir"] = base_dir
        self.paths["results"] = f"{base_dir}/results"
        self.paths["data_folder"] = f"{base_dir}/runs/data/new_nlp_case_final"
        self.paths["csv_tr"] = f"{self.paths["data_folder"]}/train.csv"
        self.paths["csv_test"] = f"{self.paths["data_folder"]}/test.csv"
        self.paths["csv_tr_p"] = f"{self.paths["data_folder"]}/train_p.csv"
        self.paths["csv_test_p"] = f"{self.paths["data_folder"]}/test_p.csv"