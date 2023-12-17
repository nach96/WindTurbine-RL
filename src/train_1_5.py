from model_parser import ModelParser

#Configuration Files
json_rel_path = "../model_hyperparams/params_model_1_5.json"
input_file_rel_path = "..\\FAST_cfg\\IEA-15-240-RWT-Monopile.fst"

mp = ModelParser(json_rel_path,input_file_rel_path)
mp.learn()
