import os

current = os.getcwd()

import os

target_folder = "TKDTS"

tkdts_path = os.path.abspath(os.getcwd()).split('TKDTS')[0] + 'TKDTS\model\\rf_model.sav'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(tkdts_path)
print(BASE_DIR.strip("src\python") + '\model\\rf_model.sav')