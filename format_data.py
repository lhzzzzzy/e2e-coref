import os
import json
import re

folder_path = "dataset/new_train"
for item in os.listdir(folder_path):
    pth = os.path.join(folder_path,item)
    
    with open(pth, "r") as f:
        data = json.load(f)
        
        for i in range(len(data["sentence"])):
            cleaned_text = re.sub(r'\{.*?\}', '', data["sentence"][i])
            data["sentence"][i] = cleaned_text
    
    updated_json = json.dumps(data,ensure_ascii=False,indent=4)
    
    with open(pth,"w") as file:
        file.write(updated_json)