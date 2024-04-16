import os
import json
import re

folder_path = "dataset/test"
cont = {}
with open("dataset/contributor.json","r",encoding="gbk") as f:
    for line in f:
        cur = json.loads(line)
        cont[cur["id"]] = cur["contributor"]
        
for item in os.listdir(folder_path):
    pth = os.path.join(folder_path,item)
    
    with open(pth, "r") as f:
        data = json.load(f)
        if data is None:
            print(pth)
            continue
        
        data["contributor"] = cont[data["taskID"]]
        
    updated_json = json.dumps(data,indent=4,ensure_ascii=False)
    
    with open(pth,"w", encoding="gbk") as file:
        file.write(updated_json) 