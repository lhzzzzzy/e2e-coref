import os
import json

folder_path="dataset/validation"

with open("dataset/raw_data.json","r") as f:
    data = json.load(f)

for item in os.listdir(folder_path):
    pth = os.path.join(folder_path,item)
    
    with open(pth,"r",encoding="gbk") as f:
        cur_data = json.load(f)
        if cur_data == None:
            continue
        
        sent_id = cur_data["pronoun"]["id"]
        if sent_id != '19980127-09-001-033':
            sent = data[sent_id]
            cur_data[sent_id] = sent
        for i in range(cur_data["antecedentNum"]):
            
            sent_id = cur_data[str(i)]["id"]
            if sent_id == '19980127-09-001-033':
                print(pth)
                break
            sent = data[sent_id]
            
            cur_data[sent_id] = sent
            
    updated_json = json.dumps(cur_data,indent=4,ensure_ascii=False)
    
    with open(pth,"w", encoding="gbk") as file:
        file.write(updated_json)