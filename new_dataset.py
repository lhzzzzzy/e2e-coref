import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
import json
import os

# dc = {}
# with open("dataset/raw_data.txt","r",encoding="gbk",) as f:
#     for line in f:
#         line = line.strip()

#         if len(line) == 0:
#             continue
        
#         split_data = line.split("  ")

#         # 去除每个分割后的数据中的反斜杠及之后的数据
#         cleaned_data = [data.split("/")[0] for data in split_data]
#         # 打印处理后的结果
#         dc[cleaned_data[0]] = cleaned_data[1:]

# with open("dataset/raw_data.json","w") as f:
#     json.dump(dc, f, ensure_ascii=False, indent=4)

with open("dataset/raw_data.json", "r") as f:
        raw_data = json.load(f)


cur_sentences = set()
prefix_len = 0
tot_sentence = []
last_sentence_len = 0
spans = []
cnt = 0
for root, dirs, files in os.walk("dataset/train"):
    for file_name in sorted(files):
        file_path = os.path.join(root, file_name)
        
        with open(file_path, "r") as f:
            cur_data = json.load(f)
            
            if cur_data is None:
                continue
            if "0" not in cur_data:
                continue

            if cur_data["0"]["id"] not in cur_sentences:
                train_elem = {"sentence": tot_sentence, "spans":spans}
                with open(f"dataset/new_train/{cnt}.json","w") as f:
                    json.dump(train_elem, f, ensure_ascii=False, indent=4)

                cur_sentences = set()
                prefix_len = 0
                tot_sentence = []
                last_sentence_len = 0
                cnt += 1
                spans = []
            
            cur_span = {"pre_spans":[]}
            for i in range(cur_data["antecedentNum:"]):
                if cur_data[str(i)]["id"] not in cur_sentences:
                    cur_sentences.add(cur_data[str(i)]["id"])
                    prefix_len += last_sentence_len
                    # if cur_data[str(i)]["id"]=='19980127-09-001-033':
                        #  continue
                    last_sentence_len = len(raw_data[cur_data[str(i)]["id"]])
                    tot_sentence.extend(raw_data[cur_data[str(i)]["id"]])
                cur_span["pre_spans"].append({
                     "begin":cur_data[str(i)]["indexFront"]+prefix_len,
                     "end":cur_data[str(i)]["indexBehind"]+prefix_len})
            
            if cur_data["pronoun"]["id"] not in cur_sentences:
                    cur_sentences.add(cur_data["pronoun"]["id"])
                    prefix_len += last_sentence_len
                    last_sentence_len = len(raw_data[cur_data["pronoun"]["id"]])
                    tot_sentence.extend(raw_data[cur_data["pronoun"]["id"]])
            cur_span["span"] = {
                     "begin":cur_data["pronoun"]["indexFront"]+prefix_len,
                     "end":cur_data["pronoun"]["indexBehind"]+prefix_len
                     }
            
            spans.append(cur_span)



# if __name__ == "__main__":
#     with open("dataset/new_train/99.json", "r") as f:
#         data = json.load(f)

#         spans = data["spans"][1]
#         print(data["sentence"][spans["pre_spans"][0]["begin"]:spans["pre_spans"][0]["end"]+1])
#         print(data["sentence"][spans["span"]["begin"]:spans["span"]["begin"]+1])