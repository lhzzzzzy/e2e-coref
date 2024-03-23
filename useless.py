import os
import shutil

# 源文件夹路径
source_folder = '/home/lhzzzzzy/school_code/e2e-coref/dataset/new_validation'

# 目标文件夹路径
destination_folder = '/home/lhzzzzzy/school_code/e2e-coref/dataset/new_train'

# 要移动的文件名列表，每个元素包含原文件名和新文件名
files_to_move = []

for i in range(601,1277):
    files_to_move.append((f"{i}.json", f"{i-600+1664}.json"))
    

# 遍历要移动的文件列表
for original_name, new_name in files_to_move:
    
    source_file_path = os.path.join(source_folder, original_name)
    # 构建目标文件路径
    destination_file_path = os.path.join(destination_folder, new_name)
    
    new_name_pth = os.path.join(source_folder, new_name)
    # 重命名文件
    os.rename(source_file_path, new_name_pth)
    # 执行移动操作
    shutil.move(os.path.join(source_folder, new_name), destination_file_path) 