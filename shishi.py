import torch
import torch.nn as nn
import json
a = "本报/rz  北京/ns  １２月/t  ３０日/t  讯/Ng  〓/w  新华社/nt  记者/n  胡/nrf  晓梦/nrg  、/wu  本报/rz  记者/n  吴/nrf  亚明/nrg  报道/vt  ：/wm  新年/t  将/d  至/vt  ，/wd  [国务院/nt  侨务/n  办公室/n]nt  主任/n  郭/nrf  东坡/nrg  今天/t  通过/p  新闻/n  媒介/n  ，/wd  向/p  海外/s  同胞/n  和/c  国内/s  归侨/n  、/wu  侨眷/n  、/wu  侨务/n  工作者/n  发表/vt  新年/t  贺词/n  。/wj  他/rr  代表/vt  [中华人民共和国/ns  国务院/nt  侨务/n  办公室/n]nt  ，/wd  向/p  广大/b  海外/s  同胞/n  和/c  国内/s  归侨/n  、/wu  侨眷/n  、/wu  侨务/n  工作者/n  ，/wd  致以/vt  亲切/a  的/ud  问候/vn  和/c  美好/a  的/ud  祝愿/vn  。/wj"
b = a.split("  ")
print(b[21:27])
print(b[47])