### 将超大文件分割成多个小的
# -*- coding:utf-8 -*-
from datetime import datetime
from progressbar import ProgressBar
import pandas as pd
 
def Main():
    source_dir = './train.txt'
 
    # 计数器
    flag = 0
 
    # 文件名
    name = 1
 
    # 存放数据
    dataList = []

    print("开始。。。。。")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
 
    with open(source_dir,'r',encoding='utf-8') as f_source:
        for line in f_source:
            flag += 1
            dataList.append(line)
            
            if flag == 50000000:
                print('正在处理第{}份数据。。。。。'.format(name))

                # 写成txt格式
                with open("./train_"+str(name)+".txt",'w+') as f_target:
                    for data in dataList:
                        f_target.write(data)
                # # 写成csv格式
                # df = pd.read_csv(target_dir+"train_"+str(name)+".txt",sep='\t',encoding='utf-8')
                # # print(df.head())
                # df.to_csv(target_dir+"train_"+str(name)+".csv",mode='w',encoding='utf-8',header=False,index=False)
                
                name += 1
                flag = 0
                dataList = []
                # break

                
    # 处理最后一批行数少于800万行的
    # 写成txt格式
    print('正在处理第{}份数据。。。。。'.format(name))
    with open("./train_"+str(name)+".txt",'w+') as f_target:
        for data in dataList:
            f_target.write(data)
    # # 写成csv格式
    # df = pd.read_csv(target_dir+"train_"+str(name)+".txt",sep='\t',encoding='utf-8')
    # df.to_csv(target_dir+"train_"+str(name)+".csv",mode='w',encoding='utf-8',header=False,index=False)
                
 
    print("完成。。。。。")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
 
if __name__ == "__main__":
    progress = ProgressBar()
    Main()
 
