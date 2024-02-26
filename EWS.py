import torch
from graphics import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader   

path = r"C:\Users\yuroc\Desktop\Early-Warning-System\data"
df1 = pd.read_csv(path+r"\DIAGNOSES_ICD.csv")
df2 = pd.read_csv(path+"\ICUSTAYS.csv")


#icd9_code: 51881 => Acute repiratry failure
#find subject_id with 51881
icd9_code = "51881"

patients = df1.loc[df1['icd9_code'] == icd9_code].loc[:,'subject_id'].tolist
intimes = pd.to_datetime(df2['intime'])
outtimes = intimes+pd.Timedelta(hours=48)

print(outtimes)








def main():
    win = GraphWin("Paitents", 1000,1000)
    title = Text(Point(win.getWidth()/2, 200), "Early Warning System")
    title.draw(win)

    win.getMouse()
    win.close

main()