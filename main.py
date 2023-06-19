import pickle
from tkinter import ttk
import tkinter as tk
import random
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

url = "./seeds_dataset.csv"

headers = []
headers.append("area A")
headers.append("permieter P")
headers.append("compactness C")
headers.append("length of kernel")
headers.append("width of kernel")
headers.append("asymmetry coefficient")
headers.append("length of kernel groove")
headers.append("class")


#rozmiar zbioru testowego
test_size = 0.10
#ziarno generatora pseudolosowego dla selekcji zbioru testowego
random_state = random.randint(0,9999)
print("seed: "+str(random_state))


root = tk.Tk()
root.resizable(False, False)

# DATA FRAME
df = pd.read_csv(url, names=headers, sep="\t")


# -----przy starcie wczytują sie dane z pliku tworzy knn z n=5 oraz ze zbioru wydzielany jest zbiór testowy
X = df.iloc[:, :-1].values

# Podział zbioru na atrybuty X i klasę y
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train,y_train)
#------------------------------------GUI=======================
root.title("knn s22408")


root.geometry("800x700")
label = tk.Label(text="ilosc sasiadow n")

entryN = tk.Entry()
arg1lab = tk.Label(text="area A")

arg2lab = tk.Label(text="permieter P")
arg3lab = tk.Label(text="compactness C")
arg4lab = tk.Label(text="length of kernel")
arg5lab = tk.Label(text="width of kernel")
arg6lab = tk.Label(text="asymmetry coefficient")
arg7lab = tk.Label(text="length of kernel groove")
classlab = tk.Label(text="class {1,2,3}")
entry1 = tk.Entry()

entry2 = tk.Entry()
entry3 = tk.Entry()
entry4 = tk.Entry()
entry5 = tk.Entry()
entry6 = tk.Entry()
entry7 = tk.Entry()
entryclass = tk.Entry()



#odświeża tabelkę w gui
def refreshTable():
    global df_rows, df, tree

    #usuwanie starych wierszy
    t = tree.get_children()
    for item in t:
        tree.delete(item)
    #wprowadzanie nagłówków
    for column in tree["columns"]:
        tree.heading(column, text=column)

    df_rows = df.to_numpy().tolist()

    #wprowadzenie wierszy
    for row in df_rows:
        tree.insert("", "end", values=row)

    for x in headers:
        tree.column(x, minwidth=0, width=80, stretch=False)

#BUTTON //wczytuje dane z pliku seeds_dataset.csv
def load():

    global df
    df = pd.read_csv(url, names=headers, sep="\t")
    print("loaded from file to dataFrame")
    refreshTable()

# BUTTON // buduje model knn na aktualnych danych z podanym n z zahardkodowaną metryką manhattan
def buildModel():
    global X, y, X_train, X_test, y_train, y_test, knn, entryN
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    n = int(entryN.get())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    knn = KNeighborsClassifier(n)
    print("knn built with n="+str(n))
    knn.fit(X_train, y_train)


#do funkcji addRecord i predict do pobierania danych z pól tekstowych
def getArgEntry():


    arg = []


    arg.append(float(entry1.get()))
    arg.append(float(entry2.get()))
    arg.append(float(entry3.get()))
    arg.append(float(entry4.get()))
    arg.append(float(entry5.get()))
    arg.append(float(entry6.get()))
    arg.append(float(entry7.get()))

    tmpX = []
    tmpX.append(arg)

    return tmpX

# BUTTON // dodaje rekord do aktualnych danych
def addRecord():


    global df
    ar = getArgEntry()[0]

    new_row = {'area A': ar[0], 'permieter P': ar[1], 'compactness C': ar[2],
               'length of kernel': ar[3], 'width of kernel' : ar[4],
               'asymmetry coefficient': ar[5], 'length of kernel groove': ar[6],
               'class': int(entryclass.get())}

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    xscroll.pack(side="bottom", fill="x")
    refreshTable()

# BUTTON // dokonuje predykcji z danych z pól testowych
def predict():
    s = str(knn.predict(getArgEntry()))
    resultString = "predicted class: "+s
    top = tk.Toplevel(root)
    top.geometry("100x100")
    top.title("Predaykcja")
    tk.Label(top, text=resultString).pack()

# BUTTON przeprowadza sprawdzian krzyżowy na zbudowanym modelu podaje wyniki w nowym oknie


def test():

    global random_state, knn, X_train, y_train, root

    rnd = random.randint(0,9999)
    k = KFold(n_splits=5, random_state=rnd, shuffle=True)
    scores = cross_val_score(knn, X_train, y_train, cv=k, scoring="accuracy")

    resultString = "Wyniki sprawdzianu krzyżowego:\n"
    resultString = resultString + str(scores)+'\n'+f"Średnia dokładność: {scores.mean()}"+"\nseed:"+str(rnd)

    top = tk.Toplevel(root)
    top.geometry("400x200")
    top.title("ocena modelu")
    tk.Label(top, text=resultString).pack()


# BUTTON // zapisuje aktualny model do pliku model.sav
def save():
    global knn
    filename = 'model.sav'
    pickle.dump(knn, open(filename, 'wb'))


# BUTTON // zastępuje aktualny model tym z pliku model.sav
def load_model():
    global knn
    filename = 'model.sav'
    knn = pickle.load(open(filename, 'rb'))

    refreshTable()


# BUTTON // czyści aktualne dane
def clear():

    global df
    df = df.head(0)
    xscroll.pack(side="bottom", fill="x")
    refreshTable()


# BUTTON // zapisuje aktualne dane do bazy danych
def saveToDataBase():

    global df


    conn = sqlite3.connect('database.db')

    df.to_sql("Dane", conn, if_exists='replace', index=False)

    conn.close()


# BUTTON // zastępuje aktualne dane tymi z bazy danych
def loadFromDataBase():

    global df

    conn = sqlite3.connect('database.db')

    df = pd.read_sql("select * from Dane", conn)

    refreshTable()


# GUI TABELKA--------treeView------------------------------------------

treeViewFrame = tk.LabelFrame(root,text="dane")
treeViewFrame.place(height=700, width=400, relx=0.4,rely=0)

tree = ttk.Treeview(treeViewFrame)

tree.place(relheight=1, relwidth=1)

tree["column"] = list(df.columns)
tree["show"] = "headings"



for column in tree["columns"]:

    tree.heading(column, text=column)

df_rows = df.to_numpy().tolist()

for row in df_rows:
    tree.insert("","end",values=row)


for x in headers:
    tree.column(x, minwidth=0, width=80,stretch=False)

xscroll = tk.Scrollbar(treeViewFrame, orient="horizontal", command=tree.xview())
yscroll = tk.Scrollbar(treeViewFrame, orient="vertical", command=tree.yview())

tree.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

yscroll.pack(side="right",fill="y")
xscroll.pack(side="bottom",fill="x")

# GUI buttons, labels and entries--------------------------------------------------------------------------------------
loadButton = tk.Button(root, text="wczytaj zbiór z pliku seeds_dataset.csv", command = load)
buildModelButton = tk.Button(root, text="zbuduj model na aktualnych danych", command = buildModel)
addRecordButton = tk.Button(root, text="dodaj rekord do aktualnych danych", command = addRecord)
predictButton = tk.Button(root, text="testuj rekord", command=predict)
testButton = tk.Button(root, text="ocena aktualnego modelu", command=test)
#refreshTableButton = tk.Button(root, text="odświerzanie tabelki", command=refreshTable)
saveButton = tk.Button(root,text="zapisz aktualny model do pliku model.sav", command=save)
clearButton = tk.Button(root, text="wyczyść dane", command=clear)
loadModelButton = tk.Button(root, text="wczytaj model z pliku model.sav", command=load_model)
saveToDataBaseButton = tk.Button(root,text="zapisz dane do bazy danych", command=saveToDataBase)
loadFromDataBaseButton = tk.Button(root,text="wczytaj dane z bazy danych", command=loadFromDataBase)


loadButton.grid(column=0,row=1, columnspan=2)
label.grid(column=0,row=2)
entryN.grid(column=1,row=2)
buildModelButton.grid(column=0,row=3,columnspan=2)
arg1lab.grid(column=0,row=4)
arg2lab.grid(column=0,row=5)
arg3lab.grid(column=0,row=6)
arg4lab.grid(column=0,row=7)
arg5lab.grid(column=0,row=8)
arg6lab.grid(column=0,row=9)
arg7lab.grid(column=0,row=10)
classlab.grid(column=0,row=11)

entry1.grid(column=1,row=4)
entry2.grid(column=1,row=5)
entry3.grid(column=1,row=6)
entry4.grid(column=1,row=7)
entry5.grid(column=1,row=8)
entry6.grid(column=1,row=9)
entry7.grid(column=1,row=10)

entryclass.grid(column=1,row=11)
addRecordButton.grid(column=0, row=12, columnspan=2)
predictButton.grid(column=0, row=13, columnspan=2)
testButton.grid(column=0,row=14,columnspan=2)
#refreshTableButton.grid(column=0, row=20, columnspan=2)
clearButton.grid(column=0, row=15, columnspan=2)
saveButton.grid(column=0, row=16, columnspan=2)
loadModelButton.grid(column=0, row=17, columnspan=2)
saveToDataBaseButton.grid(column=0, row=18, columnspan=2)
loadFromDataBaseButton.grid(column=0, row=19, columnspan=2)

# start -----
root.mainloop()
