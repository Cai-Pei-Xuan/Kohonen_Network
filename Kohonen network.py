# -*- coding: utf-8 -*-
# Unsupervised clustering for the UCI-WINE dataset using Kohonen network
# 參考網站 : https://visualstudiomagazine.com/articles/2019/01/01/self-organizing-maps-python.aspx
import numpy as np
import matplotlib.pyplot as plt

# 載入資料
def loadData():
    data_file = "wine.data"
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(1,14),
        dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", usecols=[0],
        dtype=np.int)

    return data_x, data_y

# construct the SOM
def constructSOM(Rows, Cols, Dim, StepsMax, RangeMax, LearnMax, data_x):
    print("Constructing a 30x30 SOM from the wine data")
    map = np.random.random_sample(size=(Rows,Cols,Dim))

    # The call to random_sample() generates a 30 x 30 matrix where each cell is a vector of size 4 with random values between 0.0 and 1.0.
    for s in range(StepsMax):
        if s % (StepsMax/10) == 0: print("step = ", str(s))
        pct_left = 1.0 - ((s * 1.0) / StepsMax)
        curr_range = (int)(pct_left * RangeMax)
        curr_rate = pct_left * LearnMax

        # A random data item is selected and the best matching unit map node/cell is determined
        t = np.random.randint(len(data_x))
        (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)

        # The update moves the current node vector closer to the current data item using the curr_rate value which slowly decreases over time.
        for i in range(Rows):
            for j in range(Cols):
                if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])
    
    return map

def manhattan_dist(r1, c1, r2, c2):
    return np.abs(r1-r2) + np.abs(c1-c2)
    
# Using a SOM for Dimensionality Reduction Visualization(DRV)
def DRV(Rows, Cols, data_x, data_y, map):
    print("Associating each data label to one map node ")
    mapping = np.empty(shape=(Rows, Cols), dtype=object)
    for i in range(Rows):
        for j in range(Cols):
            mapping[i][j] = [] # empty list

    for t in range(len(data_x)):
        (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
        mapping[m_row][m_col].append(data_y[t])

    label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
    for i in range(Rows):
        for j in range(Cols):
            label_map[i][j] = most_common(mapping[i][j], 4)
    
    return label_map

def closest_node(data, t, map, m_rows, m_cols):
    # (row,col) of map node closest to data[t]
    result = (0,0)
    small_dist = 1.0e20
    for i in range(m_rows):
        for j in range(m_cols):
            ed = euc_dist(map[i][j], data[t])
            if ed < small_dist:
                small_dist = ed
                result = (i, j)
    return result

def euc_dist(v1, v2):
    return np.linalg.norm(v1 - v2) 

def most_common(lst, n):
    # lst is a list of values 0 . . n
    if len(lst) == 0: return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[lst[i]] += 1
    return np.argmax(counts)

# 畫圖
def Draw(label_map):
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 13))
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    # 自訂參數
    np.random.seed(1)
    Rows = 30; Cols = 30
    Dim = 13
    StepsMax = 5000
    RangeMax = Rows + Cols
    LearnMax = 0.5
    
    # 載入資料
    data_x, data_y = loadData()
    # construct the SOM
    map = constructSOM(Rows, Cols, Dim, StepsMax, RangeMax, LearnMax, data_x)
    # Using a SOM for Dimensionality Reduction Visualization(DRV)
    label_map = DRV(Rows, Cols, data_x, data_y, map)
    # 畫圖
    Draw(label_map)