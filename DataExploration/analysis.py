import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix


def scatterPlot(df):
    scatter = scatter_matrix(df[['HTGD', 'ATGD', 'HTP', 'ATP', 'DiffFormPts', 'DiffPts']], figsize=(10, 10))
    plt.savefig("ScatterMatrix.png")


def getStats(df):
    n = df.shape[0]
    f = df.shape[1] - 1
    hw = len(df[df.FTR == 'H'])
    aw = len(df[df.FTR == 'A'])
    dr = len(df[df.FTR == 'D'])
    hw_rate = float(hw / n) * 100
    aw_rate = float(aw / n) * 100
    dr_rate = float(dr / n) * 100
    return hw, aw, dr, hw_rate, aw_rate, dr_rate


def data_explore(seasons):
    hw = []
    aw = []
    draw = []
    hw_rate = []
    aw_rate = []
    dr_rate = []
    for df in seasons:
        h, a, d, hr, ar, dr = getStats(df)
        hw.append(h)
        aw.append(a)
        draw.append(d)
        hw_rate.append(hw_rate)
        aw_rate.append(aw_rate)
        dr_rate.append(dr_rate)

    # labels = "Home Win", "Away Win", "Draw"
    # sizes = np.array([hw_rate[0], aw_rate[0], dr_rate[0]])
    # sizes = [np.average(hw_rate), np.average(aw_rate), np.average(dr_rate)]
    # explode = (0.1, 0, 0)
    # fig2 = plt.plot()
    # lt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    # shadow=True, startangle=90)
    # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.savefig('PieChart.png', dpi=fig2.dpi)

    fig1 = plt.figure()
    xpos = np.arange(len(seasons))
    plt.xticks(xpos, ('11/12', '12/13', '13/14', '14/15', '16/17', '17/18', '18/19', '19/20'))
    plt.xlabel("Seasons")
    plt.ylabel("Total")
    plt.bar(xpos + 0.1, hw, label="Home Wins", width=0.3)
    plt.bar(xpos + 0.2, aw, label="Away Wins", width=0.3)
    plt.bar(xpos + 0.3, draw, label="Draws", width=0.3)
    plt.legend(loc='best')
    fig1.savefig('TotalStats.png', dpi=fig1.dpi)
