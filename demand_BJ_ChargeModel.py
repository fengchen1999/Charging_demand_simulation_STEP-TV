import pickle
import cloudpickle
import dill

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import my_networkx as my_nx
import random
import pandas as pd

from itertools import islice

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from fitter import *
from scipy.stats import beta
from scipy.stats import cauchy
from scipy.optimize import curve_fit
import time
# import my_curve_fitting

from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy.optimize import broyden1
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import root

# 改默认字体
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']



# ----------------- trajectory class ------------------- #
class traj:
    '''
    This class define a trajetory
    '''
    def __init__(self, G, source, dest, init_time=7, init_SOC=100, temperature=25):
        self.source = source
        self.dest = dest
        self.init_time = init_time
        self.init_SOC = init_SOC
        self.path = []
        self.time_seq = []
        self.SOC_seq = []
        self.SOC_prob_seq = []

        self.path_cut = []
        self.time_seq_cut = []
        self.SOC_seq_cut = []
        self.SOC_prob_seq_cut = []

        # vehicle parameters
        self.battery_cap = 40 # kWh
        self.energy_effi = 6.075 # km per kWh

        # environment parameters
        self.temperature = temperature

        # whether chagging
        self.if_charging = 0

    def energy_effi_adjust(self, speed, temperature):
        effi_matrix = np.array([[2.03, 1.93, 1.86, 1.78, 1.67, 1.54, 1.41, 1.32, 1.29, 1.36, 1.53, 1.80, 2.13],
                                [1.77, 1.68, 1.62, 1.55, 1.46, 1.53, 1.23, 1.15, 1.13, 1.19, 1.34, 1.57, 1.86],
                                [1.63, 1.55, 1.49, 1.43, 1.34, 1.24, 1.14, 1.06, 1.04, 1.09, 1.23, 1.45, 1.71],
                                [1.46, 1.39, 1.34, 1.28, 1.21, 1.11, 1.02, 0.95, 0.93, 0.98, 1.10, 1.30, 1.53],
                                [1.35, 1.28, 1.23, 1.18, 1.11, 1.02, 0.94, 0.88, 0.86, 0.90, 1.02, 1.20, 1.41],
                                [1.32, 1.25, 1.20, 1.15, 1.08, 1.00, 0.92, 0.85, 0.84, 0.88, 0.99, 1.17, 1.38],
                                [1.35, 1.28, 1.23, 1.18, 1.11, 1.02, 0.94, 0.88, 0.86, 0.90, 1.02, 1.20, 1.41],
                                [1.41, 1.34, 1.29, 1.24, 1.16, 1.07, 0.98, 0.92, 0.90, 0.94, 1.06, 1.25, 1.47],
                                [1.47, 1.39, 1.34, 1.29, 1.21, 1.11, 1.02, 0.95, 0.93, 0.98, 1.11, 1.30, 1.54],
                                [1.55, 1.47, 1.42, 1.36, 1.28, 1.18, 1.08, 1.01, 0.99, 1.04, 1.17, 1.38, 1.63],
                                [1.64, 1.56, 1.50, 1.44, 1.35, 1.25, 1.14, 1.07, 1.04, 1.10, 1.24, 1.46, 1.72],
                                [1.68, 1.59, 1.54, 1.47, 1.39, 1.28, 1.17, 1.09, 1.07, 1.13, 1.27, 1.49, 1.76],
                                [1.82, 1.72, 1.66, 1.59, 1.50, 1.38, 1.26, 1.18, 1.15, 1.21, 1.37, 1.61, 1.90]])
        temperature_seq = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]) # °C
        speed_seq = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5]) * 1.609344 # mph -> km per hour
        temperature_step = 5
        speed_step = 5 * 1.609344 # mile -> km

        effi_matrix_expand = np.vstack((effi_matrix[0, :], effi_matrix, effi_matrix[-1, :]))
        effi_matrix_expand = np.hstack((effi_matrix_expand[:, 0].reshape(-1, 1), effi_matrix_expand, effi_matrix_expand[:, -1].reshape(-1, 1))) # expand efficient matrix

        speed_index = int((speed - speed_seq[0]) // speed_step) # origin of linear interpolation
        speed_lambda = (speed - speed_seq[0]) % speed_step / speed_step # coefficient of linear interpolation

        temperature_index = int((temperature - temperature_seq[0])) // temperature_step  # origin of linear interpolation
        temperature_lambda = (temperature - temperature_seq[0]) % temperature_step / temperature_step # coefficient of linear interpolation

        if speed_index < 0:
            speed_index = 0
        elif speed_index > 12:
            speed_index = 13
        else:
            speed_index += 1

        if temperature_index < 0:
            temperature_index = 0
        elif temperature_index > 12:
            temperature_index = 13
        else:
            temperature_index += 1

        speed_1 = (1 - speed_lambda) * effi_matrix_expand[speed_index, temperature_index] \
        + speed_lambda * effi_matrix_expand[speed_index+1, temperature_index]
        speed_2 = (1 - speed_lambda) * effi_matrix_expand[speed_index, temperature_index+1] \
        + speed_lambda * effi_matrix_expand[speed_index+1, temperature_index+1]
        effi = (1 - temperature_lambda) * speed_1 + temperature_lambda * speed_2

        return effi

    def path_gen(self, candidate_paths):
        tmp = random.randint(1, len(candidate_paths))
        self.path = candidate_paths[tmp-1]

    def time_seq_gen(self, G):
        self.time_seq.append(self.init_time)
        tmp = self.init_time
        num = len(self.path)
        if num > 1:
            for i in range(num - 1):
                tmp += G.edges[self.path[i], self.path[i+1]]['weight']/60 # min -> hour
                if tmp <= 24:
                    self.time_seq.append(tmp)
                else:
                    self.time_seq.append(tmp - 24) # second day

    def SOC_seq_gen(self, G):
        self.SOC_seq.append(self.init_SOC)
        tmp = self.init_SOC
        num = len(self.path)
        if num > 1:
            for i in range(num - 1):
                speed = G.edges[self.path[i], self.path[i+1]]['speed']
                length = G.edges[self.path[i], self.path[i+1]]['length']
                time_ = G.edges[self.path[i], self.path[i+1]]['weight'] # min
                EEA = self.energy_effi_adjust(speed, self.temperature) # energy_effic_adjust
                # tmp -= length / self.energy_effi / self.battery_cap * EEA * 100 # note: 1 -> 100
                tmp -= round(EEA * 6.41 * time_ /60 / self.battery_cap * 100, 3)
                #note: 6.41kW, 1 -> 100%, 60:min -> hour
                self.SOC_seq.append(tmp)

        # ------------- cut off trajectory where SOC below zero ------------------- #
        cut = 0
        for i in range(len(self.SOC_seq)):
            if self.SOC_seq[i] < 0:
                self.path_cut = self.path[:i]
                self.time_seq_cut = self.time_seq[:i]
                self.SOC_seq_cut = self.SOC_seq[:i]
                cut = 1
                break

        if cut == 0:
            self.path_cut = self.path
            self.time_seq_cut = self.time_seq
            self.SOC_seq_cut = self.SOC_seq
            #print('111')





def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

def read_UE_results():
    inFile = open("UE_results.dat")
    tmpIn = inFile.readline().strip().split("\t")
    print(tmpIn)
    edges = []
    for x in inFile:
        tmpIn = x.strip().split("\t")
        edges.append([float(tmp) for index, tmp in enumerate(tmpIn)])
        edges[-1][0] = int(edges[-1][0])
        edges[-1][1] = int(edges[-1][1])
    return edges


def read_XY():
    file_path=r'.\Sioux Falls network\SiouxFalls_node.txt'
    with open(file_path, 'r') as f:
        temp = f.readlines()
        data = []
        for i, rows in enumerate(temp):
            if i in range(1,len(temp)):
                temp1 = rows.strip().split("\t")
                temp1 = temp1[:-1]
                temp1 = [int(x) for x in temp1]
                temp1[2] = temp1[2] #增加y轴距离
                print(temp1)
                data.append(temp1)
        print('length', len(data))
        f.close()
    print(data)
    return data

def read_OD():
    inputLocation = "Sioux Falls network/"
    inFile = open(inputLocation + "demand.dat")
    tmpIn = inFile.readline().strip().split("\t")
    OD_flow = {}
    for x in inFile:
        tmpIn = x.strip().split("\t")
        OD_flow[(int(tmpIn[0]), int(tmpIn[1]))] = float(tmpIn[2]) * 10 # vehicles per day
    inFile.close()
    OD_flow_values = list(OD_flow.values())
    total_flow = sum(OD_flow.values())

    OD_flow_accu = []
    tmp = 0
    for i in range(len(OD_flow)):
        tmp += OD_flow_values[i]
        OD_flow_accu.append(tmp)

    return OD_flow, OD_flow_accu

def read_travel_start_time():
    path = "./materials/travel_start_time_NHTS2017.csv"
    pd_reader = pd.read_csv(path, header=None)
    data = np.array([pd_reader.iloc[:, 0].tolist(), pd_reader.iloc[:, 1].tolist()]).T
    # fix travel time
    data[:, 0] = np.rint(data[:, 0]) + 0.5 # 四舍五入到整数
    total = sum(data[:, 1])
    data[:, 1] = data[:, 1] / total # nomalization

    fig, ax = plt.subplots()
    plt.plot(data[:, 0], data[:, 1])
    plt.show()

    fig.savefig('./results_ChargeModel/travel_start_time_distribution_NHTS2017.pdf', bbox_inches='tight', pad_inches=0, dpi=600)

    return data

def read_travel_start_SOC(step = 5):
    '''
    step: the width of bins (unit: percentage)
    '''
    # 概率密度函数 PDF
    path2 = "./materials/travel_start_SOC_BJ_2023.csv"
    pd_reader2 = pd.read_csv(path2, header=None)
    data2 = np.array([pd_reader2.iloc[:, 0].tolist(), pd_reader2.iloc[:, 1].tolist()]).T

    # fix data2
    data2[:, 1] = data2[:, 1] - min(data2[:, 1])
    accu = [0] * int(100/step)

    for i in range(1, len(data2[:, 0])):
        accu[int(data2[i, 0] // step)] += (data2[i, 1] + data2[i-1, 1]) / 2 * (data2[i, 0] - data2[i-1, 0])
    total = sum(accu)
    data2_hist = [x/total for x in accu]

    # fix data2 without SOC=100%
    data2_without_100 = data2[:-1, :]
    accu_without_100 = [0] * int(100/step)

    for i in range(1, len(data2_without_100[:, 0])):
        accu_without_100[int(data2_without_100[i, 0] // step)] += (data2_without_100[i, 1] + data2_without_100[i-1, 1]) / 2\
                                                      * (data2_without_100[i, 0] - data2_without_100[i-1, 0])
    total_without_100 = sum(accu_without_100)
    data2_hist_without_100 = [x/total_without_100 for x in accu_without_100]

    fig, ax = plt.subplots()

    if step == 10:
        bar_x = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', \
            '[50, 60)', '[60, 70)', '[70, 80)', '[80, 90)', '[90, 100]']
    elif step == 5:
        bar_x = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', \
            '[50, 60)', '[60, 70)', '[70, 80)', '[80, 90)', '[90, 100]'] * 2

    index = np.arange(len(bar_x))
    bar_width = 0.3
    # bar1 = plt.bar(index-bar_width/2, data1_hist, width=bar_width, label='Ireland 2016')
    # bar2 = plt.bar(index+bar_width/2, data2_hist, width=bar_width, label='Beijing 2023')

    # bar1 = plt.bar(index-bar_width/2*3, data1_hist, width=bar_width, label='Ireland 2016')
    bar2 = plt.bar(index+bar_width/2*0, data2_hist, width=bar_width, label='Beijing 2023')
    # bar_avg = plt.bar(index+bar_width/2*3, data_avg_hist, width=bar_width, label='average')

    # # 三图版本
    # bar1 = plt.bar(index - bar_width / 2 * 3, data1_hist, width=bar_width, label='Ireland 2016')
    # bar1 = plt.bar(index - bar_width / 2 * 0, data1_hist, width=bar_width, label='Ireland 2016')
    # bar1 = plt.bar(index - bar_width / 2 * 3, data1_hist, width=bar_width, label='Ireland 2016')

    plt.xlabel('Initial SOC (%)')
    plt.ylabel('Probability')
    plt.xticks(index, bar_x)
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()
    fig.savefig("./results_ChargeModel/init_SOC.pdf", bbox_inches='tight', pad_inches=0, dpi=600)

    # ---------- plot data without SOC=100%
    fig, ax = plt.subplots()

    if step == 10:
        bar_x = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', \
            '[50, 60)', '[60, 70)', '[70, 80)', '[80, 90)', '[90, 99]']
    elif step == 5:
        bar_x = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', \
            '[50, 60)', '[60, 70)', '[70, 80)', '[80, 90)', '[90, 99]'] * 2

    index = np.arange(len(bar_x))
    bar_width = 0.3
    # bar1 = plt.bar(index-bar_width/2, data1_hist, width=bar_width, label='Ireland 2016')
    # bar2 = plt.bar(index+bar_width/2, data2_hist, width=bar_width, label='Beijing 2023')

    # bar1 = plt.bar(index-bar_width/2*3, data1_hist_without_100, width=bar_width, label='Ireland 2016')
    bar2 = plt.bar(index+bar_width/2*0, data2_hist_without_100, width=bar_width, label='Beijing 2023')
    # bar_avg = plt.bar(index+bar_width/2*3, data_avg_hist_without_100, width=bar_width, label='average')

    # # 三图版本
    # bar1 = plt.bar(index - bar_width / 2 * 3, data1_hist, width=bar_width, label='Ireland 2016')
    # bar1 = plt.bar(index - bar_width / 2 * 0, data1_hist, width=bar_width, label='Ireland 2016')
    # bar1 = plt.bar(index - bar_width / 2 * 3, data1_hist, width=bar_width, label='Ireland 2016')

    plt.xlabel('Initial SOC (%)')
    plt.ylabel('Probability')
    plt.xticks(index, bar_x)
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()
    fig.savefig("./results_ChargeModel/init_SOC_without_100.pdf", bbox_inches='tight', pad_inches=0, dpi=600)

    return data2_hist_without_100

def read_arrive_SOC():
    arrive_SOC_matrix = np.array([[2, 4.4, 6.3, 8.5, 11.2, 12.4, 14.4, 14.7, 14.5, 11.6],
                                  [5, 8.4, 9.9, 11.2, 11.9, 11.5, 10.6, 10.5, 10.4, 10.6],
                                  [4.5, 8.0, 9.0, 9.4, 10.0, 10.8, 10.4, 11.1, 12.5, 14.4],
                                  [3.7, 9.9, 12.8, 12.9, 13.9, 13.3, 11, 9.2, 7.8, 5.5],
                                  [3.4, 9.6, 13.7, 14.7, 15.3, 14.2, 11.8, 9.6, 5.8, 1.9],
                                  [4.0, 10.2, 13.4, 14.4, 14.7, 13.6, 11.3, 9.2, 6.3, 2.8],
                                  [3.8, 10.4, 13.2, 14, 14.6, 13.7, 11.3, 9.3, 6.3, 3.4],
                                  [4.1, 10.5, 13.6, 14.7, 14.7, 13.0, 10.5, 9.1, 6.4, 3.5],
                                  [4.4, 10.8, 14.1, 15.4, 14.9, 13.3, 10.8, 8.2, 5.4, 2.9],
                                  [3.8, 11.6, 15.3, 16.0, 15.5, 13.5, 10.4, 7.8, 4.4, 1.7],
                                  [2.0, 4.4, 6.3, 8.5, 11.2, 12.4, 14.4, 14.7, 14.5, 11.6]]) / 100
    insert = (arrive_SOC_matrix[3, :] + arrive_SOC_matrix[4, :]) / 2
    arrive_SOC_matrix = np.insert(arrive_SOC_matrix, 3, values=insert, axis=0)
    print(arrive_SOC_matrix.sum(axis=1))
    fig, ax = plt.subplots()
    x = np.arange(0, 100, 10) + 0.5
    for i in range(arrive_SOC_matrix.shape[0]):
        ax.plot(x, arrive_SOC_matrix[i, :], label='[' + str(i*2) + ' ,' + str(i*2+2) + ']')
    plt.xlabel('Arrive SOC (%)')
    plt.ylabel('Probability')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.15), ncol=4)  # (0.5, 1.15) --> (width, height)
    plt.show()
    fig.savefig("./results_ChargeModel/arrive_SOC.pdf", bbox_inches='tight', pad_inches=0, dpi=600)

    # joint probability version
    gmm_means = [8.08, 11.23, 20.00]
    gmm_vars = [0.735, 2.98, 2.96]
    gmm_props = [0.079, 0.282, 0.599]
    time_props = []
    for i in range(arrive_SOC_matrix.shape[0]):
        tmp = 0
        if i < 2:
            for j in range(len(gmm_props)):
                tmp += gmm_props[j] * (norm.cdf(2 * (i + 1) + 24, gmm_means[j], gmm_vars[j])
                                       - norm.cdf(2 * i + 24, gmm_means[j], gmm_vars[j]))
        else:
            for j in range(len(gmm_props)):
                tmp += gmm_props[j] * (norm.cdf(2*(i+1), gmm_means[j], gmm_vars[j])
                                       - norm.cdf(2*i, gmm_means[j], gmm_vars[j]))
        time_props.append(tmp)

    tmp = arrive_SOC_matrix * \
                        np.tile(np.array(time_props).reshape(-1,1), (1, arrive_SOC_matrix.shape[1]))
    total = tmp.sum()
    arrive_SOC_matrix = tmp / total

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _x = np.array([1,3,5,7,9,11,13,15,17,19,21,23])
    _y = np.array([5,15,25,35,45,55,65,75,85,95])
    _xx, _yy = np.meshgrid(_x, _y)
    _t, _SOC = _xx.ravel(), _yy.ravel()

    ax.bar3d(_t, _SOC, 0, 0.7, 0.7, arrive_SOC_matrix.T.ravel(), shade=True) # x, y, bottom, width, deep, top
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('SOC (%)')
    ax.set_zlabel('Joint probability')
    plt.show()


    return arrive_SOC_matrix

def fit_travel_start_SOC(initial_SOC_hist, step=5):
    # interpolation，soomth histogram
    multiply = 10
    SOCs = np.arange(0, 100, step) + step/2
    SOCs[-1] = np.arange(0, 100, step)[-1] + (step-1)/2 # 修改点：最后只有96，97，98，99四个数字
    SOCs_expand = np.concatenate((np.array([0]), SOCs, np.array([99])))

    y = initial_SOC_hist
    y_start = np.array([y[0] - (y[1] - y[0]) / (step + step - 1) * (step - 1)]) # 修改点，并不是直接/2
    y_end = np.array([y[-1] + (y[-1] - y[-2]) / (step + step - 1) * (step - 1)])
    initial_SOC_hist_expand = np.concatenate((y_start, y, y_end))

    SOCs_interp = np.arange(0, 99, step/multiply) # 修改点：没到100
    initial_SOC_hist_interp = np.interp(SOCs_interp, SOCs_expand, initial_SOC_hist_expand)

    # sampling
    sample_num = 600000
    tmp = random.choices(SOCs_interp, weights=initial_SOC_hist_interp,
                         k=sample_num)
    samples = np.zeros(sample_num)
    for i in range(sample_num):
        samples[i] = random.uniform(tmp[i], tmp[i] + step/multiply)

    # ---------- fitting by fitter module
    # f_init_SOC = Fitter(np.array(samples), timeout=100)  # 创建Fitter类
    f_init_SOC = Fitter(np.array(samples), distributions=get_common_distributions()+['beta', 'argus'], timeout=100)  # 创建Fitter类
    # f = Fitter(df.kWhDelivered.to_list(), distributions=['gamma', 'erlang', 'norm', 'uniform'], timeout=3)  # 创建Fitter类
    f_init_SOC.fit()  # 调用fit函数拟合分布
    f_init_SOC.summary(Nbest=5) # save in summary()
    plt.show()
    # plt.savefig("./results_ChargeModel/init_SOC_fitting.pdf", bbox_inches='tight', pad_inches=0, dpi=600)

    best_distributions = f_init_SOC.df_errors.sort_values(by='sumsquare_error')
    print(best_distributions)

    # # ---------- fitting by gmm
    # gmm = GaussianMixture(n_components=10).fit(samples.reshape(-1, 1))
    #
    # fig, ax = plt.subplots()
    # f_axis = samples.copy().ravel()
    # f_axis.sort()
    # a = []
    # for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
    #     a.append(weight * norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
    #     plt.plot(f_axis, a[-1])  # gauss distribution components
    #
    # multiply = 10  # 线性分段采样，读取数据的函数里面也要改
    # plt.hist(samples, bins=99, alpha=0.5, density=True)
    # plt.plot(f_axis, np.array(a).sum(axis=0), 'k-')  # gauss mixture distribution
    # initial_SOC_hist_1per = []
    # for i in range(len(initial_SOC_hist)):
    #     if i < len(initial_SOC_hist) - 1:
    #         initial_SOC_hist_1per.append(initial_SOC_hist[i] / step)
    #     else:
    #         initial_SOC_hist_1per.append(initial_SOC_hist[i] / ((step + step-1) / 2))
    # plt.plot(SOCs, initial_SOC_hist_1per)  # NHTS2017
    # plt.xlabel('Time (hour)')
    # plt.ylabel('Probability')
    # plt.tight_layout()
    # # plt.legend(['1','2','3'])
    # plt.grid(axis='y')
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # plt.legend(['component 1',
    #             'GMM fitting results', 'histogram of resample'],
    #            loc='center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # (0.5, 1.15) --> (width, height)
    #
    # plt.show()
    return f_init_SOC.fitted_param['beta']



def OD_select(OD_flow, OD_flow_accu):
    x = random.random() * OD_flow_accu[-1] # generate a random value
    # ---------- bisection method ---------- #
    low = 0
    high = len(OD_flow_accu) - 1
    while low < high:
        mid = (low + high) // 2
        dist = OD_flow_accu[mid + 1]
        if dist < x:
            low = mid + 1
        elif dist > x:
            high = mid
        else:
            low = mid
            break
    return list(OD_flow.keys())[low + 1]

def fit_travel_start_time(travel_start_time_distribution):
    # 线性插值
    multiply = 10
    data = travel_start_time_distribution
    data_start = np.array([0, data[0, 1] - (data[1, 1] - data[0, 1]) / 2])
    data_end = np.array([24, data[-1, 1] + (data[-1, 1] - data[-2, 1]) / 2])
    data_expand = np.vstack((data_start, data, data_end)) # 补充头尾值
    data_interp = np.zeros((len(data[:, 1]) * multiply, 2))
    data_interp[:, 0] = np.arange(0, 24, 1/multiply)
    data_interp[:, 1] = np.interp(data_interp[:, 0], data_expand[:, 0], data_expand[:, 1])
    travel_start_time_distribution_interp = data_interp

    # sampling
    sample_num = 600000
    tmp = random.choices(travel_start_time_distribution_interp[:, 0], weights=travel_start_time_distribution_interp[:, 1],
                         k=sample_num)
    samples = np.zeros(sample_num)
    for i in range(sample_num):
        samples[i] = random.uniform(tmp[i], tmp[i] + 1/multiply) # the number of bins is 10*5, so step = 0.2

    # gmm fitting
    gmm = GaussianMixture(n_components=4).fit(samples.reshape(-1, 1))

    # # calculate SSE
    # y, x = np.histogram(samples.reshape(-1, 1), bins=100, density=True)
    # x_avg = (x[1:] + x[:-1]) / 2
    # y_fit = np.zeros_like(y)
    # for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
    #     y_fit += weight * norm.pdf(x_avg, mean, np.sqrt(covar)).ravel()
    # sq_error = np.power((y_fit-y), 2).sum()
    # print('Sum of squares error: ', sq_error)



    fig, ax = plt.subplots()
    f_axis = samples.copy().ravel()
    f_axis.sort()
    a = []
    for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        a.append(weight * norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
        plt.plot(f_axis, a[-1])  # gauss distribution components

    multiply = 10  # 线性分段采样，读取数据的函数里面也要改
    plt.hist(samples, bins=24 * multiply, alpha=0.5, density=True)
    plt.plot(f_axis, np.array(a).sum(axis=0), 'k-')  # gauss mixture distribution
    plt.plot(travel_start_time_distribution[:, 0], travel_start_time_distribution[:, 1])  # NHTS2017
    plt.xlabel('Time (hour)', size=12)
    plt.ylabel('Probability', size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    # plt.legend(['1','2','3'])
    plt.grid(axis='y')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.75])
    plt.legend(['component 1', 'component 2', 'component 3', 'component 4',
                'GMM fitting results', 'data from NHTS 2017', 'histogram of resample'],
               loc='center', bbox_to_anchor=(0.5, 1.19), ncol=2, fontsize=12)  # (0.5, 1.15) --> (width, height)

    plt.show()
    fig.savefig('./results_ChargeModel/travel_start_time_distribution_NHTS2017_fitting.pdf', bbox_inches='tight', pad_inches=0,
                dpi=600)

    return gmm

def gen_network():
    # -------------------- draw network -------------------- #
    edges = read_UE_results()
    data_XY = read_XY()
    # build road network using networkx
    G = nx.DiGraph()
    for i in range(1,24+1):
        G.add_node(i, name=str(i))  # 结点名称不能为str,desc为标签即结点名称
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1], weight=round(edges[i][5], 3)
                   , length=edges[i][3], speed=round(edges[i][3]/edges[i][5]*60, 3))
        # time:min, length:km, speed: km/h
    pos = {}
    for i in range(len(data_XY)):
        pos[i+1] = tuple(data_XY[i][1:3])

    # draw network
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.15', arrowsize=5)
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])


    my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, rotate=True, rad=0.13, font_size=1)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_edges(G, pos, width=1, alpha=0.7, arrows=False)
    # # 画出边权值
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, font_size=8, edge_labels=edge_labels)
    ax.set_aspect(1) # the proportion of high and width
    plt.show()
    fig.savefig("./results_ChargeModel/Sioux_falls_road_network.pdf", bbox_inches='tight', pad_inches=0, dpi=600)
    return G


def gen_candidate_paths(G):
    # ----------------- deviation limited k shortest paths generation ------------------- #
    K = 3 # k shortest
    p = 0.1 # deviation coefficient
    OD_K_shortest_paths = {}
    OD_candidate_shortest_paths = {} # shortest + deviation limited
    OD_candidate_shortest_paths_num = {} # <= k
    OD_num = 0
    for i in range(1, G.number_of_nodes()+1):
        for j in range(1, G.number_of_nodes()+1):
            if i != j:
                OD_K_shortest_paths[(i, j)] = k_shortest_paths(G, i, j, K, weight='weight')
                dist = []
                #print('(', i, ',', j, ')', '=' * 20)
                for path in OD_K_shortest_paths[(i, j)]:
                    dist.append(nx.path_weight(G, path, weight='weight'))
                    #print(path)
                    #print(dist[-1])

                candidate_paths = [OD_K_shortest_paths[(i, j)][0]]
                for ii in range(1, len(dist)):
                    if dist[ii] - dist[0] < p * dist[0]:
                        candidate_paths.append(OD_K_shortest_paths[(i, j)][ii])
                    else:
                        break
                OD_candidate_shortest_paths[(i, j)] = candidate_paths
                OD_candidate_shortest_paths_num[(i, j)]= len(candidate_paths)

    for k, v in OD_candidate_shortest_paths.items():
        print(k, '='*20)
        for path in v:
            print(path)

    plt.hist(OD_candidate_shortest_paths_num.values())
    plt.show()

    # traj_1 = traj(G, 9, 15, 7, 0.8, temperature=32.5)
    # traj_1.path_gen(OD_candidate_shortest_paths[(9, 15)])
    # traj_1.time_seq_gen(G)
    # traj_1.SOC_seq_gen(G)
    # print(traj_1)

    return OD_candidate_shortest_paths

def fit_charging_pro():
    global trajs_multi_days, G, arrive_SOC_matrix, total_EV_traj, total_charging_traj
    global v
    vs_names = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11']
    p_ch = total_charging_traj / total_EV_traj # 单位轨迹充电次数

    T_S_SOC_count = np.zeros((12, G.number_of_nodes(), 10))

    for i in range(len(trajs_multi_days)):
        for j in range(len(trajs_multi_days[i])):
            # for k in range(len(trajs_multi_days[i][j].path_cut)): # 注意，改成终点才需要统计，即k=-1
            k = -1 #=============================
            if trajs_multi_days[i][j].time_seq_cut[k] < 24:
                T_index = int(trajs_multi_days[i][j].time_seq_cut[k] // 2)
            else:
                T_index == 11
            S_index = trajs_multi_days[i][j].path_cut[k] - 1
            if trajs_multi_days[i][j].SOC_seq_cut[k] < 100:
                SOC_index = int(trajs_multi_days[i][j].SOC_seq_cut[k] // 10)
            else:
                SOC_index = 9
            T_S_SOC_count[T_index, S_index, SOC_index] += 1

    T_S_count = T_S_SOC_count.sum(axis=2)  # traffic flow, vehicles per hour
    # # calculate prior probability
    # T_SOC_count = T_S_SOC_count.sum(axis=1)
    # matrixB = np.tile(T_SOC_count.sum(axis=1), (10, 1)).T
    # T_SOC_prob_prior = np.divide(T_SOC_count, matrixB)
    # # calculate posterior probability with bayes
    # T_SOC_prob_post = np.divide(arrive_SOC_matrix, T_SOC_prob_prior) * p_ch
    # T_SOC_prob_post[np.where(T_SOC_prob_post>1)] = 1
    # # # normalization
    # # matrixB = np.tile(T_SOC_prob_post.sum(axis=1), (10, 1)).T
    # # T_SOC_prob_post = T_SOC_prob_post / matrixB

    # # charging probability fitting
    # x = np.arange(0, 100, 10) + 5
    # x_expend = np.concatenate((np.array([0]), x, np.array([100])))
    # x_expend = np.tile(x_expend, (12, 1)).ravel()
    # post_expend = np.concatenate((np.ones((12, 1)), T_SOC_prob_post, np.zeros((12, 1))), axis=1)
    # post_expend = post_expend.ravel()
    #
    # def func(x, a, b, c, d):
    #     return 1 / (1 + np.exp(a * (x - b))) + 1 / (1 + np.exp(c * (x - d)))
    #
    # # popt返回的是给定模型的最优参数。我们可以使用pcov的值检测拟合的质量，其对角线元素值代表着每个参数的方差。
    # popt, _ = curve_fit(func, x_expend, post_expend)
    # a = popt[0]
    # b = popt[1]
    # c = popt[2]
    # d = popt[3]
    #
    # f_y = func(x_expend, a, b, c, d)
    # x_plot = np.arange(0, 100, 0.1)
    # y_plot = func(x_plot, a, b, c, d) #拟合y值
    # # plt.figure()
    # # plot1 = plt.plot(x_expend, post_expend, 's',label='original values')
    # # plot2 = plt.plot(x_plot, y_plot, 'r', label='polyfit values')
    # # plt.show()
    #
    # fig, ax = plt.subplots()
    # for i in range(T_SOC_prob_prior.shape[0]):
    #     ax.scatter(x_expend[i*12 : (i+1)*12], post_expend[i*12 : (i+1)*12],
    #             label='[' + str(i * 2) + ' ,' + str(i * 2 + 2) + ']') # plot posterior prob
    # ax.plot(x_plot, y_plot, 'r', label='logit fitting') # plot fitting curve
    # plt.xlabel('SOC (%)')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.show()


    # JOINT version
    # calculate prior probability JOINT
    T_SOC_count = T_S_SOC_count.sum(axis=1)
    T_SOC_count[np.where(T_SOC_count == 0)] = \
        min(T_SOC_count[np.where(T_SOC_count > 0)]) # tricks， 去掉分母为0
    T_SOC_prob_prior = T_SOC_count / T_S_count.sum()

    # calculate posterior probability with bayes
    T_SOC_prob_post = np.divide(arrive_SOC_matrix, T_SOC_prob_prior) * p_ch
    T_SOC_prob_post[np.where(T_SOC_prob_post > 1)] = 1
    # normalization
    # T_SOC_prob_post = T_SOC_prob_post / T_SOC_prob_post.sum()

    # charging probability fitting
    x = np.arange(0, 100, 10) + 5
    # x_expend = np.concatenate((np.zeros(1), x, np.tile(x[-1], 3), np.ones(1) * 100))
    # post_expend = np.concatenate((np.ones((12, 1)), T_SOC_prob_post, np.tile(T_SOC_prob_post[:, -1], (3, 1)).T, np.zeros((12, 1))), axis=1)

    x_expend = np.concatenate((np.zeros(1), x, np.ones(1) * 100))
    post_expend = np.concatenate((np.ones((12, 1)), T_SOC_prob_post, np.zeros((12, 1))), axis=1)
    # x_expend = x
    # post_expend = T_SOC_prob_post

    # def func(x, a, b, c, d):
    #     return 1 / (1 + np.exp(a * (x - b))) + 1 / (1 + np.exp(c * (x - d)))
    def func(x, a, b, c, d, e):
        return 1 / (1 + np.exp(a * (x - b))) + 1 / (1 + np.exp(c * (x - d))) + e
        # return 2 / (1 + np.exp(a * (x - b)))

    def funcc(x, a, b, c, d, e):
        return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e
        # return - a * np.power((x - b), 2) + c
        # return e * np.power(x, 4) + a * np.power(x, 3) + b * np.power(x, 2) + c * x + d

    fig, ax = plt.subplots()
    vs = []
    MSEs = []
    RMSEs = []
    popts = []
    for i in range(12):
    # popt返回的是给定模型的最优参数。我们可以使用pcov的值检测拟合的质量，其对角线元素值代表着每个参数的方差。
        if i == 0:
            popt, _ = curve_fit(funcc, x_expend, post_expend[i, :])
            a = popt[0]
            b = popt[1]
            c = popt[2]
            d = popt[3]
            e = popt[4]
            popts.append(popt)

            f_y = funcc(x_expend, a, b, c, d, e)
            x_plot = np.arange(0, 100, 0.1)
            y_plot = funcc(x_plot, a, b, c, d, e) #拟合y值

            def v(x, a=a, b=b, c=c, d=d, e=e):
                return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e
            v.__name__ = vs_names[i]
            vs.append(v)
        else:
            popt, _ = curve_fit(func, x_expend, post_expend[i, :])
            a = popt[0]
            b = popt[1]
            c = popt[2]
            d = popt[3]
            e = popt[4]
            popts.append(popt)

            f_y = func(x_expend, a, b, c, d, e)
            x_plot = np.arange(0, 100, 0.1)
            y_plot = func(x_plot, a, b, c, d, e)  # 拟合y值
            def v(x, a=a, b=b, c=c, d=d, e=e):
                return 1 / (1 + np.exp(a * (x - b))) + 1 / (1 + np.exp(c * (x - d))) + e
            v.__name__ = vs_names[i]
            vs.append(v)
        # plt.figure()
        # plot1 = plt.plot(x_expend, post_expend, 's',label='original values')
        # plot2 = plt.plot(x_plot, y_plot, 'r', label='polyfit values')
        # plt.show()

        ax.scatter(x_expend, post_expend[i, :],
                label='[' + str(i * 2) + ' ,' + str(i * 2 + 2) + ']') # plot posterior prob
        ax.plot(x_plot, y_plot, label='logit fitting ' + str(i)) # plot fitting curve

        MSE = np.power((f_y - post_expend[i,:]), 2).sum() / (len(f_y)) # ------- 注意要加post_expend[i,:]
        RMSE = np.power(MSE, 1/2)
        # print("MSE of ", len(popt), "parameters: ", MSE)
        # print("RMSE of ", len(popt), "parameters: ", RMSE)
        MSEs.append(MSE)
        RMSEs.append(RMSE)

    plt.xlabel('SOC (%)', size=12)
    plt.ylabel('Probability', size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=12)

    plt.show()

    # fig.savefig('./results_ChargeModel/charging_pro_12curves_v2.pdf', bbox_inches='tight', pad_inches=0, dpi=600)

    return vs, MSEs, RMSEs, T_S_SOC_count, popts

def charging_demand(traj):
    '''
    if not charging to 100% SOC
    '''
    random_charging_demand = 0.35 * norm.rvs(loc=81.64, scale=60.55)\
                             + 0.67 * norm.rvs(loc=139.36, scale=94.11)



G = gen_network()
OD_candidate_shortest_paths = gen_candidate_paths(G)

# -------------------- generation of trajectories -------------------- #
# ---------- total number of charging trajectories per day
EV_ratio = 0.1
charging_freq = 1/2.074 # the number of charging per day
# source(2.074): ((1/3+0)*8.55 + (1/2+1/3)*3.59 + (1+1/2)*19 + (1+2)*23.7 + (2+3)*14.12 + (3+4)*9.23 + (4+4)*21.76) * 0.5 * 0.01
households = 77326
# households = 1000
# source: https://www.census.gov/quickfacts/fact/table/siouxfallscitysouthdakota/POP010220 accessed in 2023.01.11
vehilces_per_household = 1.88
home_charging_rate = 0.88
total_EV_traj = households * vehilces_per_household * EV_ratio * \
                (1*23.6 + 2*45.1 + 3*17.9 + 4*8.9 + 5*3 + 6*0.9 + 7*0.2) * 0.01
total_charging_traj = households * vehilces_per_household * EV_ratio * charging_freq * (1 - home_charging_rate)
total_charging_traj = int(total_charging_traj)
print('the number of total charging trajectories: ', total_charging_traj)
# ---------- OD flow
OD_flow, OD_flow_accu = read_OD() # vehicles per day, 计算累计分布，方便后续计算

# # ---------- read travel initial time
# travel_start_time_distribution = read_travel_start_time()
#
# # ---------- fit travel initial time
# init_time_gmm = fit_travel_start_time(travel_start_time_distribution)
#
# # ---------- read travel initial SOC
# initial_SOC_hist = read_travel_start_SOC(step=10)
#
# # ---------- fit travel initial SOC
# init_SOC_beta = fit_travel_start_SOC(initial_SOC_hist, step=10)
# #
# with open("./results_ChargeModel/init_fitting_results.pkl", "wb") as file:
#     pickle.dump([init_time_gmm, init_SOC_beta], file, True)

# ---------- load fitted distribution
with open("./results/init_fitting_results.pkl", "rb") as file:
    tmp = pickle.load(file)
    init_time_gmm = tmp[0]
    init_SOC_beta = tmp[1]
    a, b, loc1, scale1 = init_SOC_beta # (a, b), loc1, scale: shape, location, hight

# temperatures sampling: sioux falls 2022, unit: F
temper_high = np.array([32.3, 33.5, 45.1, 55.8, 71.5, 84.5, 87.7, 84.6, 80.1, 65.8, 45.4, 25.1])
temper_low = np.array([5.3, 6.9, 24.2, 31.6, 48.8, 60.5, 66.4, 62.5, 53.3, 37.9, 24.2, 9.5])
temper_avg = (temper_low + temper_high) / 2
samples_per_month = 2
temperatures = temper_avg.repeat(samples_per_month)
temperatures = (5 / 9 * (temperatures - 32)) # from °F to °C

# temperatures[-5:-1] = temperatures[-5:-1] - 10

temperatures = temperatures.round(1)

# temperatures = np.array([25, 25])

arrive_SOC_matrix = read_arrive_SOC()

# PDF_AT_100 = (15.903 + 51.627) / 2 # from Ireland 2016 and Beijing 2023
PDF_AT_100 = 15.903 # from Beijing 2023

# -------------------- fitting charging probability with Bayes -------------------- #

count = 0
trajs_multi_days = []
for k in temperatures:
    trajs_one_day = []
    while len(trajs_one_day) < total_EV_traj:
        count += 1
        if count % 100 == 0:
            print('=' * 20, 'traj ', count)
        # choose OD pair
        OD_tmp = OD_select(OD_flow, OD_flow_accu)
        # choose initial time
        init_time_tmp = init_time_gmm.sample(1)[0][0][0]
        # choose initial SOC
        if random.uniform(0, 100) < PDF_AT_100:
            init_SOC_tmp = 100
        else:
            init_SOC_tmp = beta.rvs(a, b, loc=loc1, scale=scale1, size=1)[0]
        # print(init_time_tmp)
        # print(init_SOC_tmp)

        traj_tmp = traj(G, OD_tmp[0], OD_tmp[1],
                        init_time_tmp, init_SOC_tmp, temperature=k)

        traj_tmp.path_gen(OD_candidate_shortest_paths[(OD_tmp[0], OD_tmp[1])])
        traj_tmp.time_seq_gen(G)
        traj_tmp.SOC_seq_gen(G)
        # print(traj_tmp.path_cut)
        # print(traj_tmp.time_seq_cut)
        # print(traj_tmp.SOC_seq_cut)
        # print(traj_tmp.SOC_prob_seq_cut)
        # print('---------')
        trajs_one_day.append(traj_tmp)
    trajs_multi_days.append(trajs_one_day)

# -------------------------- Bayes without trajectories ---------------------- #
charging_pro_logit, MSEs, RMSEs, T_S_SOC_count, popts = fit_charging_pro()
# with open("./results_ChargeModel/charging_pro_fit_results_v2.pkl", "wb") as file:
#     cloudpickle.dump([charging_pro_logit, MSEs, RMSEs, T_S_SOC_count, popts, trajs_multi_days], file, True)
#
# with open("./results/charging_pro_fit_results_v2.pkl", "rb") as file:
#     tmp = cloudpickle.load(file)
#     charging_pro_logit, MSEs, RMSEs, T_S_SOC_count, popts, trajs_multi_days_train = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]

timer0 = time.time()
# # -------------------- generate trajectories for test -------------------- #
# samples_per_month = 2
# temperatures = temper_avg.repeat(samples_per_month)
# temperatures = np.arange(-5, 115, 5) # ------------ 测温度打开 -------------- #
# temperatures = (5 / 9 * (temperatures - 32)) # from °F to °C
# temperatures = temperatures.round(1)
#
# # temperatures = np.array([25, 25])
#
# count = 0
# trajs_multi_days = []
# last_nums = []
# for k in temperatures:
#     trajs_one_day = []
#     count = 0
#     while len(trajs_one_day) < total_charging_traj:
#         count += 1
#         if count % 100 == 0:
#             print('=' * 20, ' traj ', count, ' Day ', k)
#
#         # choose OD pair
#         OD_tmp = OD_select(OD_flow, OD_flow_accu)
#         # choose initial time
#         init_time_tmp = init_time_gmm.sample(1)[0][0][0]
#         # choose initial SOC
#         if random.uniform(0, 100) < PDF_AT_100:
#             init_SOC_tmp = 100
#         else:
#             init_SOC_tmp = beta.rvs(a, b, loc=loc1, scale=scale1, size=1)[0]
#
#         traj_tmp = traj(G, OD_tmp[0], OD_tmp[1],
#                         init_time_tmp, init_SOC_tmp, temperature=k)
#
#         traj_tmp.path_gen(OD_candidate_shortest_paths[(OD_tmp[0], OD_tmp[1])])
#         traj_tmp.time_seq_gen(G)
#         traj_tmp.SOC_seq_gen(G)
#         if traj_tmp.time_seq_cut[-1] != 24:
#             tt = int(traj_tmp.time_seq_cut[-1] // 2)
#         else:
#             tt = 11
#
#         traj_tmp.SOC_prob_seq_cut.append(charging_pro_logit[tt](traj_tmp.SOC_seq_cut[-1]))
#         if random.random() < traj_tmp.SOC_prob_seq_cut[0] and traj_tmp.SOC_prob_seq_cut[0] > 0:
#             # traj_tmp.if_charging = 1
#             trajs_one_day.append(traj_tmp)
#         # print(traj_tmp.path_cut)
#         # print(traj_tmp.time_seq_cut)
#         # print(traj_tmp.SOC_seq_cut)
#         # print(traj_tmp.SOC_prob_seq_cut)
#         # print('---------')
#     trajs_multi_days.append(trajs_one_day)
#     last_nums.append(count)

# with open("./results_ChargeModel/data_trajs_for_test.pkl", "wb") as file:
#     cloudpickle.dump([trajs_multi_days], file, True)

with open("./results_ChargeModel/data_trajs_for_test.pkl", "rb") as file:
    tmp = cloudpickle.load(file)
    trajs_multi_days = tmp[0]
t1 = time.time() - timer0
timer0 = time.time()
# # -------------------- Charge Model -------------------- #
# from evse_class import EVSE_class
# from ElectricVehicles import ElectricVehicles
#
#
# ####################################################################
# ### Initialization
# ####################################################################
# t0 = 0
# tf = 24 * 60 * 60
# dt = 60
#
# Days_T_S_flow_ChargeModel = np.zeros((len(trajs_multi_days), 24, G.number_of_nodes()))
# Days_T_S_load_ChargeModel = np.zeros((len(trajs_multi_days), 24, G.number_of_nodes()))
#
#
# for d in range(len(temperatures)):
#     evs = []
#     evses = []
#     EV_T = np.zeros((len(trajs_multi_days[d]), tf//dt))
#
#     for j in range(G.number_of_nodes()):
#         evses.append(EVSE_class(efficiency=0.90, Prated_kW=80,
#                                 evse_id=j))
#         evses[j].receive_from_server(setpoint_kW=80)
#
#     for j in range(len(trajs_multi_days[d])):
#         tmpTime = round(trajs_multi_days[d][j].time_seq_cut[-1]*3600)
#         tmpSoc = trajs_multi_days[d][j].SOC_seq_cut[-1]/100
#         tmpChargeid = trajs_multi_days[d][j].path_cut[-1] - 1
#         evs.append(ElectricVehicles(arrival_time=tmpTime,
#                                     initial_soc=tmpSoc,
#                                     target_soc=0.9499,
#                                     batterycapacity_kWh=40.0))
#         evs[j].assign_evse(tmpChargeid)
#     Pmax = 0.0
#     for t in np.arange(t0, tf+3600, dt):
#         if t % 600 == 0:
#             print('='*20, ' time ', t//60, 'min', ' Day ', d)
#         for j in range(len(trajs_multi_days[d])):
#
#             evs[j].chargevehicle(t, dt=dt, evsePower_kW=Pmax)
#                 ### EV -> EVSE
#             if evs[j].pluggedin and evs[j].readytocharge:
#                 evses[0].receive_from_ev(evs[j].packvoltage, evs[j].packpower,
#                                               evs[j].soc, evs[j].pluggedin, evs[j].readytocharge)
#                 ### EVSE -> EV
#                 Pmax = evses[0].send_to_ev()
#                 # EV_T[j, t//dt] = evses[0].ev_power / evses[0].efficiency
#                 if t//dt < tf // dt:
#                     EV_T[j, t//dt] = evses[0].ev_power
#                 else:
#                     EV_T[j, (t-tf) // dt] = evses[0].ev_power
#     EV_T_load = EV_T.reshape((len(evs), 24, -1)).sum(axis=2) / 1000 * dt/3600# 压缩成24小时，W->kWh
#     EV_T_flow = EV_T
#     EV_T_flow[np.where(EV_T != 0)] = 1
#     no_zero = np.where(EV_T.sum(axis=-1) != 0)[0]
#     EV_T_flow[no_zero,:] = EV_T_flow[no_zero,:] / (np.tile(EV_T_flow[no_zero,:].sum(axis=-1).reshape(-1, 1), (1, tf//dt)))
#     EV_T_flow = EV_T_flow.reshape((len(evs), 24, -1)).sum(axis=2)
#     for j in range(len(evs)):
#         Days_T_S_load_ChargeModel[d, :, evs[j].evse_id] += EV_T_load[j, :]
#         Days_T_S_flow_ChargeModel[d, :, evs[j].evse_id] += EV_T_flow[j, :]
#
# t2 = time.time() - timer0
# print('t1: ', t1)
# print('t2: ', t2)

# with open("./results_ChargeModel/results_ChargeModel.pkl", "wb") as file:
#     cloudpickle.dump([Days_T_S_load_ChargeModel, Days_T_S_flow_ChargeModel], file, True)

with open("./results_ChargeModel/results_ChargeModel.pkl", "rb") as file:
    tmp = cloudpickle.load(file)
    Days_T_S_load_ChargeModel, Days_T_S_flow_ChargeModel = tmp[0], tmp[1]


# -------------------- statistics of simulation results -------------------- #
Days_T_S_SOC_count = np.zeros((len(trajs_multi_days), 24, G.number_of_nodes(), 10))
Days_T_S_demand = np.zeros((len(trajs_multi_days), 24, G.number_of_nodes()))
Days_finial_SOC = [[]] * len(trajs_multi_days)

for i in range(len(trajs_multi_days)):
    for j in range(len(trajs_multi_days[i])):
        if trajs_multi_days[i][j].time_seq_cut[-1] < 24:
            T_index = int(trajs_multi_days[i][j].time_seq_cut[-1] // 1)
        else:
            T_index = 23
        S_index = trajs_multi_days[i][j].path_cut[-1] - 1
        if trajs_multi_days[i][j].SOC_seq_cut[-1] < 100:
            SOC_index = int(trajs_multi_days[i][j].SOC_seq_cut[-1] // 10)
        else:
            SOC_index = 9
        Days_T_S_SOC_count[i, T_index, S_index, SOC_index] += 1
        Days_T_S_demand[i, T_index, S_index] += \
            max((95 - trajs_multi_days[i][j].SOC_seq_cut[-1]), 0) * trajs_multi_days[i][j].battery_cap * 0.01
        Days_finial_SOC[i].append(trajs_multi_days[i][j].SOC_seq_cut[-1])
Days_T_SOC_count = Days_T_S_SOC_count.sum(axis=2)
Days_T_SOC_prob = np.zeros((len(trajs_multi_days), 12, 10))
for i in range(Days_T_SOC_prob.shape[0]):
    for j in range(Days_T_SOC_prob.shape[1]):
        Days_T_SOC_prob[i, j, :] = (Days_T_SOC_count[i, 2*j, :]
                                    + Days_T_SOC_count[i, 2*j+1, :]) / Days_T_SOC_count[i, :, :].sum()

Days_T_S_traffic_flow = Days_T_S_SOC_count.sum(axis=-1) # traffic flow (unit: vehicles per hour)

Days_T_S_demand1 = Days_T_S_demand
Days_T_S_traffic_flow1 = Days_T_S_traffic_flow
Days_T_S_demand1[np.where(Days_T_S_demand1 == 0)] = 0
Days_T_S_traffic_flow1[np.where(Days_T_S_traffic_flow1 == 0)] = 1e-6
Days_T_S_demand_per_car = np.divide(Days_T_S_demand1, Days_T_S_traffic_flow1) # average charging demand
# Days_T_S_demand_per_car = Days_T_S_demand.sum(axis=0) / Days_T_S_traffic_flow.sum(axis=0)


# ----------------- plot v.s. --------------------- #
print(Days_T_S_flow_ChargeModel[0, :, :].sum())
print(Days_T_S_traffic_flow[0, :, :].sum())
print(Days_T_S_load_ChargeModel[0, :, :].sum())
print(Days_T_S_demand[0, :, :].sum())

import plot_results
plot_results.charging_demand_between(Days_T_S_load_ChargeModel)
plot_results.charging_demand_between_vsCalculation(Days_T_S_load_ChargeModel, Days_T_S_demand)
plot_results.traffic_flow_between(Days_T_S_flow_ChargeModel)
plot_results.charging_demand_between_vsEVI(Days_T_S_load_ChargeModel)
plot_results.charging_demand_between_vsEVI_temperatures(Days_T_S_load_ChargeModel)

plot_results.fit_tmeperature_loads(temperatures, Days_T_S_load_ChargeModel)
plot_results.effi()





# # -------------------- plot simulation results -------------------- #
# fig, axs = plt.subplots(3, 4, sharex=False, sharey=False)
# x = np.arange(5, 100, 10)
# T_SOC_prob = T_S_SOC_count.sum(axis=1) / T_S_SOC_count.sum(axis=1).sum() # data from Beijing 2023
#
# for i in range(12):
#     for j in range(Days_T_SOC_prob.shape[0]):
#         if j == 0:
#             axs[i//4, i%4].scatter(x, Days_T_SOC_prob[j, i, :], color='b', s=8, label='Simulation results') # simulation data
#         else:
#             axs[i // 4, i % 4].scatter(x, Days_T_SOC_prob[j, i, :], color='b', s=8)  # simulation data
#     axs[i//4, i%4].scatter(x, arrive_SOC_matrix[i, :], color='r', s=10,
#                 label='Real data') # real data
#     # axs[i//4, i%4].set_xticks(range(0, 101, 25))
#     if i < 11:
#         axs[i // 4, i % 4].set_title(r't$\in$[' + str(i * 2) + ' ,' + str(i * 2 + 2) + ')')
#     else:
#         axs[i // 4, i % 4].set_title(r't$\in$[' + str(i * 2) + ' ,' + str(i * 2 + 2) + ']')
#     if i == 0:
#         axs[i // 4, i % 4].set_xlabel('SOC (%)')
#         axs[i // 4, i % 4].set_ylabel('Probability')
# # mento carlo simulation results
#
# # plt.xlabel('SOC (%)')
# # plt.ylabel('Probability')
# lines, labels = axs[0, 0].get_legend_handles_labels()
# # 虽然show不完整，但是保存是完整的
# fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.05, 0.95, 1, 0.1), fontsize=12)
# plt.tight_layout()
# plt.show()
# fig.savefig('./results_ChargeModel/charging_pro_simulation_v2.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
#
# # # <editor-fold desc="plot arrive_SOC_T_pro">
# # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
# #           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9edae5', '#ffbb78']
# # fig, ax = plt.subplots()
# # x = np.arange(5, 100, 10)
# #
# # for i in range(12):
# #     if i < 11:
# #         ax.plot(x, arrive_SOC_matrix[i, :], color=colors[i],
# #                    label='t$\in$[' + str(i * 2) + ' ,' + str(i * 2 + 2) + ')')  # plot posterior prob
# #     else:
# #         ax.plot(x, arrive_SOC_matrix[i, :], color=colors[i],
# #                    label='t$\in$[' + str(i * 2) + ' ,' + str(i * 2 + 2) + ']')  # plot posterior prob
# # plt.xlabel('SOC (%)', size=12)
# # plt.ylabel('Probability', size=12)
# # plt.xticks(size=12)
# # plt.yticks(size=12)
# # ax.set_xticks(range(0,101,10))
# # ax.legend(loc='center left', ncol=1, fontsize=12, handlelength=1, labelspacing=0.8, bbox_to_anchor=(1.0, 0.5))
# # plt.show()
# # # 现实不完美，保存的pdf是完美的
# # fig.savefig('./results_ChargeModel/arrive_T_SOC_prob.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
# # # </editor-fold>
#
# Days_T_S_traffic_flow = Days_T_S_traffic_flow
# Days_T_S_demand = Days_T_S_demand
# # -------------------- plot traffic flow -------------------- #
# fig, axs = plt.subplots(3, 3)
# axis_t = np.arange(24) + 0.5
# for i in range(9):
# # for i in range(Days_T_S_traffic_flow.shape[2]):
#     for j in range(Days_T_S_traffic_flow.shape[0]):
#         axs[i//3, i%3].plot(axis_t, Days_T_S_traffic_flow[j, :, i])
#         if i == 6:
#             axs[i // 3, i % 3].set_xlabel('Time (hour)')
#             axs[i // 3, i % 3].set_ylabel('Traffic flow (vehicles per hour)')
# # for j in range(Days_T_S_traffic_flow.shape[0]):
# #     axs[4, 4].plot(axis_t, Days_T_S_traffic_flow[j, :, :].sum(-1))
# plt.tight_layout()
# plt.show()
#
# fig.savefig('./results_ChargeModel/traffic_flow_different_nodes.pdf', bbox_inches='tight'
#             , pad_inches=0, dpi=600)
#
# fig, ax = plt.subplots()
# for j in range(Days_T_S_traffic_flow.shape[0]):
#     ax.plot(axis_t, Days_T_S_traffic_flow[j, :, :].sum(-1))
# ax.set_xlabel('Time (hour)', fontsize=12)
# ax.set_ylabel('Traffic flow (vehicles per hour)', fontsize=12)
# plt.tick_params(labelsize=12)
# plt.show()
# fig.savefig('./results_ChargeModel/traffic_flow.pdf', bbox_inches='tight'
#             , pad_inches=0, dpi=600)
#
# # -------------------- plot charging demand -------------------- #
# fig, axs = plt.subplots(3, 3)
# axis_t = np.arange(24) + 0.5
# # for i in range(Days_T_S_demand.shape[2]):
# for i in range(9):
#     for j in range(Days_T_S_demand.shape[0]):
#         axs[i//3, i%3].plot(axis_t, Days_T_S_demand[j, :, i])
#         if i== 6:
#             axs[i // 3, i % 3].set_xlabel('Time (hour)')
#             axs[i // 3, i % 3].set_ylabel('Charging demand (kWh)')
# # for j in range(Days_T_S_demand.shape[0]):
# #     axs[4, 4].plot(axis_t, Days_T_S_demand[j, :, :].sum(-1))
# plt.tight_layout()
# plt.show()
# fig.savefig('./results_ChargeModel/charging_demand_different_nodes.pdf', bbox_inches='tight'
#             , pad_inches=0, dpi=600)
#
# fig, ax = plt.subplots()
# for j in range(Days_T_S_demand.shape[0]):
#     ax.plot(axis_t, Days_T_S_demand[j, :, :].sum(-1))
# ax.set_xlabel('Time (hour)', fontsize=12)
# ax.set_ylabel('Charging demand (kWh)', fontsize=12)
# plt.tick_params(labelsize=12)
# plt.show()
# fig.savefig('./results_ChargeModel/charging_demand.pdf', bbox_inches='tight'
#             , pad_inches=0, dpi=600)
#
# fig, ax = plt.subplots()
# for j in range(Days_T_S_demand.shape[0]):
#     if j == 22:
#         ax.plot(axis_t, Days_T_S_demand[j, :, :].sum(-1), label=r'-8.2$^{\circ}C$', color='r')
#     elif j == 12:
#         ax.plot(axis_t, Days_T_S_demand[j, :, :].sum(-1), label=r'25.0$^{\circ}C$', color='b')
#     else:
#         ax.plot(axis_t, Days_T_S_demand[j, :, :].sum(-1), alpha=0.3)
# ax.set_xlabel('Time (hour)')
# ax.set_ylabel('Charging demand (kWh)')
# plt.legend()
# plt.show()
# fig.savefig('./results_ChargeModel/charging_demand_hot_cold.pdf', bbox_inches='tight'
#             , pad_inches=0, dpi=600)
#
# # look the impact of temperatures
# Days_traffic = []
# Days_demand = []
# L = Days_T_S_traffic_flow.shape[0]
# for i in range(L):
#     Days_traffic.append(Days_T_S_traffic_flow[i, :, :].sum())
#     Days_demand.append(Days_T_S_demand[i, :, :].sum())
# Days_traffic = np.array(Days_traffic).reshape((L, -1))
# Days_demand = np.array(Days_demand).reshape((L, -1))
# temperaturess = temperatures.reshape((L, -1))
# Days_demand_devide_traffic = Days_demand / Days_traffic
# aa = np.concatenate((Days_traffic, Days_demand, temperaturess, Days_demand_devide_traffic), axis=1)
# print(aa)
#
# def Curve_Fitting(x,y,deg):
#     parameter = np.polyfit(x, y, deg)  # 拟合deg次多项式
#     p = np.poly1d(parameter)  # 拟合deg次多项式
#     aa = ''  # 方程拼接  ——————————————————
#     for i in range(deg + 1):
#         bb = round(parameter[i], 2)
#         if bb > 0:
#             if i == 0:
#                 bb = str(bb)
#             else:
#                 bb = '+' + str(bb)
#         else:
#             bb = str(bb)
#         if deg == i:
#             aa = aa + bb
#         else:
#             if deg - i == 1:
#                 aa = aa + bb + 'x'
#             else:
#                 aa = aa + bb + 'x^' + str(deg - i)  # 方程拼接  ——————————————————
#     fig, ax = plt.subplots()
#     ax.plot(x, p(x), color='r')  # 画拟合曲线
#     ax.scatter(x, y)  # 原始数据散点图
#
#     # plt.text(-1,0,aa,fontdict={'size':'10','color':'b'})
#     plt.legend([aa + r' (R$^2: $' + str(round(np.corrcoef(y, p(x))[0, 1] ** 2, 2)) + ')', 'Data from different days'], fontsize=12)
#     plt.xlabel(r'Temperature ($^{\circ}C$)', size=12)
#     plt.ylabel(r'Average charging demand per vehicle (kWh)', size=12)
#     plt.show()
#     fig.savefig('./results_ChargeModel/relationship_temp_demand.pdf'
#                 , bbox_inches='tight', pad_inches=0, dpi=600)
# Curve_Fitting(aa[:, -2], aa[:, -1], 1)
#
#
# # different seasons
# season_traffic = []
# season_demand = []
# season_labels = ['Spring', 'Summer', 'Fall', 'Winter']
#
# fig, ax = plt.subplots()
# for i in range(4):
#     ax.plot(axis_t, (Days_T_S_traffic_flow[i*6:i*6+6, :, :].sum(axis=-1)).sum(axis=0), label=season_labels[i])
#     season_traffic.append(Days_T_S_traffic_flow[i*6:i*6+6, :, :].sum())
# plt.legend()
# plt.show()
# fig.savefig('./results_ChargeModel/traffic_flow_seasons.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
#
# fig, ax = plt.subplots()
# for i in range(4):
#     ax.plot(axis_t, (Days_T_S_demand[i*6:i*6+6, :, :].sum(axis=-1)).sum(axis=0), label=season_labels[i])
#     season_demand.append(Days_T_S_demand[i*6:i*6+6, :, :].sum())
# plt.legend()
# plt.show()
# fig.savefig('./results_ChargeModel/charging_demand_seasons.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
#
# # save data
#
# with open("./results_ChargeModel/data_to_robust_location.pkl", "wb") as file:
#     cloudpickle.dump([G, Days_T_S_traffic_flow, Days_T_S_demand, Days_T_S_demand_per_car], file, True)
#
# with open("./results/data_to_robust_location.pkl", "rb") as file:
#     tmp = cloudpickle.load(file)
#     G, Days_T_S_traffic_flow, Days_T_S_demand, Days_T_S_demand_per_car \
#         = tmp[0], tmp[1], tmp[2], tmp[3]
#
#
#
# print('1111')
#
# # # final SOC
# # ass = []
# # bs = []
# # cs = []
# # for i in range(12):
# #     print(charging_pro_logit[i](95))
# #     ass.append(charging_pro_logit[i](90))
# #     bs.append(charging_pro_logit[i](95))
# #     cs.append(charging_pro_logit[i](100))
# # ass = np.array(ass).reshape([12, 1])
# # bs = np.array(bs).reshape([12, 1])
# # cs = np.array(cs).reshape([12, 1])
# # abcs = np.concatenate([ass, bs, cs], axis=1)
# #
# # fig, axs = plt.subplots(6,4)
# # for i in range(len(trajs_multi_days)):
# #     axs[i//4, i%4].hist(Days_finial_SOC[i], bins=100)
# # plt.show()
#


