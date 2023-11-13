import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from psth import time_histogram

def spikes2bin(x, y, win_size):
    bin_num = y.shape[0]
    channel = x.shape[1]
    input_records = torch.zeros((bin_num, channel))
    for win in range(bin_num):
        start_T = win_size * win
        end_T = win_size * (win + 1)
        spike_train = x[start_T:end_T, :]
        inp = torch.sum(spike_train, dim=0)
        input_records[win] = inp
    return input_records

def draw(dt=1):
    """
    Plot the 

    Parameters
    ------
    dt: float
        bin time length(ms),to cal velocity 其实是没用到的可以删掉

    Returns
    ------
    Tuple(np.ndarray, np.ndarray)
        hist: np.ndarray
            The values of the histogram.
        bin_edges: np.ndarray
            Time bins of ``(length(hist)+1)``.
    """
    data = np.load('arrays.npz')
    pred = data['pred']
    y = data['y']
    spikes_record = data['spikes_record']
    X_train = data['X_train']
    X_train = torch.from_numpy(X_train)
    
    ##输入1
    pos_xy = y # win_num * 2    
    #spikes_record_binned = spikes2bin(spikes_record,pred,win_size).T # N * win_num
    spikes_record_binned = spikes2bin(X_train,pred,12).T # N * win_num
    ##输入2
    spikes_record_binned = spikes_record_binned.numpy() 
    
    ###你可以认为下面才是对你这个有用的部分，上面只是数据读入，spikes_record_binned 使用 time_histogram 处理得到
    ###因为我这个spike不是时间戳的形式，是01的，所以分bin的方式不一样，你那个直接用time_histogram就行了
    
    #Wprint(spikes_record_binned.shape)
    v_xy = np.diff(pos_xy,axis=0)/dt # (win_num-1) * 2
    dir_angle = np.arctan(v_xy[:,1]/v_xy[:,0])
    pos = v_xy[:,1] < 0
    dir_angle[pos] += np.pi
    #print(dir_angle.shape)
    n_dirs = 8
    ans = np.zeros(dir_angle.shape[0])
    #print(np.max(dir_angle)/np.pi,np.min(dir_angle)/np.pi)
    bins_allv = []
    bins_v_average = []
    for i in range(n_dirs):
        angle1 = 2*np.pi/n_dirs*i-np.pi/2
        angle2 = 2*np.pi/n_dirs*(i+1)-np.pi/2
        pos = np.logical_and((dir_angle>=angle1),(dir_angle<angle2))
        #ans += pos
        allv = spikes_record_binned[:,:-1][:,pos] # (N, N_dirs_in_bin)
        print(allv.shape)
        v_average = np.mean(allv,axis=1) # (N, )
        bins_allv.append(allv)
        bins_v_average.append(v_average)
    #总的速率
    bins_v_average_in_one = bins_v_average[0].reshape(-1,1)

    for i in range(1,n_dirs):
        bins_v_average_in_one = np.concatenate((bins_v_average_in_one,bins_v_average[i].reshape(-1,1)),1)

    print(bins_v_average_in_one.shape)
    #savemat(os.path.dirname(os.path.abspath(__file__))+"/bins_v_average_in_one.mat",{"bins_v_average_in_one":bins_v_average_in_one})

    #tuning curve
    dir_angle = (dir_angle/np.pi + 0.5) * np.pi
    fr = spikes_record_binned[:,:-1] # N * win_num

    from scipy.optimize import curve_fit
    def cal_R2(y,pred_y):
        yhat = pred_y                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        sserr = np.sum((y-yhat)**2)
        return 1 - sserr / sstot
    
    for i in range(fr.shape[0]):
        # print(i)
        if i>=100:
            break
        np.random.seed(0)  # 为了结果的可重复性
        # x_data = dir_angle  
        # y_data = fr[0,:]  
        x_data = np.arange(n_dirs)/n_dirs*2*np.pi
        y_data = bins_v_average_in_one[i,:]


        # 定义拟合的函数形式，这里我们用一个余弦函数
        def cos_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.cos(frequency * x + phase) + offset

        # 使用curve_fit进行拟合
        # params, params_covariance = curve_fit(cos_func, x_data, y_data, p0=[2, 1, 0, 0],bounds=([-np.inf,1,-np.inf,-np.inf],[np.inf,1,np.inf,np.inf]))
        # break
        try:
            params, params_covariance = curve_fit(cos_func, x_data, y_data, p0=[2, 1, 0, 0],bounds=([-np.inf,0,-np.inf,-np.inf],[np.inf,1,np.inf,np.inf]))
        except:
            print(i)
            plt.subplot(10, 10, i+1)
            plt.scatter(x_data, y_data, label='Data')
            plt.xticks(ticks=[0, np.pi, 2*np.pi])
            plt.xlim(0,2*np.pi)
            continue
        # if params[1]>1:
        #     paramsmaxr2 = []
        #     maxr2 = 0
        #     frmaxr2 = -1
        #     for fre in np.arange(0,1,0.1):
        #         print("fr=%.2f"%fre,end='')
        #         def cos_func1(x, amplitude, phase, offset):
        #              return amplitude * np.cos(fre * x + phase) + offset
        #         try:
        #             params, params_covariance = curve_fit(cos_func1, x_data, y_data, p0=[2, 1.5, 0, 0])
        #         except:
        #             print("no",end='')
        #             continue
        #         r2 = cal_R2(y_data,cos_func1(x_data,*params))
        #         if r2 > maxr2:
        #             maxr2 = r2
        #             paramsmaxr2 = params
        #             frmaxr2 = fre
        #     params = paramsmaxr2
        #     params.insert(1,frmaxr2)
        #     if frmaxr2==-1:
        #         continue

        r2 = cal_R2(y_data,cos_func(x_data,*params))
        
        # 绘制数据点和拟合的函数

        plt.subplot(10, 10, i+1)
        #plt.figure(figsize=(10, 5))
        plt.scatter(x_data, y_data, label='Data')
        xx = np.arange(0,2*np.pi,np.pi/100)
        if r2>0.8:
            plt.plot(xx, cos_func(xx, *params), label='Fitted function', color='red')
        else:
            plt.plot(xx, cos_func(xx, *params), label='Fitted function', color='black')

    

        # 标注图形
        # plt.legend(loc='best')
        # plt.xlabel('v direction')
        # plt.ylabel('firing rate')
        plt.title('R2:'+str(r2)[:4], fontsize=8,loc='left',pad=1)
        plt.xticks(ticks=[0, np.pi, 2*np.pi],labels=['0','π','2π'])
        plt.xlim(0,2*np.pi)
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)
        # 显示图形
    plt.show()
    return 1,1


cc,dis = draw()




