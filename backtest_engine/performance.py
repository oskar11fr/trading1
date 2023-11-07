import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("seaborn-dark-palette")

def performance_measures(r, plot=False, path="/images"):
    moment = lambda x,k: np.mean((x-np.mean(x))**k)
    stdmoment = lambda x,k: moment(x,k)/moment(x,2)**(k/2)
    cr = np.cumprod(1 + r)
    lr = np.log(cr)
    mdd=cr/cr.cummax() - 1
    rdd_fn = lambda cr,pr: cr/cr.rolling(pr).max() - 1
    rmdd_fn = lambda cr,pr: rdd_fn(cr,pr).rolling(pr).min()
    srtno = np.mean(r.values)/np.std(r.values[r.values<0])*np.sqrt(253)
    shrpe = np.mean(r.values)/np.std(r.values)*np.sqrt(253)
    mu1 = np.mean(r)*253
    med = np.median(r)*253
    stdev = np.std(r)*np.sqrt(253)
    var = stdev**2
    skw = stdmoment(r,3)
    exkurt = stdmoment(r,4)-3
    cagr_fn = lambda cr: (cr[-1]/cr[0])**(1/len(cr))-1
    cagr_ann_fn = lambda cr: ((1+cagr_fn(cr))**253) - 1
    cagr = cagr_ann_fn(cr)
    rcagr = cr.rolling(5*253).apply(cagr_ann_fn,raw=True)
    calmar = cr.rolling(3*253).apply(cagr_ann_fn,raw=True) / rmdd_fn(cr=cr,pr=3*253)*-1
    var95 = np.percentile(r,0.95)
    cvar = r[r < var95].mean()
    table = {
        "cum_ret": cr,
        "log_ret": lr,
        "max_dd": mdd,
        "cagr": cagr,
        "srtno": srtno,
        "sharpe": shrpe,
        "mean_ret": mu1,
        "median_ret": med,
        "vol": stdev,
        "var": var,
        "skew": skw,
        "exkurt": exkurt,
        "cagr": cagr,
        "rcagr": rcagr,
        "calmar": calmar,
        "var95": var95,
        "cvar": cvar
    }
    if plot:
        import os
        from pathlib import Path
        Path(os.path.abspath(os.getcwd()+path)).mkdir(parents=True,exist_ok=True)
        f1,axs1 = plt.subplots(1,2,figsize=(15,11),width_ratios=[3, 1],sharey=True)
        axs1[1].hist(r,orientation='horizontal',bins=40)
        axs1[0].plot(r)
        axs1[0].set_ylabel('Returns')
        # axs1[1].axhline(x=np.median(r), linestyle="dotted")
        # axs1[0].axhline(x=np.mean(r), linestyle="dashed")
        f1.savefig(f".{path}/rets_dist.png")
        plt.close()

        fig = plt.figure(constrained_layout=True,figsize=(15,11))
        ax = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(ax[0:2, 0:2])
        ax2 = fig.add_subplot(ax[2:, 0:2],sharex=ax1)
        ax3 = fig.add_subplot(ax[0:2, -1])
        ax4 = fig.add_subplot(ax[-1, -1],sharey=ax2)
        
        ax1.plot(lr)
        ax1.set_ylabel('log capital returns')

        ax2.plot(rdd_fn(cr,253))
        ax2.plot(rmdd_fn(cr,253))
        ax2.set_ylabel('drawdowns')

        pd_series = pd.Series(table)[["cagr",
                                        "srtno",
                                        "sharpe",
                                        "mean_ret",
                                        "median_ret",
                                        "vol",
                                        "var",
                                        "skew",
                                        "exkurt",
                                        "cagr",
                                        "var95"]].apply(lambda x:np.round(x,3))
        
        pd_frame = pd_series.reset_index().rename(columns={'index':'Metric',0:'Value'})
        ax3.table(
            cellText=pd_frame.values,colLabels=pd_frame.keys(),loc='center',colWidths=[0.3,0.3],colColours=['grey','grey'],cellLoc='left'
        )
        ax3.axis('off')
        
        ax4.hist(rdd_fn(cr,253),orientation='horizontal',bins=40)

        fig.savefig(f".{path}/log_ret.png")
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.close()

    return table