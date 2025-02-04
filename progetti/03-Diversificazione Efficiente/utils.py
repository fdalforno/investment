import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick

from matplotlib.collections import LineCollection

def get_geometric_mean(returns:pd.Series,periods_per_year:int) -> float:
    n_periods = returns.shape[0]
    return np.exp(np.sum(np.log(1+returns)) * periods_per_year/n_periods) - 1


def get_max_drawdown(returns: pd.Series) -> float:
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

def get_annualised_sharpe(returns: pd.Series,risk_free_rate:float, periods_per_year: int) -> float:
    ret_est = (returns - risk_free_rate).mean()
    ann_ret_est = ((ret_est+1)**periods_per_year-1)

    std_est = returns.std()
    ann_std_est = std_est*np.sqrt(periods_per_year)

    return ann_ret_est/ann_std_est

def get_summary_statistics(returns: pd.Series, risk_free_rate: float = 0.0, rounding:int = 2, remove_nan:bool = True) -> pd.DataFrame:
    summary = pd.Series()
    
    start_date = returns.index[0].strftime("%d/%m/%Y")
    end_date = returns.index[-1].strftime("%d/%m/%Y")


    annual_count = returns.resample('Y').count().mean()
    num_part = int(annual_count.max())

    print('Summary Statistic Information from ' + start_date + ' to ' + end_date + ':')

    if(returns.isnull().values.any() & remove_nan):
        print('WARNING: Some firms have missing data during this time period!')
        print('Dropping firms: ')
        for Xcol_dropped in list(returns.columns[returns.isna().any()]): print(Xcol_dropped)
        returns = returns.dropna(axis='columns')

  
    summary = pd.DataFrame(index = returns.columns)
    summary['First Valid Date'] = returns.apply(lambda x: x.first_valid_index().strftime("%d/%m/%Y"))
    summary['Total Return(%)'] = np.round((((returns+1).cumprod()-1)*100).iloc[-1] , rounding)
    summary['Average Return(%)'] = np.round(returns.mean()*100, rounding)
    summary['Geometric Mean(%)'] = np.round(returns.apply(get_geometric_mean,periods_per_year=num_part)*100, rounding)
    summary['Annu. Ave Return(%)'] = np.round(((returns.mean()+1)**num_part-1)*100, rounding)
    summary['Annu. Std(%)'] = np.round(returns.std()*np.sqrt(num_part)*100, rounding)
    summary['Annu. Sharpe Ratio'] = np.round(returns.apply(get_annualised_sharpe,risk_free_rate=risk_free_rate, periods_per_year=num_part), rounding)
    summary['Max Drawdown(%)'] = np.round(get_max_drawdown(returns)*100, rounding)
    return summary

def plot_graphical_analysis(d: np.array,partial_correlations: np.array, colors: list,names:np.array,labels:np.array, embedding:np.array,val_max:np.float64,title:str):
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    n_labels = labels.max()

    #For correlation network graph
    fig = plt.figure(1, facecolor='w', figsize=(12, 12))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=500 * d ** 2, c= colors)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r, 
                        norm=plt.Normalize(0, .7 * val_max))
    lc.set_array(values)
    temp = (15 * values)
    temp2 = np.repeat(5, len(temp))
    w = np.minimum(temp, temp2)
    lc.set_linewidths(w)
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label('Strength')

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6))


    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    ftse_df = pd.read_csv('./data/ftse_return.csv',index_col=0,parse_dates=True)
    get_annualised_sharpe(ftse_df['PST'],0.0,252)