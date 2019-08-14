import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import cycle

parser = argparse.ArgumentParser('Plot columns from log files (csv or Panda) spread across multiple directories')
parser.add_argument('--files', type=str, required=True,nargs='+',
        help='List of files (csv or pickled Panda) to open')
parser.add_argument('--savefile', type=str, default=None,
        help='where to save file')
parser.add_argument('--xname', type=str, required=True,
        help='x axis column name')
parser.add_argument('--yname', type=str, default=None,
        help='y axis column name. If not specified, a CDF of the x variable is plotted.')
parser.add_argument('--labels', type=str, nargs='*',default=None,
        help='legend labels')
parser.add_argument('--xlabel', type=str, default=None,
        help='x axis label')
parser.add_argument('--ylabel', type=str, default=None,
        help='y axis label')
parser.add_argument('--title', type=str, default=None,
        help='figure title')
parser.add_argument('--rolling-window', type=int, default=0,
        help='rolling average window (default: 0)')
parser.add_argument('--quantile', type=float, default=0.,
        help='plot quantile error bars at this quantile  (default: 0)')
parser.add_argument('--scale', type=float, default=1.,
        help='scale x axis  (default: 1)')
parser.add_argument('--show', action='store_true', default=False, help='show the plot')
parser.add_argument('--sort', action='store_true', default=False, help='sort data by xname before plotting')
parser.add_argument('--logy', action='store_true', default=False, help='log y axis')
parser.add_argument('--logx', action='store_true', default=False, help='log x axis')
parser.add_argument('--xlim', type=float, nargs='*', default=None)
parser.add_argument('--ylim', type=float, nargs='*', default=None)
parser.add_argument('--figsize', type=float, nargs='*', default=(5,5))
parser.add_argument('--fontsize', type=float, default=16.)
parser.add_argument('--lines',nargs='*', type=str, default=None)
parser.add_argument('--colors',nargs='*', type=str, default=None)


def main():
    args = parser.parse_args()

    plot(args.files, args.xname, args.yname, sort=args.sort,
            labels = args.labels, savefile=args.savefile,
            show=args.show, logx=args.logx, logy=args.logy,
            xlim=args.xlim, ylim=args.ylim, figsize=args.figsize,
            fontsize=args.fontsize, lines=args.lines, colors=args.colors,
            xlabel=args.xlabel, ylabel=args.ylabel, title=args.title,
            rolling_window=args.rolling_window, quantile=args.quantile,
            scale=args.scale)




def plot(files, xname, yname, labels=None, sort=False, 
        savefile=None, show=True, xlabel=None, ylabel=None,
        title=None, rolling_window=0., scale=1, quantile=0.,
        colors=None, lines=None, ylim=None, xlim=None, logx=False,
        logy=False, figsize=(5,5), fontsize=16):
    sns.set_palette(palette='colorblind')
    if colors is None:
        colors = sns.color_palette()
    colorcycler0= cycle(colors)
    colorcycler1= cycle(colors)
    if lines is None:
        lines = ['-']
    linecycler= cycle(lines)

    if yname is None:
        sort=True
        

    fsz = fontsize
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=0.75*fsz)
    plt.rc('figure', titlesize=fsz)

    fig, ax = plt.subplots(1, figsize=(5,5))


    for f in files:
        _, ext = os.path.splitext(f)
        if ext=='.csv':
            df = pd.read_csv(f)
        elif ext=='.pkl':
            df = pd.read_pickle(f)

        if sort:
            df = df.sort_values(xname)

        if rolling_window>0 and yname is not None:
            df[xname] = pd.to_timedelta(df[xname], unit='s')
            df = df.set_index(xname)
            df = df[yname]
            rs = df.resample(str(rolling_window)+'s')
            y = rs.mean()
            x = y.index / np.timedelta64(1, 's')
        else:
            x = df[xname]
            if yname is not None:
                y = df[yname]
            else:
                y = np.arange(len(x), dtype=np.float)/len(x)
        ax.plot(x*scale, y, next(linecycler),lw=1.5,  color=next(colorcycler0))

        if quantile>0 and rolling_window>0 and yname is not None:

            if quantile==0.:
                yu = rs.max()
                yl = rs.min()
            else:
                yu = rs.apply(lambda x: x.quantile(1-quantile))
                yl = rs.apply(lambda x: x.quantile(quantile))

            ax.fill_between(x*scale, yl, yu, alpha=0.2, color=next(colorcycler1))

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if logx:
        ax.set_xscale('log',nonposx='clip')
    if logy:
        ax.set_yscale('log',nonposy='clip')

    if labels:
        ax.legend(labels)
    else:
        ax.legend(files)
    ax.grid()
    extras = []
    if xlabel:
        extras.append(ax.set_xlabel(xlabel))
    else:
        extras.append(ax.set_xlabel(xname))
    if ylabel:
        extras.append(ax.set_ylabel(ylabel))
    else:
        extras.append(ax.set_ylabel(yname))
    if title:
        extras.append(ax.set_title(title))

    if show:
        plt.show()

    if savefile:
        dirname = os.path.dirname(savefile)
        os.makedirs(dirname, exist_ok=True)
        fig.savefig(savefile, format='pdf', bbox_extra_artists=extras,
                bbox_inches='tight', dpi=600)

    return fig, ax

if __name__=="__main__":
    main()
