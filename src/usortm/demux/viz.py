import numpy as np
import matplotlib.pyplot as plt

def plot_quality_hist(reads, means):
    """
    """
    # Compute the 10th percentile of the mean qualities
    plt_q_10 = np.quantile(means, 0.1)

    # Plot
    ax = plt.subplot()

    ax.hist(means,bins=50)
    ax.axvspan(plt_q_10,max(means),0,870,color='green',zorder=-10, alpha=0.2)
    ax.set_xlabel('Mean Q Score')
    ax.set_xlim(0,50)
    ax.set_ylabel('Count')
    ax.set_yticklabels([f"{int(x):,}" for x in ax.get_yticks()])

    # Get N reads string:
    if (len(reads) >= 1000) and (len(reads) < 1000000):
        n_reads_str = f'N reads = {len(reads)/1000:.1f}k'
    elif len(reads) >= 1000000:
        n_reads_str = f'N reads = {len(reads)/1000000:.1f}M'
    else:
        n_reads_str = f'N reads = {len(reads)}'

    ax.text(s=n_reads_str,x=0.05,y=0.9,fontdict={'fontsize':10}, transform=plt.gca().transAxes)
    ax.text(s=f'90% above Q{int(plt_q_10)}',x=0.05,y=0.85,fontdict={'fontsize':9}, color='green', transform=plt.gca().transAxes)

    return ax