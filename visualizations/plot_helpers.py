import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_clusters(cluster_affinity, dim_1, dim_2, fig_name='visualizations/images/scattered_image.jpg'):
    plt.figure(figsize=(8, 6))

    plt.scatter(dim_1, dim_2, c=cluster_affinity, marker='o', edgecolor='none', cmap=discrete_cmap(10, 'jet'))
    plt.colorbar(ticks=range(10))
    plt.grid(True)
    plt.savefig(fig_name)
    plt.close('all')

def plot_clusters_3D(cluster_affinity, dim_1, dim_2, dim_3, fig_name='visualizations/images/scattered_image_3D.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dim_1, dim_2, dim_3, c=cluster_affinity, cmap=discrete_cmap(10, 'jet'))
    fig.savefig(fig_name)
    plt.close('all')