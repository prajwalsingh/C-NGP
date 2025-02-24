# Reference: https://towardsdatascience.com/how-to-create-a-radar-chart-in-python-36b9ebaa7a64

import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    
    if not os.path.isdir('results/plots'):
        os.makedirs('results/plots', exist_ok=True)

    ##### Random Sampling ########
    datasets   = ['Chair', 'Lego', 'Materials', 'Mic', 'Ship', 'Chair']
    psnr_table = {
                    'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                    'k-64':[34.86, 34.90, 28.24, 34.94, 29.36, 34.86],
                    'k-32':[33.54, 32.38, 25.35, 32.99, 26.74, 33.54],
                    'k-16':[29.38, 27.47, 23.25, 30.60, 21.07, 29.38],
                    'k-8':[22.08, 23.99, 18.11, 25.74, 16.67, 22.08],
                    'k-4':[18.42, 15.02, 14.15, 18.22, 12.68, 18.42],
                    'k-2':[15.08, 11.45, 10.49, 12.57, 10.80, 15.08],
                  }
    color = ['tomato', 'green', 'red', 'blue', 'orange', 'yellow', 'pink', 'brown']
    
    angles = np.linspace(0, 2*np.pi, len(datasets)-1, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]], axis=-1) # Completing full circle

    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(polar=True)
    ax.set_thetagrids(angles * 180/np.pi, datasets)
    ax.set_rgrids(np.round(np.linspace(0, 35, num=7), 2), labels=np.round(np.linspace(0, 35, num=7), 2), angle=45, fontsize=9, weight='normal')
    ax.set_rmax(50.0)
    ax.tick_params(pad=10)
    for idx, (key, value) in enumerate(psnr_table.items()):
        ax.plot(angles, value, label=key, marker='o', linestyle='--')
        ax.fill(angles, value, alpha=0.05, color=color[idx])
    # for angle in angles:
    #     ax.set_rgrids(np.linspace(0, 30, num=5), labels=np.linspace(0, 30, num=5), angle=angle, fontsize=8)
    # # ax.xaxis.grid(True,color='black',linestyle='-')
    plt.title('PSNR Synthetic Dataset [Cluster: Random]', fontsize=12, loc='center', y=-0.21)
    plt.grid(True)
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17), fancybox=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/plots/psnr_synthetic_dataset_cluster_random.pdf', bbox_inches='tight', dpi=300)

    ##### Max Sampling ########
    datasets   = ['Chair', 'Lego', 'Materials', 'Mic', 'Ship', 'Chair']
    psnr_table_lst = [
                    {
                        'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                        'k-64':[34.86, 34.90, 28.24, 34.94, 29.36, 34.86],
                        'k-max-64':[35.13, 35.05, 27.96, 35.02, 29.35, 35.13],
                    },
                    {
                        'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                        'k-32':[33.54, 32.38, 25.35, 32.99, 26.74, 33.54],
                        'k-max-32':[33.57, 32.23, 25.27, 32.77, 26.04, 33.57],
                    },
                    {
                        'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                        'k-16':[29.38, 27.47, 23.25, 30.60, 21.07, 29.38],
                        'k-max-16':[29.81, 29.09, 23.31, 30.27, 20.47, 29.81],
                    },
                    {
                        'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                        'k-8':[22.08, 23.99, 18.11, 25.74, 16.67, 22.08],
                        'k-max-8':[23.53, 22.60, 18.38, 27.58, 16.88, 23.53],
                    },
                    {
                        'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                        'k-4':[18.42, 15.02, 14.15, 18.22, 12.68, 18.42],
                        'k-max-4':[17.83, 15.25, 13.95, 20.91, 12.38, 17.83],
                    },
                    {
                        'all-100':[35.01, 35.59, 29.36, 35.80, 29.95, 35.01],
                        'k-2':[15.08, 11.45, 10.49, 12.57, 10.80, 15.08],
                        'k-max-2':[15.13, 11.09, 11.81, 11.21, 10.50, 15.13],
                    },
                  ]
    # color = ['tomato', 'green', 'red', 'blue', 'orange', 'yellow', 'pink', 'brown']
    
    angles = np.linspace(0, 2*np.pi, len(datasets)-1, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]], axis=-1) # Completing full circle

    for cluster_idx, psnr_table in enumerate(psnr_table_lst):
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(polar=True)
        ax.set_thetagrids(angles * 180/np.pi, datasets)
        ax.set_rgrids(np.round(np.linspace(0, 35, num=7), 2), labels=np.round(np.linspace(0, 35, num=7), 2), angle=45, fontsize=9, weight='normal')
        ax.set_rmax(50.0)
        ax.tick_params(pad=10)
        for idx, (key, value) in enumerate(psnr_table.items()):
            ax.plot(angles, value, label=key, marker='o', linestyle='--')
            # ax.fill(angles, value, alpha=0.05, color=color[idx])
            ax.fill(angles, value, alpha=0.05)
        # for angle in angles:
        #     ax.set_rgrids(np.linspace(0, 30, num=5), labels=np.linspace(0, 30, num=5), angle=angle, fontsize=8)
        # # ax.xaxis.grid(True,color='black',linestyle='-')
        plt.title('PSNR Synthetic Dataset [Cluster: Max]', fontsize=12, loc='center', y=-0.21)
        plt.grid(True)
        plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17), fancybox=True)
        plt.tight_layout()
        # plt.show()
        plt.savefig('results/plots/psnr_synthetic_dataset_cluster_max_{}.pdf'.format(cluster_idx+1), bbox_inches='tight', dpi=300)

