from base_cons_p import *
from time import time
import matplotlib.pyplot as plt

PPinv_dict = {
    # 'DBM(UMAP+NNInv)': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=NNinv_torch()),
    # 'DBM(t-SNE+NNInv)': PPinvWrapper(P=TSNE(n_components=2, random_state=0), Pinv=NNinv_torch()),
    "SSNP": SSNP(verbose=False)
}

### save directory
save_dir = f'./results/dist2boundary/'
os.makedirs(save_dir, exist_ok=True)

GRID = 256
data_name = 'MNIST'
dataset = datasets_real[data_name]
X, _, y, _ = dataset
for ppinv_name, ppinv in PPinv_dict.items():
    print(f"PP: {ppinv_name}")
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    ppinv.fit(X=X, y=y)
    try:
        X2d = ppinv.X2d
    except:
        X2d = ppinv.transform(X)

    mapbuilder = MapBuilder(ppinv=ppinv, clf=None, X=X, y=y, scaling=0.9, X2d=X2d)

    time0 = time()
    # ax = mapbuilder.plot_dist_map(grid=GRID, fast=True, content='dist_map_general')
    ax = mapbuilder.plot_gradient_map(grid=GRID, fast=True)
    # ax.set_title(f'{ppinv_name} {data_name} {GRID}')
    time1 = time()
    time_diff = time1 - time0
    print(f"Time: {time_diff}")

    plt.show()