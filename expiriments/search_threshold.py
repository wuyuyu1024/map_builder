from base_cons_p import *
from tqdm import tqdm


save_dir = f'./results/search_threshold/'
os.makedirs(save_dir, exist_ok=True)
GRID = 256

# threshold_list = [0.05, 0.075, 0.1, 1.25, 1.5, 1.75, 0.2, 0.225, 0.25, 0.275, 0.3, 0.4, 0.5]
threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

PPinv_dict = {
    'DBM(UMAP+iLAMP)': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=Pinv_ilamp()),
    'DBM(t-SNE+iLAMP)': PPinvWrapper(P=TSNE(n_components=2, random_state=0), Pinv=Pinv_ilamp()),
    'DBM(UMAP+NNInv)': PPinvWrapper(P=UMAP(n_components=2, random_state=0), Pinv=NNinv_torch()),
    'DBM(t-SNE+NNInv)': PPinvWrapper(P=TSNE(n_components=2, random_state=0), Pinv=NNinv_torch()),
    "SSNP": SSNP(verbose=0),
}

results = pd.DataFrame()
# for data_name, dataset in datasets_real.items():
data_name = 'MNIST'
dataset = datasets_real[data_name]
print(f"Data: {data_name}")
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

    time0 = time.time()
    label_gt, nearest_gt, _ = mapbuilder.get_map(content='nearest', resolution=GRID, fast_strategy=False)
    time1 = time.time()
    time_near_gt = time1 - time0


    time_grad_gt = 0
    # time2 = time.time()
    # _, grad_map_gt, _ = mapbuilder.get_map(content='gradient', resolution=GRID, fast_strategy=False)
    # time3 = time.time()
    # time_grad_gt = time3 - time2


    for t in tqdm(threshold_list):
        time0 = time.time()
        _, near_map, sparse_near = mapbuilder.get_map(content='nearest', resolution=GRID, fast_strategy=True, threshold=t)
        time1 = time.time()
        time_near = time1 - time0

        error_near_abs = np.abs(nearest_gt - near_map).mean()
        error__near_sq = np.sum((nearest_gt - near_map)**2) / np.sum(nearest_gt**2 )

        time_grad = 0
        error_grad_abs = 0
        error_grad_sq = 0
        sparse_grad = []

        # time2 = time.time()
        # _, grad_map, sparse_grad = mapbuilder.get_map(content='gradient', resolution=GRID, fast_strategy=True, threshold=t)
        # time3 = time.time()
        # time_grad = time3 - time2

        # error_grad_abs = np.abs(grad_map_gt - grad_map).mean()
        # error_grad_sq = np.sum((grad_map_gt - grad_map)**2) / np.sum(grad_map_gt**2 )

        results = results._append({'Data': data_name, 'PPinv': ppinv_name, 'time near dummy': time_near_gt, 'time near': time_near, 'time grad dummy': time_grad_gt, 'time grad': time_grad, 'threshold': t, 'error_near_abs': error_near_abs, 'error_near_sq': error__near_sq, 'error_grad_abs': error_grad_abs, 'error_grad_sq': error_grad_sq, 'num_sparse_near':len(sparse_near), 'num_sparse_grad': len(sparse_grad)}, ignore_index=True)

        results.to_csv(f'{save_dir}/search_threhold_{date}.csv', index=False)
