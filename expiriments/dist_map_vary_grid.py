from base_cons_p import *


save_dir = f'./results/dist2boundary/'
os.makedirs(save_dir, exist_ok=True)

GRID_list = [64, 128, 256, 512]


results = pd.DataFrame()
for data_name, dataset in datasets_real.items():
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
        for GRID in GRID_list:
            time0 = time.time()
            label_gt, dist_gt, _ = mapbuilder.get_map(content='dist_map', resolution=GRID, fast_strategy=False)
            time1 = time.time()
            time_gt = time1 - time0

            time2 = time.time()
            label_fast, dist_fast, sparse = mapbuilder.get_map(content='dist_map', resolution=GRID, fast_strategy=True)
            time3 = time.time()
            time_fast = time3 - time2

            error_abs = np.abs(dist_gt - dist_fast).mean()
            error_sq = np.sum((dist_gt - dist_fast)**2) / np.sum(dist_gt**2 )

            results = results._append({'Data': data_name, 'PPinv': ppinv_name, 'time dummy': time_gt, 'time fast': time_fast, 'grid': GRID, 'error_abs': error_abs, 'error_sq': error_sq}, ignore_index=True)

            results.to_csv(f'{save_dir}/results{date}.csv', index=False)
