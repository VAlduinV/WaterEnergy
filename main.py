if __name__ == '__main__':
    # from three_sigma import *
    # from map import MapPlotter
    # from k_method_default import *
    import logging
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    from sklearn.metrics import adjusted_rand_score

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    file_path = r'.\data\VILLAGE.xlsx'
    df = pd.read_excel(file_path)
    logging.info(df.head())
    logging.info(df[24:32])
    '''
    n_samples - загальна кількість зразків для генерації.
    centers – це кількість центрів для генерації.
    cluster_std - стандартне відхилення.
    '''
    features, true_labels = make_blobs(
        n_samples=len(df[['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR', 'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_',
                          'Water_NEAR', 'occup_NEAR', 'power_NEAR']]), centers=9, cluster_std=2.75, random_state=42)

    # Використання перших чотирьох стовпців даних для генерації міток
    logging.info(f'{df.iloc[:, 24:32].values}')
    logging.info(true_labels)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    logging.info(f'Scaled_features: {scaled_features}')

    kmeans = KMeans(
        init="k-means++",
        n_clusters=9,
        n_init=30,
        max_iter=1000,
        random_state=42)
    logging.info(f'kmeans.fit(scaled_features): {kmeans.fit(scaled_features)}')
    # The lowest SSE value
    logging.info(f'kmeans.inertia_: {kmeans.inertia_}')
    # Final locations of the centroid
    logging.info(f'kmeans.cluster_centers_: {kmeans.cluster_centers_}')
    # The number of iterations required to converge
    logging.info(f'kmeans.n_iter: {kmeans.n_iter_}')
    logging.info(f'Labels: {kmeans.labels_}')

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 30,
        "max_iter": 1000,
        "random_state": 42,
    }
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        logging.info(f'n_clusters: {k}, SSE: {sse}')
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(16, 5))
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 10), sse, marker="o", color="red")
    plt.xticks(range(1, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(
        range(1, 10), sse, curve="convex", direction="decreasing")

    logging.info(kl.elbow)

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    print(kmeans.labels_)

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        logging.info(f"{k} clusters: {kmeans}")
        logging.info(kmeans.fit(scaled_features))
        score = silhouette_score(scaled_features, kmeans.labels_, metric='euclidean')
        silhouette_coefficients.append(score)

    plt.figure(figsize=(19.2, 16.5))
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 10), silhouette_coefficients, marker='o')
    plt.xticks(range(2, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

    features, true_labels = make_moons(n_samples=len(df[['Water_NEAR', 'power_NEAR']]), noise=0.05, random_state=42)
    scaled_features = scaler.fit_transform(features)
    logging.info(f'Scaled Features: {scaled_features}')
    # Instantiate k-means and dbscan algorithms
    kmeans = KMeans(n_clusters=2)
    dbscan = DBSCAN(eps=0.3, min_samples=2)

    # Fit the algorithms to the features
    kmeans.fit(scaled_features)
    dbscan.fit(scaled_features)

    unique_labels = np.unique(dbscan.labels_)
    if len(unique_labels) > 1:
        dbscan_silhouette = silhouette_score(scaled_features, dbscan.labels_).round(2)
        logging.info(f'dbscan_silhouette: {dbscan_silhouette}')
    else:
        logging.warning("DBSCAN found only one cluster or no clusters at all.")

    # Compute the silhouette scores for each algorithm
    kmeans_silhouette = silhouette_score(scaled_features, kmeans.labels_).round(2)
    dbscan_silhouette = silhouette_score(scaled_features, dbscan.labels_).round(2)

    logging.info(f'kmeans_silhouette: {kmeans_silhouette}')
    logging.info(f'dbscan_silhouette: {dbscan_silhouette}')

    # Вибір лише двох параметрів Water_NEAR і power_NEAR
    selected_features = scaled_features

    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 10.8), sharex=True, sharey=True)
    fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
    }
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.scatter(selected_features[:, 0], selected_features[:, 1], c=km_colors, label='Cluster')
    ax1.set_title(f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12})
    ax1.set_xlabel('Water_NEAR')
    ax1.set_ylabel('power_NEAR')
    ax1.legend()

    # The dbscan plot
    db_colors = [fte_colors[label] for label in dbscan.labels_]
    ax2.scatter(selected_features[:, 0], selected_features[:, 1], c=db_colors, label='Cluster')
    ax2.set_title(f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12})
    ax2.set_xlabel('Water_NEAR')
    ax2.set_ylabel('power_NEAR')
    ax2.legend()

    plt.show()

    ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
    ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)

    logging.info(round(ari_kmeans, 2))
    logging.info(round(ari_dbscan, 2))

    # def main():
    #     # Загрузка данных из Excel файла
    #     file_path = r'.\data\VILLAGE.xlsx'
    #     df = pd.read_excel(file_path)
    #     print(df.head())
    #
    #     # Выбор нужных колонок
    #     selected_columns = ['CITY2_NEAR', 'TOWN_NEAR_']
    #     df, selected_data = load_data(file_path, selected_columns)
    #
    #     # Calculate statistics
    #     mean_values, std_values, up_values = calculate_statistics(selected_data)
    #
    #     # Добавление нового столбца с градацией качества жизни в DataFrame
    #     df['Life_Quality_Gradation'] = df['CITY2_NEAR'].apply(lambda x: calculate_gradation(x, up_values))
    #
    #     # Вывод данных
    #     logging.info(f"Average value: {mean_values}")
    #     logging.info(f"STD: {std_values}")
    #     logging.info(df)
    #
    #     plot_histogram(df['CITY2_NEAR'])
    #
    #     optimise_k_means(
    #         df[['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR', 'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_', 'Water_NEAR',
    #             'occup_NEAR', 'power_NEAR']], 10)
    #
    #     n_clusters = 9  # Максимальное количество кластеров
    #     features = ['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR', 'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_', 'Water_NEAR',
    #                 'occup_NEAR', 'power_NEAR']
    #     visualize_clusters(df, features, n_clusters)
    #
    #     range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
    #     target = ['OBJECTID']
    #     kmeans_clustering(df, features, target, range_n_clusters)
    #
    #     path = r'.\data\map_data\gadm41_UKR_1.shp'
    #     plotter = MapPlotter(path)
    #     plotter.plot_map(facecolor='red', edgecolor='black', figsize=(16, 5))
    #
    #
    # main()
    # Load data
    # data = pd.read_excel(r'.\data\VILLAGE.xlsx')
    #
    # # Select relevant columns
    # X = data[["Water_NEAR", "power_NEAR"]]

    # Initialize number of clusters and select random centroids
    # K = 3
    # Centroids = X.sample(n=K)
    #
    # # Plot initial data and centroids
    # plt.scatter(X["power_NEAR"], X["Water_NEAR"], c='black')
    # plt.scatter(Centroids["power_NEAR"], Centroids["Water_NEAR"], c='red')
    # plt.xlabel('power_NEAR')
    # plt.ylabel('Water_NEAR (In Thousands)')
    # plt.title('Initial Data and Centroids')
    # plt.show()
    #
    # # Initialize loop variables
    # diff = 1
    # j = 0
    #
    # # Run K-means algorithm
    # while (diff != 0):
    #     XD = X.copy()  # Make a copy of the original data
    #     for index1, row_c in Centroids.iterrows():
    #         # Calculate Euclidean distance
    #         XD['distance_from_{}'.format(index1)] = (
    #             np.sqrt(
    #                 (XD['power_NEAR'] - row_c['power_NEAR']) ** 2 + (XD['Water_NEAR'] - row_c['Water_NEAR']) ** 2
    #             )
    #         )
    #     # Assign each point to the closest centroid
    #     cluster_label = XD.iloc[:, 2:].idxmin(axis=1)
    #     XD['Cluster'] = cluster_label.map(lambda x: int(x.lstrip('distance_from_')))
    #
    #     # Update centroids
    #     Centroids_new = XD.groupby(['Cluster']).mean()[['Water_NEAR', 'power_NEAR']]
    #
    #     # Calculate difference between old and new centroids
    #     if j == 0:
    #         diff = 1
    #         j += 1
    #     else:
    #         diff = (
    #                 (Centroids_new['Water_NEAR'] - Centroids['Water_NEAR']).sum() +
    #                 (Centroids_new['power_NEAR'] - Centroids['power_NEAR']).sum()
    #         )
    #         Centroids = Centroids_new.copy()
    #
    # # Plot final clusters
    # color_map = {1: 'r', 2: 'g', 3: 'b'}  # Color map for clusters
    # plt.scatter(X['power_NEAR'], X['Water_NEAR'], c=cluster_label.map(color_map), alpha=0.5)
    # plt.scatter(Centroids['power_NEAR'], Centroids['Water_NEAR'], marker='o', s=200, c='black')
    # plt.xlabel('power_NEAR')
    # plt.ylabel('Water_NEAR (In Thousands)')
    # plt.title('Final Clusters')
    # plt.show()
