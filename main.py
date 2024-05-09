from sklearn.cluster import KMeans

if __name__ == '__main__':
    from k_method_clusterisation import (load_data, plot_elbow_curve, calculate_silhouette_coefficients,
                                         plot_silhouette_coefficients, perform_clustering, create_pipelines,
                                         fit_pipeline, plot_explained_variance,
                                         display_cluster_scatter_plot, plot_3d_clusters,
                                         evaluate_clusters_and_plot, evaluate_clusters_and_plot_snd_method,
                                         apply_pca, display_cluster_umap)
    from sklearn.preprocessing import StandardScaler
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def main():
        file_path = r'.\data\VILLAGE.xlsx'
        selected_columns = ['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR', 'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_',
                            'Water_NEAR', 'occup_NEAR', 'power_NEAR']
        df, selected_data = load_data(file_path, selected_columns)
        print(selected_data)

        # ------------------------------------------------------------- #
        logging.info(f'Obtained data:\n {df.head()}')
        logging.info(f'Retrieved data from selected columns:\n {df[selected_columns].head()}')
        # ------------------------------------------------------------- #
        n_clusters = 9
        # ------------------------------------------------------------- #
        scaled_features = df[selected_columns]
        kmeans_kwargs = {
            "init": "k-means++",
            "n_init": 100,
            "max_iter": 1000,
            "random_state": 42,
        }

        # This class implements a type of feature scaling called standardization
        scaler = StandardScaler()
        scaled_features_rate = scaler.fit_transform(scaled_features)

        plot_elbow_curve(scaled_features, kmeans_kwargs, n_clusters, scaled_features_rate)
        # ------------------------------------------------------------- #

        # ------------------------------------------------------------- #
        silhouette_coefficients, optimal_k = calculate_silhouette_coefficients(scaled_features, kmeans_kwargs,
                                                                               n_clusters)
        plot_silhouette_coefficients(silhouette_coefficients, optimal_k, n_clusters)
        # ------------------------------------------------------------- #

        # ------------------------------------------------------------- #

        # Range of number of clusters
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
        markers = ['o', 's', '^', 'p', '*', 'd', 'v', '<', '>']  # Different markers for different clusters
        perform_clustering(range_n_clusters, selected_data, markers)
        # ------------------------------------------------------------- #

        # ------------------------------------------------------------- #
        preprocessor, clusterer, pipe = create_pipelines(n_clusters=n_clusters)
        clustered_data, centroids = fit_pipeline(pipe, selected_data)
        display_cluster_scatter_plot(clustered_data, centroids)
        plot_explained_variance(selected_data)
        # ------------------------------------------------------------- #

        # Converting Data to a Numpy Array for KMeans: selected_data.values
        plot_3d_clusters(selected_data.values, n_clusters, selected_columns)
        evaluate_clusters_and_plot(selected_data, n_clusters)
        evaluate_clusters_and_plot_snd_method(selected_data, n_clusters)

        # PCA application
        n_components = 9
        principal_components = apply_pca(scaled_features, n_components)

        # Clustering (you might need to ensure you have a clustering function ready, here I use KMeans as an example)
        kmeans = KMeans(n_clusters=9, random_state=42)
        cluster_labels = kmeans.fit_predict(principal_components)

        # UMAP visualization
        display_cluster_umap(principal_components, cluster_labels)

    main()
