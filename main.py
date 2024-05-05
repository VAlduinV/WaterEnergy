import numpy as np

if __name__ == '__main__':
    from k_method_clusterisation import (load_data, plot_elbow_curve, calculate_silhouette_coefficients,
                                         plot_silhouette_coefficients, cluster_analysis)
    from sklearn.preprocessing import StandardScaler
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)


    def main():
        file_path = r'.\data\VILLAGE.xlsx'
        selected_columns = ['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR', 'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_',
                            'Water_NEAR', 'occup_NEAR', 'power_NEAR']
        df, selected_data = load_data(file_path, selected_columns)

        # ------------------------------------------------------------- #
        logging.info(f'Obtained data:\n {df.head()}')
        logging.info(f'Retrieved data from selected columns:\n {df[selected_columns].head()}')
        # ------------------------------------------------------------- #

        # ------------------------------------------------------------- #
        scaled_features = df[selected_columns]
        kmeans_kwargs = {
            "init": "k-means++",
            "n_init": 100,
            "max_iter": 1000,
            "random_state": 42,
        }
        n_clusters = 9

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
        # Generate random labels for the number of clusters
        num_samples = len(selected_data)  # Number of Samples
        # Generate random labels for each sample
        synthetic_labels = np.random.randint(0, n_clusters, size=num_samples)
        cluster_analysis(synthetic_labels, n_clusters, scaled_features_rate)
        # ------------------------------------------------------------- #


    main()
