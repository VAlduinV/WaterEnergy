# main.py
if __name__ == '__main__':
    from KMethod_cluster.k_method_clusterisation import (load_data, plot_elbow_curve, calculate_silhouette_coefficients,
                                                         plot_silhouette_coefficients, perform_clustering, create_pipelines,
                                                         fit_pipeline, plot_explained_variance,
                                                         display_cluster_scatter_plot, plot_3d_clusters,
                                                         evaluate_clusters_and_plot, apply_pca, display_cluster_umap, perform_clustering_ing)
    from fuzzy_clusterisation.fuzzy_c_means import (preprocess_data, perform_fcm_clustering, plot_clusters_fuzzy,
                                                    perform_multiple_fcm_clusterings, plot_multiple_clusters, plot_pairplot,
                                                    plot_data_distribution)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from prettytable import PrettyTable
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Configure logging


    def display_village_clusters(data, labels):
        """
        Display the village names and their corresponding cluster labels in a table.

        Args:
            data (pd.DataFrame): The data containing the village names.
            labels (np.ndarray): Array of cluster labels corresponding to each row in data.
        """
        table = PrettyTable()  # Initialize PrettyTable
        table.field_names = ["Village Name", "Cluster"]  # Set table field names
        for name, label in zip(data['admin4Na_1'].head(20), labels[:20]):  # Iterate over first 20 entries
            table.add_row([name, label])  # Add row to table
        print(table)  # Print table


    def main():
        """
        Main function to load data, perform clustering, and visualize results.
        """
        file_path = r'.\data\VILLAGE.xlsx'
        selected_columns = ['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR', 'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_', 'Water_NEAR', 'occup_NEAR', 'power_NEAR']
        df, selected_data = load_data(file_path, selected_columns)  # Load data
        df_df, selected_data_df = load_data(file_path, selected_columns + ['admin4Na_1'])  # Load data with village names

        kmeans = KMeans(n_clusters=9, random_state=42)  # Initialize KMeans
        cluster_labels = kmeans.fit_predict(selected_data_df.drop(['admin4Na_1'], axis=1))  # Predict clusters
        display_village_clusters(selected_data_df, cluster_labels)  # Display village clusters

        logging.info(f'Obtained data:\n {df.head()}')  # Log data
        logging.info(f'Retrieved data from selected columns:\n {df[selected_columns].head()}')  # Log selected data

        n_clusters = 9
        scaled_features = df[selected_columns]  # Select features
        kmeans_kwargs = {"init": "k-means++", "n_init": 100, "max_iter": 1000, "random_state": 42}  # Set KMeans parameters
        scaler = StandardScaler()  # Initialize StandardScaler
        scaled_features_rate = scaler.fit_transform(scaled_features)  # Scale features

        plot_elbow_curve(scaled_features, kmeans_kwargs, n_clusters, scaled_features_rate)  # Plot elbow curve
        silhouette_coefficients, optimal_k = calculate_silhouette_coefficients(scaled_features, kmeans_kwargs, n_clusters)  # Calculate silhouette coefficients
        plot_silhouette_coefficients(silhouette_coefficients, optimal_k, n_clusters)  # Plot silhouette coefficients

        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
        markers = ['o', 's', '^', 'p', '*', 'd', 'v', '<', '>']
        perform_clustering(range_n_clusters, selected_data, markers)  # Perform clustering
        perform_clustering_ing([3, 4, 6, 9], selected_data)  # Perform clustering for specific clusters

        preprocessor, clusterer, pipe = create_pipelines(n_clusters=n_clusters)  # Create pipelines
        clustered_data, centroids = fit_pipeline(pipe, selected_data)  # Fit pipeline
        display_cluster_scatter_plot(clustered_data, centroids)  # Display cluster scatter plot
        plot_explained_variance(selected_data)  # Plot explained variance

        plot_3d_clusters(selected_data.values, n_clusters, selected_columns)  # Plot 3D clusters
        evaluate_clusters_and_plot(selected_data, n_clusters)  # Evaluate clusters and plot

        X = preprocess_data(selected_data)  # Preprocess data
        fcm, labels = perform_fcm_clustering(X, n_clusters)  # Perform FCM clustering
        plot_clusters_fuzzy(X, labels, fcm.centers)  # Plot fuzzy clusters

        models = perform_multiple_fcm_clusterings(X, range_n_clusters)  # Perform multiple FCM clusterings
        plot_multiple_clusters(X, models, range_n_clusters)  # Plot multiple clusters

        plot_data_distribution(selected_data)  # The results display that there are no severe extreme values, or outliers.

        results = selected_data.copy()  # Copy selected data
        results['Labels'] = labels  # Add labels to results
        plot_pairplot(results)  # Plot pairplot


    main()
