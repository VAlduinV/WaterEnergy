import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import mplcyberpunk

plt.style.use("cyberpunk")


def kmeans_clustering(df, features, target, range_n_clusters):
    # Вибір всіх параметрів
    X = df[features]
    Y = df[target]

    # Підготовка списку для збереження значень коефіцієнта силуету
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(19.2, 10.8)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.axvline(x=silhouette_avg, color="red", lw=2, linestyle="--")
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # Customize legend labels to display feature names corresponding to each cluster label with colors
        legend_labels = []
        legend_handles = []  # to store handles for legend entries
        for i in range(n_clusters):
            if n_clusters == 2:
                color = cm.nipy_spectral(float(i) / n_clusters)
                legend_labels.append(f'Cluster {i}: RD_m1_NEAR' if i == 0 else f'Cluster {i}: RD_m2_NEAR')
            else:
                color = cm.nipy_spectral(float(i) / n_clusters)
                legend_labels.append(f'Cluster {i}: {X.columns[i]}')

            # Add a patch with the color to the legend
            legend_handles.append(ax2.scatter([], [], marker="o", s=80, lw=2, alpha=0.7, color=color))
            # dummy scatter plot

            # Add a patch with the color to the legend
            # ax2.scatter([], [], marker="o", s=80, lw=2, alpha=0.7, c=color, label=legend_labels[i])

        # Update scatter plot based on the number of clusters
        for i in range(1, n_clusters + 1):
            if n_clusters == 2:
                ax2.scatter(X[f'RD_m{i}_NEAR'], Y['OBJECTID'], marker="v", s=80, lw=2, alpha=0.7, c=colors,
                            edgecolor="k")
                ax2.set_xlabel("Feature space for RD_m1_NEAR and RD_m2_NEAR")
            else:
                ax2.scatter(X[X.columns[i - 1]], Y['OBJECTID'], marker="v", s=80, lw=2, alpha=0.7, c=colors,
                            edgecolor="k")
                ax2.set_xlabel(f"Feature space for {X.columns[i - 1]}")
            ax2.set_ylabel("OBJECTID")

        # centers = kmeans.cluster_centers_
        # for j, c in enumerate(centers):
        #     ax2.scatter(c[i - 1], c[0], marker=f"${j}$", alpha=1, s=80, edgecolor="red")

        ax2.set_title("The visualization of the clustered data.")
        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        legend = ax2.legend(legend_handles, legend_labels, loc='best',
                            labelcolor="black", fontsize=14,
                            frameon=True,
                            scatterpoints=1)
        legend.set_title('Clusters', prop={'size': 14, 'weight': 'bold', 'family': 'Times New Roman'})
        # Customize the legend box
        frame = legend.get_frame()
        frame.set_facecolor('lightgrey')  # background color
        frame.set_edgecolor('red')  # border color
        plt.setp(legend.get_title(), color='black')
        plt.show()

    k = np.argmax(silhouette_scores) + 2
    print('The optimal number of cluster for a given data is {}'.format(k))
    fig, ax3 = plt.subplots(1, figsize=(16, 5))
    ax3.plot([n_cluster for n_cluster in range_n_clusters], silhouette_scores, linewidth=3,
             markersize=8, color='RED', label='silhouette curve')
    ax3.grid(which='major', color='BLACK', linestyle='-')
    ax3.axvline(x=k, linewidth=2, color='GREEN', linestyle='--',
                label='Optimal number of clusters ({})'.format(k))
    legend_for_optcluster = ax3.legend(shadow=True, frameon=True, loc='best', fontsize='14', edgecolor='red',
                                       labelcolor='black')
    legend_for_optcluster.set_title('Optimal Cluster', prop={'size': 14, 'weight': 'bold', 'family': 'Times New Roman'})
    # Customize the legend box
    frame_optcluster = legend_for_optcluster.get_frame()
    frame_optcluster.set_facecolor('lightgrey')  # background color
    frame_optcluster.set_edgecolor('red')  # border color
    plt.setp(legend_for_optcluster.get_title(), color='black')
    plt.show()


def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
        print(f'means: {means}, \t inertias: {inertias}')

    # Generate the Elbow plot
    fig = plt.subplots(figsize=(12.7, 7.2))
    plt.plot(means, inertias, 'o-', color='red')
    plt.axvline(x=max_k - 1, linewidth=2, color='BLUE', linestyle='--',
                label='Optimal number of clusters ({})'.format(max_k - 1))
    mplcyberpunk.add_glow_effects(gradient_fill=True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


def visualize_clusters(df, features, n_clusters):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(19.2, 10.8), dpi=300)
    # dpi: роздільна здатність фігури в точках на дюйм.

    for i, ax in enumerate(axs.flatten(), start=1):
        if i <= n_clusters:
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(df[features])
            df[f'KMeans_{i}'] = kmeans.labels_
            ax.scatter(x=df[features[i - 1]], y=df['OBJECTID'], c=df[f'KMeans_{i}'], cmap='tab20')
            ax.set_title(f'n_clusters: {i}', fontsize=14)
            ax.set_xlabel(features[i - 1], fontsize=14)
            ax.set_ylabel('OBJECTID', fontsize=14)
        else:
            ax.axis('off')  # Отключаем оставшиеся графики, если n_clusters превышает количество доступных графиков
    plt.tight_layout()
    plt.show()

    for i in range(1, min(10, n_clusters + 1)):  # Ограничиваем вывод графиков до 9 для наглядности
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df[features])
        df[f'KMeans_{i}'] = kmeans.labels_

        fig, ax = plt.subplots(figsize=(19.2, 10.8))  # Создаем новый fig, ax для каждого графика
        ax.scatter(x=df[features[i - 1]], y=df['OBJECTID'], c=df[f'KMeans_{i}'], cmap='tab20')
        ax.set_title(f'n_clusters: {i}')
        ax.set_xlabel(features[i - 1])
        ax.set_ylabel('OBJECTID')
        plt.show()
