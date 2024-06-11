import geopandas as gpd
import matplotlib.pyplot as plt


class MapPlotter:
    def __init__(self, map_shapefile):
        self.map_shapefile = map_shapefile

    def plot_map(self):
        self.gdf = gpd.read_file(self.map_shapefile)
        self.gdf.plot()
        plt.show()

    def plot_clustered_villages(self, villages_gdf, cluster_col):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        base = self.gdf.plot(ax=ax, color='white', edgecolor='black')

        # Генерация цветов для кластеров
        num_clusters = villages_gdf[cluster_col].nunique()
        colors = plt.cm.get_cmap('tab20', num_clusters)  # Использование цветовой карты

        for cluster in range(num_clusters):
            cluster_villages = villages_gdf[villages_gdf[cluster_col] == cluster]
            cluster_villages.plot(ax=base, marker='o', color=colors(cluster), markersize=50, label=f'Cluster {cluster}')

        plt.legend()
        plt.show()
