import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


class MapPlotter:
    def __init__(self, map_shapefile):
        self.map_shapefile = map_shapefile
        self.gdf = gpd.read_file(map_shapefile)

    def plot_map(self):
        self.gdf.plot()
        plt.show()

    def plot_clustered_villages(self, villages_gdf, cluster_col, output_image, title):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        base = self.gdf.plot(ax=ax, color='white', edgecolor='black')

        # Генерация цветов для кластеров
        num_clusters = villages_gdf[cluster_col].nunique()
        colors = plt.cm.get_cmap('jet', num_clusters)  # Использование цветовой карты

        for cluster in range(1, num_clusters + 1):
            cluster_villages = villages_gdf[villages_gdf[cluster_col] == cluster]
            # Check if the cluster_villages GeoDataFrame is empty
            if not cluster_villages.empty:
                cluster_villages.plot(ax=base, marker='o', color=colors(cluster - 1), markersize=5,
                                      label=f'Cluster {cluster}')

        ax.set_aspect('equal')
        legend = plt.legend(title="Mapping clusters\n by settlements\n in Ukraine",
                            title_fontsize=14,
                            prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
                            labelcolor="black",
                            loc="best",
                            shadow=True,
                            frameon=True,
                            fancybox=True,
                            framealpha=0.8,
                            facecolor="white",
                            edgecolor="red",
                            )  # Add legend
        plt.setp(legend.get_title(), color="black")  # Set legend title color
        plt.title(title)
        plt.savefig(output_image)
        plt.show()

    def save_clusters_to_shapefile(self, villages_gdf, cluster_col, output_file):
        """
        Save the clustered villages to a shapefile.

        Args:
            villages_gdf (GeoDataFrame): The GeoDataFrame with village data.
            cluster_col (str): The column name of the cluster labels.
            output_file (str): The output shapefile path.
        """
        villages_gdf.to_file(output_file, driver='ESRI Shapefile')

    def create_village_gdf(self, df, lon_col, lat_col, cluster_col):
        """
        Create a GeoDataFrame for villages.

        Args:
            df (DataFrame): The DataFrame with village data.
            lon_col (str): The name of the longitude column.
            lat_col (str): The name of the latitude column.
            cluster_col (str): The name of the cluster column.

        Returns:
            GeoDataFrame: The created GeoDataFrame.
        """
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.crs = "EPSG:4326"  # WGS84 coordinate system
        return gdf

    def filter_regions(self, gdf, include_all=True, region_col='admin1Na_1'):
        if not include_all:
            exclude_regions = ['Запорізька', 'Донецька', 'Луганська', 'Херсонська', 'Автономна Республіка Крим']
            gdf = gdf[~gdf[region_col].isin(exclude_regions)]
        return gdf
