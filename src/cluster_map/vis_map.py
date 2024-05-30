# WaterEnergy/cluster_map/vis_map.py
import geopandas as gpd
import matplotlib.pyplot as plt
import mplcyberpunk

plt.style.use("cyberpunk")


class MapPlotter:
    """
    Class for plotting maps using GeoPandas and Matplotlib.
    """

    def __init__(self, file_path: str):
        """
        Initializes the MapPlotter with the file path to the shapefile.

        Args:
            file_path (str): Path to the shapefile.
        """
        self.file_path = file_path

    def plot_map(
        self,
        facecolor: str = "red",
        edgecolor: str = "black",
        figsize: tuple = (19.2, 16.8),
    ):
        """
        Plots the map with the given facecolor, edgecolor, and figure size.

        Args:
            facecolor (str): Color of the map's face.
            edgecolor (str): Color of the map's edges.
            figsize (tuple): Size of the figure.
        """
        try:
            ukraine = gpd.read_file(self.file_path)  # Read shapefile
            ukraine.plot(
                facecolor=facecolor, edgecolor=edgecolor, figsize=figsize
            )  # Plot map
            plt.show()  # Show plot
        except Exception as e:
            print(f"An error occurred: {e}")  # Print error message


map_plotter = MapPlotter(file_path="../data/map_data/gadm41_UKR_1.shp")
map_plotter.plot_map()  # Plot map
