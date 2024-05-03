import geopandas as gpd
import matplotlib.pyplot as plt
import mplcyberpunk

plt.style.use("cyberpunk")


class MapPlotter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def plot_map(self, facecolor: str = 'red', edgecolor: str = 'black', figsize: tuple = (16, 5)):
        try:
            ukraine = gpd.read_file(self.file_path)
            ukraine.plot(facecolor=facecolor, edgecolor=edgecolor, figsize=figsize)
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")
