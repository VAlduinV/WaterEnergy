import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk

plt.style.use("cyberpunk")


def load_data(file_path, columns):
    """Load data from Excel file and select specified columns."""
    df = pd.read_excel(file_path)
    selected_data = df[columns]
    return df, selected_data


def calculate_statistics(data):
    """Calculate mean and standard deviation for each indicator."""
    # Обрахунок середнього значення та стандартного відхилення для кожного показника
    mean_values = data.mean()
    std_values = data.std()
    # Розрахунок верхньої межі для кожного показника за правилом 3-сигма
    up_values = mean_values + 3 * std_values
    return mean_values, std_values, up_values


# Функція для визначення градації якості життя для кожного села
def calculate_gradation(distance_to_city, up_values):
    if distance_to_city <= up_values['CITY2_NEAR'] / 3:
        return 0
    elif up_values['CITY2_NEAR'] / 3 < distance_to_city <= 2 * up_values['CITY2_NEAR'] / 3:
        return 1
    else:
        return 2


def plot_histogram(data, num_bins=70, color='red', edgecolor='black', figsize=(19.2, 10.8)):
    # Створення фігури та осі
    fig, ax = plt.subplots(figsize=figsize)
    # Побудова гістограми
    bars, bins, patches = ax.hist(data, bins=num_bins, color=color, edgecolor=edgecolor)
    ax.set_xlabel('Distance from village to city (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Distance from Village to City. City or Town')
    # Додавання ефекту градієнту до барів
    mplcyberpunk.add_bar_gradient(bars=patches)
    plt.show()
