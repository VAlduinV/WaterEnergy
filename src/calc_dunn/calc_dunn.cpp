#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <unordered_map>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

// Функция для поиска файла в указанной директории и поддиректориях и получения абсолютного пути
fs::path findFile(const fs::path& baseDir, const string& fileName) {
    cout << "Searching in directory: " << baseDir << endl;
    for (const auto& entry : fs::recursive_directory_iterator(baseDir)) {
        if (entry.path().filename() == fileName) {
            return fs::absolute(entry.path());
        }
    }
    return "";
}

// Функция для чтения данных из файла CSV и выбора нужных столбцов
vector<vector<double>> readCSVData(const string& filePath, const vector<string>& selectedColumns) {
    vector<vector<double>> data;
    ifstream file(filePath);
    unordered_map<string, int> columnIndices;
    vector<int> selectedIndices;

    if (!file.is_open()) {
        cerr << "Could not open the file: " << filePath << endl;
        return data;
    }

    string line;
    bool isHeader = true;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string value;

        if (isHeader) {
            int colIndex = 0;
            while (getline(ss, value, ',')) {
                columnIndices[value] = colIndex++;
            }
            for (const string& colName : selectedColumns) {
                if (columnIndices.find(colName) != columnIndices.end()) {
                    selectedIndices.push_back(columnIndices[colName]);
                } else {
                    cerr << "Column not found: " << colName << endl;
                    return {};
                }
            }
            isHeader = false;
        } else {
            int colIndex = 0;
            while (getline(ss, value, ',')) {
                if (find(selectedIndices.begin(), selectedIndices.end(), colIndex) != selectedIndices.end()) {
                    try {
                        row.push_back(stod(value));
                    } catch (const invalid_argument& e) {
                        cerr << "Invalid number in file: " << value << " in line: " << line << endl;
                        row.clear();
                        break;
                    } catch (const out_of_range& e) {
                        cerr << "Number out of range in file: " << value << " in line: " << line << endl;
                        row.clear();
                        break;
                    }
                }
                colIndex++;
            }
            if (!row.empty()) {
                data.push_back(row);
            }
        }
    }

    file.close();
    return data;
}

// Функция для вычисления расстояния между двумя точками
double calculateDistance(const vector<double>& point1, const vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

// Реализация алгоритма K-Means
vector<int> kMeans(const vector<vector<double>>& data, int k) {
    size_t numPoints = data.size();
    size_t numDimensions = data[0].size();

    vector<vector<double>> centroids(k, vector<double>(numDimensions));
    vector<int> labels(numPoints, -1);

    // Инициализация центроидов случайным образом
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[rand() % numPoints];
    }

    bool changed;
    do {
        changed = false;

        // Присваиваем метки кластеров
        for (size_t i = 0; i < numPoints; ++i) {
            double minDist = numeric_limits<double>::max();
            int bestCluster = -1;
            for (int j = 0; j < k; ++j) {
                double dist = calculateDistance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
        }

        // Обновляем центроиды
        vector<vector<double>> newCentroids(k, vector<double>(numDimensions, 0.0));
        vector<int> counts(k, 0);
        for (size_t i = 0; i < numPoints; ++i) {
            int cluster = labels[i];
            for (size_t j = 0; j < numDimensions; ++j) {
                newCentroids[cluster][j] += data[i][j];
            }
            counts[cluster]++;
        }
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (size_t d = 0; d < numDimensions; ++d) {
                    newCentroids[j][d] /= counts[j];
                }
            }
        }
        centroids = newCentroids;

    } while (changed);

    return labels;
}

// Функция для вычисления индекса Данна
double calculateDunnIndex(const vector<vector<double>>& data, const vector<int>& labels) {
    double minInterClusterDist = numeric_limits<double>::max();
    double maxIntraClusterDist = 0.0;

    map<int, vector<size_t>> clusters;
    for (size_t i = 0; i < labels.size(); ++i) {
        clusters[labels[i]].push_back(i);
    }

    for (const auto& cluster : clusters) {
        for (size_t i = 0; i < cluster.second.size(); ++i) {
            for (size_t j = i + 1; j < cluster.second.size(); ++j) {
                double dist = calculateDistance(data[cluster.second[i]], data[cluster.second[j]]);
                maxIntraClusterDist = max(maxIntraClusterDist, dist);
            }
        }
    }

    for (auto it1 = clusters.begin(); it1 != clusters.end(); ++it1) {
        for (auto it2 = next(it1); it2 != clusters.end(); ++it2) {
            for (size_t i : it1->second) {
                for (size_t j : it2->second) {
                    double dist = calculateDistance(data[i], data[j]);
                    minInterClusterDist = min(minInterClusterDist, dist);
                }
            }
        }
    }

    if (maxIntraClusterDist == 0) {
        return 0;
    }

    return minInterClusterDist / maxIntraClusterDist;
}

int main() {
    string fileName = "VILLAGE_ok_Dist2_coord-.csv";
    fs::path baseDir = fs::current_path().parent_path() / "data";  // Указание базовой директории для поиска
    fs::path filePath = findFile(baseDir, fileName);

    if (filePath.empty()) {
        cerr << "Could not find the file: " << fileName << " in " << baseDir << endl;
        return 1;
    }

    cout << "Found file at: " << filePath << endl;

    vector<string> selectedColumns = {
        "RD_m1_NEAR", "RD_m2_NEAR", "RD_m3_NEAR", "CITY2_NEAR", "Kiyv_NEAR_", "TOWN_NEAR_", "Water_NEAR", "occup_NEAR", "power_NEAR"
    };

    // Чтение данных из файла CSV с выбором нужных столбцов
    vector<vector<double>> data = readCSVData(filePath.string(), selectedColumns);

    if (data.empty()) {
        cerr << "No valid data found in the file." << endl;
        return 1;
    }

    int maxClusters = 4;
    vector<double> dunnScores;

    for (int k = 2; k <= maxClusters; ++k) {
        vector<int> labels = kMeans(data, k);
        double dunnIndex = calculateDunnIndex(data, labels);
        dunnScores.push_back(dunnIndex);
        cout << "Dunn Index for k=" << k << ": " << dunnIndex << endl;
    }

    return 0;
}
