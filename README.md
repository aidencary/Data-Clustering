# Data Clustering - K-Means Implementation

A C++ implementation of the K-Means clustering algorithm.

**Course:** CSCI 4372 - Data Clustering, Dr. Emre Celebi
**Author:** Aiden Cary

---

## Project Structure

```
DataClustering/
├── datasets/          # Input data files (.txt)
├── output_sheets/     # CSV/Excel output from runs
├── include/
│   ├── Kmeans.h
│   └── Point.h
└── src/
    ├── main.cpp
    ├── Kmeans.cpp
    └── Point.cpp
```

## Requirements

- C++17 compatible compiler (g++, clang, MSVC)

## Compilation

Navigate to the `src/` directory, then run:

```bash
g++ main.cpp ../src/Kmeans.cpp ../src/Point.cpp -o main.exe -std=c++17 -I../include -O2
```

## Usage

```bash
./main.exe <F> <K> <I> <T> <R>
```

| Parameter | Description |
|-----------|-------------|
| `F` | Dataset filename (e.g., `iris_bezdek.txt`) |
| `K` | Number of clusters (integer > 1) |
| `I` | Maximum iterations (positive integer) |
| `T` | Convergence threshold (>= 0.0 and < 1.0) |
| `R` | Number of runs (positive integer) |

The program searches for `F` in the current directory first, then in `../datasets/`.

### Example

```bash
./main.exe iris_bezdek.txt 3 100 0.0001 100
```

### Phase 3 Dataset Examples

```bash
./main.exe ecoli.txt 8 100 0.0001 100
./main.exe glass.txt 6 100 0.0001 100
./main.exe ionosphere.txt 2 100 0.0001 100
./main.exe iris_bezdek.txt 3 100 0.0001 100
./main.exe landsat.txt 6 100 0.0001 100
./main.exe letter_recognition.txt 26 100 0.0001 100
./main.exe segmentation.txt 7 100 0.0001 100
./main.exe vehicle.txt 4 100 0.0001 100
./main.exe wine.txt 3 100 0.0001 100
./main.exe yeast.txt 10 100 0.0001 100
```

## Dataset Format

The first line of each file contains `<N> <D>` (number of points and dimensionality). Each subsequent line contains one data point with space-separated values.

## Features

- Min-Max normalization (scales all attributes to [0, 1])
- Two centroid initialization methods: Random Selection and Random Partition
- K-Means with SSE (Sum of Squared Errors) tracking
- Multiple independent runs with best-result selection
- CSV output of metrics to `output_sheets/`
