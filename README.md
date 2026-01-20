# Data Clustering - K-Means Implementation

A C++ implementation of the K-Means clustering algorithm for unsupervised machine learning and data analysis.

## Overview

## Project Structure

```
Data-Clustering/
├── DataClustering/
│   ├── CMakeLists.txt
│   ├── DataClustering.vcxproj
│   ├── datasets/                 # Input data files
│   │   ├── ecoli.txt
│   │   ├── glass.txt
│   │   ├── ionosphere.txt
│   │   ├── iris_bezdek.txt
│   │   ├── landsat.txt
│   │   ├── letter_recognition.txt
│   │   ├── segmentation.txt
│   │   ├── vehicle.txt
│   │   ├── wine.txt
│   │   └── yeast.txt
│   ├── include/                  # Header files
│   │   ├── Kmeans.h
│   │   └── Point.h
│   └── src/                      # Source files
│       ├── main.cpp
│       ├── Kmeans.cpp
│       └── Point.cpp
└── README.md
```

## Class Methods

### `Point` Class
Represents a single data point with multiple dimensions.

- **`Point()`** - Constructor; initializes an empty point
- **`void addDimension(double val)`** - Adds a dimension value to the point
- **`double getVal(int index) const`** - Returns the value at a specific dimension
- **`void print() const`** - Prints the point to console (space-separated values)

### `Kmeans` Class
Manages the K-Means clustering algorithm.

- **`Kmeans(const std::string& fileName, int numClusters, int maxIterations, double convergenceThreshold, int numOfRuns)`**
  - Constructor that initializes the K-Means object with configuration parameters
  - `fileName`: Path to the input dataset file
  - `numClusters`: Number of clusters (K > 1)
  - `maxIterations`: Maximum number of iterations
  - `convergenceThreshold`: Threshold for convergence (when to stop iterating)
  - `numOfRuns`: Number of independent runs to perform

- **`bool readData()`** - Reads dataset from file
  - Attempts to open file from the provided path
  - If not found, tries prepending `datasets/` to the path
  - File format: First line contains `<number_of_points> <dimensionality>`, followed by data points
  - Returns `true` on success, `false` on failure

- **`void printData() const`** - Prints all data points in the dataset to console

- **`bool contains(const std::vector<T>& vec, const T& value)`** - Template utility function
  - Returns `true` if value exists in vector, `false` otherwise
  - Used for checking uniqueness during center selection

- **`void selectAndPrintCenters()`** - Selects and prints initial cluster centers
  - Randomly selects K unique data points as initial centers
  - Prints selected centers to console

## Compilation (may use Cmake later for better portability)

### Using g++ (GCC Compiler)
```bash
g++ main.cpp Kmeans.cpp Point.cpp -o main.exe -std=c++17 -I./include
```

### Using CMake (may implement later)
```bash
```

## Running the Program

```bash
./main.exe <filename> <K> <I> <T> <R>
```

### Parameters
- **`<filename>`**: Name of the input dataset file (just the filename, e.g., `iris_bezdek.txt`)
  - The program will first look for the file in the current directory
  - If not found, it will automatically look in the `../datasets/` folder
- **`<K>`**: Number of clusters (positive integer > 1)
- **`<I>`**: Maximum number of iterations (positive integer)
- **`<T>`**: Convergence threshold (non-negative real number)
- **`<R>`**: Number of runs (positive integer)

### Example Usage
```bash
./main.exe iris_bezdek.txt 3 100 0.000001 100
```
This runs K-Means with 3 clusters, up to 100 iterations, convergence threshold of 0.000001, and 100 runs.

The program will automatically search for `iris_bezdek.txt` in the current directory first, then in `../datasets/` if not found.

## File Path Flexibility

The program intelligently handles file paths:
- If you provide just the filename (e.g., `iris_bezdek.txt`), it will try to find it in the `datasets/` folder
- You can also provide a full relative path (e.g., `datasets/iris_bezdek.txt`)
- The program will report an error if the file is not found in either location

## Requirements

- C++17 compatible compiler (g++, clang, MSVC)
- Standard C++ libraries: `<iostream>`, `<fstream>`, `<vector>`, `<string>`, `<random>`, `<ctime>`

## Notes

Currently implemented features:
- Data file reading with error handling
- Random cluster center initialization
- File path resolution (with fallback to `datasets/` folder)

Future implementation needed:
- Point-to-center distance calculation
- Assignment of points to nearest centers
- Center recalculation and convergence checking
- Multiple runs and result aggregation
- Output file generation

## Author
Aiden Cary

## Course
CSC 4372 - Data Clustering
Professor: Dr. Emre Celebi