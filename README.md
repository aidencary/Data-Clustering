# Data Clustering - K-Means (Phase 4)

C++17 implementation of K-means with internal validation for model order selection.

Course: CSCI 4372 Data Clustering (Dr. Emre Celebi)  
Author: Aiden Cary

## What This Phase Does

Phase 4 runs K-means over a range of cluster counts and evaluates each result using:

- Calinski-Harabasz index (CH)
- Silhouette Width (SW)

Instead of passing K manually, the program sweeps:

- Kmin = 2
- Kmax = round(sqrt(N / 2))

where N is the number of data points in the selected dataset.

## Project Structure

```text
datasets/                 Input datasets (.txt)
docs/                     Project docs
include/
    Kmeans.h
    Point.h
src/
    main.cpp
    Kmeans.cpp
    Point.cpp
    create_iris_mod.cpp
output_phase4/            Phase 4 CSV outputs (one file per dataset)
output_sheets/            Earlier phase result sheets
```

## Build

Use any C++17 compiler.

From src:

```bash
g++ main.cpp ../src/Kmeans.cpp ../src/Point.cpp -o main.exe -std=c++17 -I../include -O2
```

On Windows PowerShell (from repo root):

```powershell
g++ .\src\main.cpp .\src\Kmeans.cpp .\src\Point.cpp -o .\src\main.exe -std=c++17 -I.\include -O2
```

## Usage

```bash
./main.exe <F> <I> <T> <R>
```

Parameter details:

- F: dataset filename (example: iris_bezdek.txt)
- I: maximum iterations, integer > 0
- T: convergence threshold, 0 <= T < 1
- R: number of runs per K value, integer > 0

Important:

- K is not a command-line parameter in Phase 4.
- The program first attempts to open F directly, then tries ../datasets/F.

## Example Commands

```bash
./main.exe iris_bezdek.txt 100 0.0001 100
./main.exe wine.txt 100 0.0001 100
./main.exe yeast.txt 100 0.0001 100
```

## Input Dataset Format

Each dataset must follow this format:

1. First line: N D
2. Next N lines: D space-separated numeric values per point

## Output (Phase 4)

For dataset name <name>.txt, results are written to:

- output_phase4/results_<name>.csv

Each CSV contains:

- Per-K rows: K, CH, SW
- Summary rows: Best K (CH), Best K (SW)

The console output also prints:

- CH and SW for each tested K
- Final table with both indices
- Estimated K from CH and estimated K from SW

## Implementation Notes

- Min-max normalization is applied per attribute (column-wise) to [0, 1].
- K-means uses random partition initialization during the Phase 4 validation sweep.
- For each K, the best run is selected by lowest final SSE before computing CH and SW.

## Typical Dataset Runs

```bash
./main.exe ecoli.txt 100 0.0001 100
./main.exe glass.txt 100 0.0001 100
./main.exe ionosphere.txt 100 0.0001 100
./main.exe iris_bezdek.txt 100 0.0001 100
./main.exe landsat.txt 100 0.0001 100
./main.exe letter_recognition.txt 100 0.0001 100
./main.exe segmentation.txt 100 0.0001 100
./main.exe vehicle.txt 100 0.0001 100
./main.exe wine.txt 100 0.0001 100
./main.exe yeast.txt 100 0.0001 100
```
