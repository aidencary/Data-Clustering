/*
Author: Aiden Cary
Professor: Dr. Emre Celebi
CSCI 4372 Data Clustering
Phase 1
Date: 21 January 2026

Programming Practices: https://google.github.io/styleguide/cppguide.html

How to Compile: Use a C++17 compatible compiler
Navigate to the directory containing main.cpp (src) and run the following command:
Compile (using g++): g++ main.cpp ../src/Kmeans.cpp ../src/Point.cpp -o main.exe -std=c++17 -I../include
Run: ./main.exe <filename> <K> <I> <T> <R>
Examples: 
>> ./main.exe iris_bezdek.txt 3 100 0.0001 100
>> ./main.exe glass.txt 6 100 0.000001 100
>> ./main.exe ionosphere.txt 2 100 0.000001 100
>> ./main.exe iris_bezdek.txt 3 100 0.000001 100
>> ./main.exe landsat.txt 6 100 0.000001 100
>> ./main.exe letter_recognition.txt 26 100 0.000001 100
>> ./main.exe segmentation.txt 7 100 0.000001 100
>> ./main.exe vehicle.txt 4 100 0.000001 100
>> ./main.exe wine.txt 3 100 0.000001 100
>> ./main.exe yeast.txt 10 100 0.000001 100
*/
#include <iostream>
#include <string>

#include "../include/Kmeans.h"
#include "../include/Point.h"

// Print expected parameters message
void printExpectedParameters(const char* program_name)
{
    std::cerr
        << "  Only 5 parameters expected: " << program_name << " <F> <K> <I> <T> <R>\n"
        << "  <F>: data file name\n"
        << "  <K>: number of clusters (> 1)\n"
        << "  <I>: maximum number of iterations (positive int)\n"
        << "  <T>: convergence threshold (non-negative real)\n"
        << "  <R>: number of runs (positive int)\n";
}

// Set parameters from command line arguments
bool setParameters(
    int argc,
    char* argv[],
    std::string& data_file_name,
    int& num_clusters,
    int& max_iterations,
    double& convergence_threshold,
    int& num_runs)
{
    try
    {
        data_file_name = argv[1];
        num_clusters = std::stoi(argv[2]);
        max_iterations = std::stoi(argv[3]);
        convergence_threshold = std::stod(argv[4]);
        num_runs = std::stoi(argv[5]);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error parsing arguments: " << ex.what() << std::endl;
        return false;
    }

    if (num_clusters <= 1 || max_iterations <= 0 || convergence_threshold < 0.0 || num_runs <= 0)
    {
        std::cerr << "Invalid values: require K>1, I>0, T>=0, R>0." << std::endl;
        return false;
    }

    return true;
}

// Print the parameters to standard output
void printParameters(
    const std::string& data_file_name,
    int num_clusters,
    int max_iterations,
    double convergence_threshold,
    int num_runs)
{
    std::cout << "Data File Name <F>: " << data_file_name << std::endl;
    std::cout << "Number of Clusters <K>: " << num_clusters << std::endl;
    std::cout << "Maximum Iterations <I>: " << max_iterations << std::endl;
    std::cout << "Convergence Threshold <T>: " << convergence_threshold << std::endl;
    std::cout << "Number of Runs <R>: " << num_runs << std::endl;
}



/*
Arguments:
argc - Argument count
argv - Argument vector for five command line arguments (not hard-coded):
<F>: name of the data file
	- First line of F contains the number of Points N (positive integer) and the dimensionality of each point (D)
    - Each of the subsequent lines contains one data point in blank separated format
<K>: number of clusters (positive integer > 1)
<I>: maximum number of iterations (positive integer)
<T>: convergence threshold (non-negative real)
<R>: number of runs (positive integer)
*/
int main(int argc, char* argv[])
{
	if (argc != 6)
    {
        printExpectedParameters(argv[0]);
        return 1;
    }

    std::string data_file_name;
    int num_clusters;
    int max_iterations;
    double convergence_threshold;
    int num_runs;

    if (!setParameters(
            argc,
            argv,
            data_file_name,
            num_clusters,
            max_iterations,
            convergence_threshold,
            num_runs))
    {
        return 1;
    }

    Kmeans kmeans(
        data_file_name,
        num_clusters,
        max_iterations,
        convergence_threshold,
		num_runs);
    
    if (!kmeans.readData())
    {
        return 1;
	}
    
    // Phase 1: Select and print K random centers
	// kmeans.selectAndPrintCenters();

    // Phase 2: Implement K-means algorithm
    kmeans.KmeansAlgorithm();

    return 0;
}