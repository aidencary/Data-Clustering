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
>> ./main.exe iris_bezdek.txt 3 100 0.000001 100
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

void printExpectedParameters(const char* programName)
{
    std::cerr
        << "  Only 5 parameters expected: " << programName << " <F> <K> <I> <T> <R>\n"
        << "  <F>: data file name\n"
        << "  <K>: number of clusters (> 1)\n"
        << "  <I>: maximum number of iterations (positive int)\n"
        << "  <T>: convergence threshold (non-negative real)\n"
        << "  <R>: number of runs (positive int)\n";
}

bool setParameters(
    int argc,
    char* argv[],
    std::string& dataFileName,
    int& numClusters,
    int& maxIterations,
    double& convergenceThreshold,
    int& numRuns)
{
    try
    {
        // Set parameters from command line arguments
        dataFileName = argv[1];
        // If stoi is given a floating point number, it will only read up to the decimal point (e.g., "3.5" becomes 3)
        // May change to stod and static_cast<int> if this is not desired
        numClusters = std::stoi(argv[2]);
        maxIterations = std::stoi(argv[3]);
        convergenceThreshold = std::stod(argv[4]);
        numRuns = std::stoi(argv[5]);
    }
	// Catch conversion errors when input is not valid
    catch (const std::exception& ex)
    {
        std::cerr << "Error parsing arguments: " << ex.what() << std::endl;
        return false;
    }

    // Input validation
    if (numClusters <= 1 || maxIterations <= 0 || convergenceThreshold < 0.0 || numRuns <= 0)
    {
        std::cerr << "Invalid values: require K>1, I>0, T>=0, R>0." << std::endl;
        return false;
    }

    return true;
}

void printParameters(
    const std::string& dataFileName,
    int numClusters,
    int maxIterations,
    double convergenceThreshold,
    int numRuns)
{
    std::cout << "Data File Name <F>: " << dataFileName << std::endl;
    std::cout << "Number of Clusters <K>: " << numClusters << std::endl;
    std::cout << "Maximum Iterations <I>: " << maxIterations << std::endl;
    std::cout << "Convergence Threshold <T>: " << convergenceThreshold << std::endl;
    std::cout << "Number of Runs <R>: " << numRuns << std::endl;
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
    // Check for correct number of arguments
	if (argc != 6)
    {
        printExpectedParameters(argv[0]);
    }

	// Declare parameters
    std::string dataFileName;
    int numClusters;
    int maxIterations;
    double convergenceThreshold;
    int numRuns;

	// Set parameters (from command line arguments)
    if (!setParameters(
            argc,
            argv,
            dataFileName,
            numClusters,
            maxIterations,
            convergenceThreshold,
            numRuns))
    {
        return 1;
    }

	// Print parameters to verify
    // printParameters(dataFileName, numClusters, maxIterations, convergenceThreshold, numRuns);

	// Create Kmeans object with the provided parameters
    Kmeans kmeans(
        dataFileName,
        numClusters,
        maxIterations,
        convergenceThreshold,
		numRuns);
    
	// Read data from the specified file
    if (!kmeans.readData())
    {
        return 1;
	}

    // Test if data was read correctly
    // kmeans.printData();
    
	// Select and print initial random cluster centers
	kmeans.selectAndPrintCenters();

    return 0;
}


