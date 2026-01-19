/*
Author: Aiden Cary
Professor: Dr. Emre Celebi
CSC1 4372 Data Clustering
Date: 18 January 2026

*/
#include <iostream>
#include <string>

bool setParameters(
    int argc,
    char* argv[],
    std::string& dataFileName,
    int& numClusters,
    int& maxIterations,
    double& convergenceThreshold,
    int& numRuns)
{
    // Expect: program + 5 args
    if (argc != 6)
    {
        std::cerr
            << "Usage: " << argv[0] << " <F> <K> <I> <T> <R>\n"
            << "  <F>: data file name\n"
            << "  <K>: number of clusters (> 1)\n"
            << "  <I>: maximum number of iterations (positive int)\n"
            << "  <T>: convergence threshold (non-negative real)\n"
            << "  <R>: number of runs (positive int)\n";
        return false;
    }

	// Parse arguments
    try
    {
        dataFileName = argv[1];
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
    // testPrintArgs(argc, argv);

	// Declare parameters
    std::string dataFileName;
    int numClusters;
    int maxIterations;
    double convergenceThreshold;
    int numRuns;

	// Set parameters from command line arguments
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
    std::cout << "Data File Name <F>: " << dataFileName << std::endl;
    std::cout << "Number of Clusters <K>: " << numClusters << std::endl;
    std::cout << "Maximum Iterations <I>: " << maxIterations << std::endl;
    std::cout << "Convergence Threshold <T>: " << convergenceThreshold << std::endl;
    std::cout << "Number of Runs <R>: " << numRuns << std::endl;

    return 0;
}

