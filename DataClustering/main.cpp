/*
Author: Aiden Cary
Professor: Dr. Emre Celebi
CSC1 4372 Data Clustering
Date: 18 January 2026
*/
#include <iostream>
#include <string>

void testPrintArgs(int argc, char* argv[])
{
	// Print the number of arguments and each argument on a new line (excluding the program name)
    std::cout << "Number of arguments: " << (argc - 1) << std::endl;
    for (int i = 1; i < argc; ++i)
    {
        std::cout << "Argument <" << i << ">: " << argv[i] << std::endl;
    }
}

/*
Arguments:
arhc - Argument count
argv - Argument vector for five command line arguments:
<F>: name of the data file
<K>: number of clusters (positive integer > 1)
<I>: maximum number of iterations (positive integer)
<T>: convergence threshold (non-negative real)
<R>: number of runs (positive integer)
*/

int main(int argc, char* argv[])
{
    // testPrintArgs(argc, argv);
    
    std::string dataFileName = argv[1];
	int numClusters = std::stoi(argv[2]);
	int maxIterations = std::stoi(argv[3]);
	double convergenceThreshold = std::stod(argv[4]);
    int numRuns = std::stoi(argv[5]);
    std::cout << "Data File Name <F>: " << dataFileName << std::endl;
    std::cout << "Number of Clusters <K>: " << numClusters << std::endl;
    std::cout << "Maximum Iterations <I>: " << maxIterations << std::endl;
    std::cout << "Convergence Threshold <T>: " << convergenceThreshold << std::endl;
    std::cout << "Number of Runs <R>: " << numRuns << std::endl;
	return 0;

}

