/*
Author: Aiden Cary
Professor: Dr. Emre Celebi
CSCI 4372 Data Clustering
Phase 4
Date: 1 April 2026

Programming Practices: https://google.github.io/styleguide/cppguide.html

How to Compile: Use a C++17 compatible compiler
Navigate to the directory containing main.cpp (src) and run the following command:
Compile (using g++): g++ main.cpp ../src/Kmeans.cpp ../src/Point.cpp -o main.exe -std=c++17 -I../include -O2
Run: ./main.exe <filename> <I> <T> <R>
Phase 4 Examples:
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
*/
#include <iostream>
#include <string>

#include "../include/Kmeans.h"
#include "../include/Point.h"

// Print expected parameters message
void printExpectedParameters(const char* program_name)
{
    // K is no longer a parameter 
    // K is swept automatically from Kmin=2 to Kmax=round(sqrt(N/2))
    std::cerr
        << "  Only 4 parameters expected: " << program_name << " <F> <I> <T> <R>\n"
        << "  <F>: data file name\n"
        << "  <I>: maximum number of iterations (positive int)\n"
        << "  <T>: convergence threshold (non-negative real less than 1.0)\n"
        << "  <R>: number of runs (positive int)\n";
}

// Set parameters from command line arguments
// K is no longer supplied by the user
bool setParameters(
    int argc,
    char* argv[],
    std::string& data_file_name,
    int& max_iterations,
    double& convergence_threshold,
    int& num_runs)
{
    try
    {
        // Parse each positional argument from the command line
        data_file_name = argv[1];
        max_iterations = std::stoi(argv[2]);
        convergence_threshold = std::stod(argv[3]);
        num_runs = std::stoi(argv[4]);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error parsing arguments: " << ex.what() << std::endl;
        return false;
    }
    /*
    <F>: Upper Limit = N/A, Lower Limit = file must exist and be readable (handled in Kmeans::readData())
    <I>: Upper Limit = N/A, Lower Limit = 1 (must be positive)
    <T>: Upper Limit = N/A, Lower Limit = 0.0 (must be non-negative) and less than 1.0 (must be less than 1.0 to make sense as a relative improvement threshold)
    <R>: Upper Limit = N/A, Lower Limit = 1 (must be positive)
    */
    if (max_iterations <= 0 || convergence_threshold < 0.0 || convergence_threshold >= 1.0 || num_runs <= 0)
    {
        std::cerr << "Invalid values: require I>0, T>=0 and T<1.0, R>0." << std::endl;
        return false;
    }

    return true;
}

// Print the parameters to standard output
void printParameters(
    const std::string& data_file_name,
    int max_iterations,
    double convergence_threshold,
    int num_runs)
{
    // Display all parameters so the user can verify the input
    std::cout << "Data File Name <F>: " << data_file_name << std::endl;
    std::cout << "Maximum Iterations <I>: " << max_iterations << std::endl;
    std::cout << "Convergence Threshold <T>: " << convergence_threshold << std::endl;
    std::cout << "Number of Runs <R>: " << num_runs << std::endl;
}



/*
Arguments:
argc - Argument count
argv - Argument vector for four command line arguments (not hard-coded):
<F>: name of the data file
	- First line of F contains the number of Points N (positive integer) and the dimensionality of each point (D)
    - Each of the subsequent lines contains one data point in blank separated format
<I>: maximum number of iterations (positive integer)
<T>: convergence threshold (non-negative real)
<R>: number of runs (positive integer)
Note: K is no longer a user-supplied argument. The program sweeps K from Kmin=2 to Kmax=round(sqrt(N/2)).
*/
int main(int argc, char* argv[])
{
    // Verify the correct number of arguments were passed
    if (argc != 5)
    {
        printExpectedParameters(argv[0]);
        return 1;
    }

    std::string data_file_name;
    int max_iterations;
    double convergence_threshold;
    int num_runs;

    // Parse and validate command line arguments
    if (!setParameters(
            argc,
            argv,
            data_file_name,
            max_iterations,
            convergence_threshold,
            num_runs))
    {
        return 1;
    }

    // Print the parameters so the user can verify them
    printParameters(data_file_name, max_iterations, convergence_threshold, num_runs);

    // Construct Kmeans with a placeholder K=2; the sweep in runInternalValidation
    // will override num_clusters_ for each K value in the range [Kmin, Kmax]
    Kmeans k_means(
        data_file_name,
        2,
        max_iterations,
        convergence_threshold,
        num_runs);

    // Read the dataset from file
    if (!k_means.readData())
    {
        return 1;
    }

    // Phase 4: Normalize data using min-max method, then run internal validation sweep
    k_means.minmaxNormalize();

    // Run the internal validation sweep over K = Kmin..Kmax,
    // computing CH and SW for each K and reporting estimated cluster counts
    k_means.runInternalValidation();

    return 0;
}