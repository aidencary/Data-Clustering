/*
Author: Aiden Cary
Professor: Dr. Emre Celebi
CSCI 4372 Data Clustering
Phase 2 - Bonus: Data Generator for Modified Iris Bezdek Dataset
Date: 18 February 2026

This program creates iris_bezdek_mod.txt by reading iris_bezdek.txt and
duplicating every 5th data point 10 times (9 additional copies).
Result: 150 original + 30 * 9 duplicates = 420 total points

How to Compile: Use a C++17 compatible compiler
Compile (using g++): g++ create_iris_mod.cpp -o create_iris_mod.exe -std=c++17
Run: ./create_iris_mod.exe
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

int main() {
    // Open input file
    std::ifstream input_file("../datasets/iris_bezdek.txt");
    
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open iris_bezdek.txt" << std::endl;
        return 1;
    }

    // Read header: number of points and dimensionality
    int num_points, dimensionality;
    if (!(input_file >> num_points >> dimensionality)) {
        std::cerr << "Error: Could not read header from iris_bezdek.txt" << std::endl;
        return 1;
    }

    std::cout << "Reading iris_bezdek.txt..." << std::endl;
    std::cout << "Original points: " << num_points << std::endl;
    std::cout << "Dimensionality: " << dimensionality << std::endl;

    // Read all data points
    std::vector<std::vector<double>> dataset;
    
    for (int i = 0; i < num_points; ++i) {
        std::vector<double> point(dimensionality);
        for (int d = 0; d < dimensionality; ++d) {
            if (!(input_file >> point[d])) {
                std::cerr << "Error: Could not read data point " << i << std::endl;
                return 1;
            }
        }
        dataset.push_back(point);
    }
    
    input_file.close();
    std::cout << "Successfully read " << dataset.size() << " points." << std::endl;

    // Create modified dataset
    std::vector<std::vector<double>> modified_dataset;
    int duplicate_count = 0;

    for (int i = 0; i < num_points; ++i) {
        // Add the original point
        modified_dataset.push_back(dataset[i]);
        
        // Check if this is every 5th point (1-indexed: 5, 10, 15, 20, ...)
        // In 0-indexed: 4, 9, 14, 19, ... which is (i+1) % 5 == 0
        if ((i + 1) % 5 == 0) {
            // Duplicate this point 9 additional times (total of 10 copies including original)
            for (int copy = 0; copy < 9; ++copy) {
                modified_dataset.push_back(dataset[i]);
                duplicate_count++;
            }
        }
    }

    std::cout << "\nCreating modified dataset..." << std::endl;
    std::cout << "Points duplicated: " << (duplicate_count / 9) << " (every 5th point)" << std::endl;
    std::cout << "Total duplicates added: " << duplicate_count << std::endl;
    std::cout << "Total points in modified dataset: " << modified_dataset.size() << std::endl;

    // Verify the count
    int expected_points = num_points + (num_points / 5) * 9;
    if (modified_dataset.size() != (size_t)expected_points) {
        std::cerr << "Warning: Expected " << expected_points << " points but got " 
                  << modified_dataset.size() << std::endl;
    }

    // Write to output file
    std::ofstream output_file("../datasets/iris_bezdek_mod.txt");
    
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not create iris_bezdek_mod.txt" << std::endl;
        return 1;
    }

    // Write header
    output_file << modified_dataset.size() << " " << dimensionality << std::endl;

    // Write all points
    for (const auto& point : modified_dataset) {
        for (size_t d = 0; d < point.size(); ++d) {
            output_file << point[d];
            if (d < point.size() - 1) {
                output_file << " ";
            }
        }
        output_file << std::endl;
    }

    output_file.close();
    std::cout << "\nSuccessfully created iris_bezdek_mod.txt with " 
              << modified_dataset.size() << " points." << std::endl;

    return 0;
}
