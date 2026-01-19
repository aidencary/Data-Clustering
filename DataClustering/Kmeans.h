#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include "Point.h"

class Kmeans {
private:
	std::string fileName; // <F>
	int numClusters, maxIterations, numOfRuns; // <K>, <I>, <R>
	double convergenceThreshold; // <T>
	int numOfPoints, dimensionality;

	// Dataset of points
	std::vector<Point> dataset;

public:
	// Constructor
	Kmeans(
		const std::string& fileName,
		int numClusters,
		int maxIterations,
		double convergenceThreshold,
		int numOfRuns)
		: fileName(fileName),
		  numClusters(numClusters),
		  maxIterations(maxIterations),
		  convergenceThreshold(convergenceThreshold),
		  numOfRuns(numOfRuns),
		  numOfPoints(0),
		  dimensionality(0) {}

	// Read data from the file
	bool readData() {
		std::ifstream inputFile(fileName);
		if (!inputFile.is_open()) {
			std::cerr << "Error opening file: " << fileName << std::endl;
			return false;
		}

		// Read number of data points and dimensionality from the first line and if valid, store them
		if (!(inputFile >> numOfPoints >> dimensionality)) {
			std::cerr << "Error reading number of points and dimensionality." << std::endl;
			return false;
		}

		// Print number of points and dimensionality
		std::cout << "Number of Points: " << numOfPoints << ", Dimensionality: " << dimensionality << std::endl;

		// Read the number of data points
		for (int i = 0; i < numOfPoints; ++i) {
			Point point;
			for (int d = 0; d < dimensionality; ++d) {
				double val;
				// Read each dimension value and if valid, add it to the point
				if (!(inputFile >> val)) {
					std::cerr << "Error reading data point values." << std::endl;
					return false;
				}
				point.addDimension(val);
			}
			dataset.push_back(point);
		}

		// Close the input file
		inputFile.close();
		return true;
	}

	// Utility function to check if a vector contains a value
	template <typename T>
	// Returns true if value is found in vec, false otherwise
	bool contains(const std::vector<T>& vec, const T& value) {
		for (const auto& elem : vec) {
			if (elem == value) return true;
		}
		return false;
	}

	// Selects K centers uniformly at random from the existing data points
	void selectAndPrintCenters() {
		// Seed RNG
		std::srand(static_cast<unsigned int>(std::time(0)));

		std::vector<int> selectedIndices;

		// Loop until we have found K unique centers
		while (selectedIndices.size() < (size_t)numClusters) {
			// Select index uniformly at random [cite: 26]
			int randomIndex = std::rand() % numOfPoints;

			// Ensure we don't pick the same point twice
			if (!contains(selectedIndices, randomIndex)) {
				selectedIndices.push_back(randomIndex);

				// Print the point immediately
				dataset[randomIndex].print();
			}
		}
	}	

};