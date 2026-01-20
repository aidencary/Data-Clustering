#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
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
		int numOfRuns);

	// Read data from the file
	bool readData();

	// Print dataset
	void printData() const;

	// Utility function to test if a vector contains a value
	template <typename T>
	// Returns true if value is found in vec, false otherwise
	bool contains(const std::vector<T>& vec, const T& value) {
		for (const auto& elem : vec) {
			if (elem == value) return true;
		}
		return false;
	}

	// Selects K centers uniformly at random from the existing data points
	void selectAndPrintCenters();

};