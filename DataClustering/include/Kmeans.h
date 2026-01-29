#pragma once

#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Point.h"

// Implements K-means clustering algorithm.
class Kmeans {
private:
	std::string file_name_;
	int num_clusters_;
	int max_iterations_;
	int num_of_runs_;
	double convergence_threshold_;
	int num_of_points_;
	int dimensionality_;
	std::vector<Point> dataset_;

public:
	// Constructor
	Kmeans(
		const std::string& file_name,
		int num_clusters,
		int max_iterations,
		double convergence_threshold,
		int num_of_runs);
	// Reads data from the file specified in the constructor.
	bool readData();
	// Prints all data points to standard output.
	void printData() const;
	// Selects K random centers and prints them to standard output and file.
	void selectAndPrintCenters();
	// Implements the K-means algorithm.
	void KmeansAlgorithm();
	// Returns true if value is found in vec.
	template <typename T>
	bool contains(const std::vector<T>& vec, const T& value) {
		for (const auto& elem : vec) {
			if (elem == value) return true;
		}
		return false;
	}
};