#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
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
	// Normalizes the dataset using min-max normalization to [0,1] range.
	// Normalizes across columns (attributes), not rows.
	void minmaxNormalize();
	// Prints all data points to standard output.
	void printData() const;
	// Dumps the dataset to a file for verification.
	void dumpDataToFile(const std::string& filename) const;
	// Selects K random centers and returns them.
	std::vector<Point> selectRandomSelectionCentroids();
	// Selects K random centers using the random partition method and returns them.
	std::vector<Point> selectRandomPartitionCentroids();
	// Assigns each point to its nearest centroid and calculates SSE.
	double calculateSSE(const std::vector<Point>& centroids, std::vector<int>& assignments);
	// Implements the K-means algorithm.
	void runKmeans();
	// Implements the K-means algorithm with detailed metrics tracking for initialization comparison.
	void runKmeansWithMetrics(const std::string& initialization_method, const std::string& normalization_method);
	// Implements the K-means algorithm with singleton cluster handling for coincident centers.
	void runKmeansCoincident();
	// Test method for Iris Bezdek dataset: checks if SSE is at or below global optimum (which should not be possible).
	// Returns true if perfect clustering (78.8514) is reached and algorithm should break.
	bool checkIrisBezdekOptimum(double current_sse) const;
	// Returns true if value is found in vec.
	template <typename T>
	bool contains(const std::vector<T>& vec, const T& value) {
		for (const auto& elem : vec) {
			if (elem == value) return true;
		}
		return false;
	}

	// Runs the internal validation sweep over K = Kmin..Kmax.
	// For each K, runs k-means R times (random partition init), takes the best run,
	// and computes the CH and SW indices. Reports estimated cluster counts.
	void runInternalValidation();

	// Computes the Calinski-Harabasz (CH) index for the given partition.
	// CH(K) = [tr(Sb) / (K-1)] / [SSE / (N-K)]
	// tr(Sw) = SSE (already computed by k-means, per Hint #1 in the spec)
	double computeCH(
		const std::vector<Point>& centroids,
		const std::vector<int>& assignments,
		double sse,
		int k);

	// Computes the Silhouette Width (SW) index for the given partition.
	// SW(K) = mean over all points of s(i), where s(i) = (b(i) - a(i)) / max(a(i), b(i))
	double computeSW(const std::vector<int>& assignments, int k);

	// Validates that a silhouette value is within [-1, 1].
	// Prints an error message if it falls outside this range (indicates a bug).
	void checkSilhouetteRange(double s, int point_index);
};