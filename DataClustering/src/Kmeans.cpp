#include "../include/Kmeans.h"

Kmeans::Kmeans(
	const std::string& file_name,
	int num_clusters,
	int max_iterations,
	double convergence_threshold,
	int num_of_runs)
	: file_name_(file_name),
	  num_clusters_(num_clusters),
	  max_iterations_(max_iterations),
	  convergence_threshold_(convergence_threshold),
	  num_of_runs_(num_of_runs),
	  num_of_points_(0),
	  dimensionality_(0) {}

bool Kmeans::readData() {
	std::ifstream inputFile(file_name_);
	
	if (!inputFile.is_open()) {
		std::string pathWithPrefix = "../datasets/" + file_name_;
		inputFile.open(pathWithPrefix);
		
		if (!inputFile.is_open()) {
			std::cerr << "Error opening file: " << file_name_ << " or " << pathWithPrefix << std::endl;
			return false;
		}
	}

	if (!(inputFile >> num_of_points_ >> dimensionality_)) {
		std::cerr << "Error reading number of points and dimensionality." << std::endl;
		return false;
	}

	for (int i = 0; i < num_of_points_; ++i) {
		Point point;
		for (int d = 0; d < dimensionality_; ++d) {
			double val;
			if (!(inputFile >> val)) {
				std::cerr << "Error reading data point values." << std::endl;
				return false;
			}
			point.addDimension(val);
		}
		dataset_.push_back(point);
	}
	inputFile.close();
	return true;
}

void Kmeans::printData() const {
	for (const auto& point : dataset_) {
		point.print();
	}
}

std::vector<Point> Kmeans::selectCenters() {
	// Step 1 of K-means Algorithm: Select K points as initial centroids
	// Centroids are selected uniformly at random from the dataset using C++11 <random> library
	std::random_device rd; // seed source for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
	std::uniform_int_distribution<> dis(0, num_of_points_ - 1);

	std::vector<int> selectedIndices;
	std::vector<Point> centers;

	// Loop until we have found K unique centers
	while (selectedIndices.size() < (size_t)num_clusters_) {
		// Select index uniformly at random
		int randomIndex = dis(gen);

		// Ensure we don't pick the same point twice
		if (!contains(selectedIndices, randomIndex)) {
			selectedIndices.push_back(randomIndex);
			centers.push_back(dataset_[randomIndex]);
		}
	}
	return centers;
}

bool Kmeans::checkIrisBezdekOptimum(double currentSSE) const {
	// Test condition for the Iris Bezdek dataset
	// Check if SSE is lower than the known global optimum
	if (currentSSE < 78.8514) {
		std::cerr << "Lower than global optimum for Iris Bezdek dataset, something is wrong." << std::endl;
	}
	
	// Perfect Clustering Check for Iris Bezdek dataset
	double roundedSSE = std::round(currentSSE * 10000.0) / 10000.0;
	if (roundedSSE == 78.8514) {
		std::cout << "----\nGlobal opt (78.8514) reached\n----" << std::endl;
		return true; // Perfect clustering - signal to break
	}
	return false;
}

void Kmeans::KmeansAlgorithm() {
	// Algorithm 7.1 Basic K-means Algorithm (from Cluster Analysis Basic Concepts and Algorithms)
	// Do not use pow() or sqrt()!
	// Double variables use max() to not round or truncate double vals

	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << std::endl;
		std::cout << "-----" << std::endl;

		// Step 1: Select K points as initial centroids
		std::vector<Point> centroids = selectCenters();
		
		// Vector to hold cluster assignments for each point
		std::vector<int> assignments(num_of_points_);

		// Step 2: Repeat until convergence
		double previousSSE = std::numeric_limits<double>::max();

		for (int iteration = 0; iteration < max_iterations_; ++iteration) {
			// Step 3: Form K clusters by assigning each point to its closest centroid
			for (int i = 0; i < num_of_points_; ++i) {
				double minDistance = std::numeric_limits<double>::max();
				int nearestCluster = 0;
				
				for (int k = 0; k < num_clusters_; ++k) {
					// Calculate squared Euclidean distance (no sqrt needed for comparison)
					double squaredDist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - centroids[k].getVal(d);
						squaredDist += diff * diff;
					}
					
					// Find the nearest cluster center
					if (squaredDist < minDistance) {
						minDistance = squaredDist;
						nearestCluster = k;
					}
				}
				// Assign point to nearest cluster
				assignments[i] = nearestCluster;
			}

			// Step 4: Recompute the centroid of each cluster
			std::vector<Point> newCenters;
			std::vector<int> clusterSizes(num_clusters_, 0);
			
			// Count points in each cluster
			for (int i = 0; i < num_of_points_; ++i) {
				clusterSizes[assignments[i]]++;
			}
			
			// Calculate new centers
			for (int k = 0; k < num_clusters_; ++k) {
				// Sum dimensions of points assigned to this cluster
				std::vector<double> sums(dimensionality_, 0.0);

				for (int i = 0; i < num_of_points_; ++i) {
					// If point i is assigned to cluster k, add its dimensions to sums
					if (assignments[i] == k) {
						for (int d = 0; d < dimensionality_; ++d) {
							sums[d] += dataset_[i].getVal(d);
						}
					}
				}
				
				// Calculate mean for each dimension to get new center
				Point newCenter;

				for (int d = 0; d < dimensionality_; ++d) {
					// Avoid dividing by zero
					if (clusterSizes[k] > 0) {
						newCenter.addDimension(sums[d] / clusterSizes[k]);
					} else {
						// If a cluster has no points assigned, retain the old center
						newCenter.addDimension(centroids[k].getVal(d));
					}
				}
				newCenters.push_back(newCenter);
			}
			
			centroids = newCenters;

			// Calculate SSE (Sum of Squared Error)/Scatter
			double currentSSE = 0.0;
			// Calculate the error of each data point to its assigned centroid and then compute the total SSE.
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - centroids[cluster].getVal(d);
					currentSSE += diff * diff;
				}
			}
			
			std::cout << "Iteration " << (iteration + 1) << ": SSE = " << currentSSE << std::endl;

			/* Uncomment to enable Iris Bezdek dataset test
			if (checkIrisBezdekOptimum(currentSSE)) {
			     break; // Perfect clustering reached
			}
			*/

			// Step 5: Check if centroids have converged (until centroids do not change)
			// Convergence Check: Check relative improvement in SSE
			if (previousSSE != std::numeric_limits<double>::max()) {
				double relativeImprovement = (previousSSE - currentSSE) / previousSSE;
				if (relativeImprovement < convergence_threshold_) {
					break; // Converged
				}
			}
			previousSSE = currentSSE;
		}
	}

}
