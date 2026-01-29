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

// Selects K centers uniformly at random from the existing data points
void Kmeans::selectAndPrintCenters() {
	std::random_device rd; // seed source for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
	std::uniform_int_distribution<> dis(0, num_of_points_ - 1);

	std::vector<int> selectedIndices;
	
	// Create output file in the output folder
	std::string outputFileName = "../output/output_" + file_name_;
	std::ofstream outputFile(outputFileName);
	
	if (!outputFile.is_open()) {
		std::cerr << "Error: Could not create output file: " << outputFileName << std::endl;
	}

	// Loop until we have found K unique centers
	while (selectedIndices.size() < (size_t)num_clusters_) {
		// Select index uniformly at random
		int randomIndex = dis(gen);

		// Ensure we don't pick the same point twice
		if (!contains(selectedIndices, randomIndex)) {
			selectedIndices.push_back(randomIndex);

			// Print the point
			dataset_[randomIndex].print();
			
			// Write to a file
			if (outputFile.is_open()) {
				dataset_[randomIndex].print(outputFile);
			}
		}
	}
	if (outputFile.is_open()) {
		outputFile.close();
	}

}

void Kmeans::KmeansAlgorithm() {
	// Implementation of the K-means algorithm would go here
	std::random_device rd; // seed source for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
	std::uniform_int_distribution<> dis(0, num_of_points_ - 1);

	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << std::endl;
		std::cout << "-----" << std::endl;

		// Vector to hold K cluster centers (Points)
		std::vector<Point> clusterCenters;
		
		// Vector to hold cluster assignments for each point
		std::vector<int> assignments(num_of_points_);

		// Select K unique random points as initial cluster centers
		std::vector<int> selectedIndices;
		while (selectedIndices.size() < (size_t)num_clusters_) {
			int randomIndex = dis(gen);
			if (!contains(selectedIndices, randomIndex)) {
				selectedIndices.push_back(randomIndex);
				clusterCenters.push_back(dataset_[randomIndex]);
			}
		}
		// K-means iterations (using max() to not round or truncate double vals)
		double previousSSE = std::numeric_limits<double>::max();

		for (int iteration = 0; iteration < max_iterations_; ++iteration) {
			// Assignment Step: Assign each point to the nearest cluster center
			for (int i = 0; i < num_of_points_; ++i) {
				double minDistance = std::numeric_limits<double>::max();
				int nearestCluster = 0;
				
				for (int k = 0; k < num_clusters_; ++k) {
					// Calculate squared Euclidean distance (L2)
					double squaredDist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - clusterCenters[k].getVal(d);
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

			// Update Step: Recalculate cluster centers
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
					if (assignments[i] == k) {
						for (int d = 0; d < dimensionality_; ++d) {
							sums[d] += dataset_[i].getVal(d);
						}
					}
				}
				
				Point newCenter;
				for (int d = 0; d < dimensionality_; ++d) {
					if (clusterSizes[k] > 0) {
						newCenter.addDimension(sums[d] / clusterSizes[k]);
					} else {
						// If a cluster has no points assigned, retain the old center
						newCenter.addDimension(clusterCenters[k].getVal(d));
					}
				}
				newCenters.push_back(newCenter);
			}
			
			clusterCenters = newCenters;

			// Calculate SSE (Sum of Squared Error)
			double currentSSE = 0.0;
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - clusterCenters[cluster].getVal(d);
					currentSSE += diff * diff;
				}
			}
			
			std::cout << "Iteration " << (iteration + 1) << ": SSE = " << currentSSE << std::endl;

			/*
			// A test condition for the Iris Bezdek dataset for if the SSE is lower than the known global optimum
			if (currentSSE < 78.8514) {
				std::cerr << "Lower than global optimum for Iris Bezdek dataset, something is wrong." << std::endl;
			}
			*/
			
			/*
			// Perfect Clustering Check for Iris Bezdek dataset
			double temp = currentSSE = std::round(currentSSE * 10000.0) / 10000.0;
			if (currentSSE == 78.8514) {
				std::cout << "----\nGlobal opt (78.8514) reached\n----" << std::endl;
				break; // Perfect clustering
			}
			*/

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
