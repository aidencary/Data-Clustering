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
	  dimensionality_(0),
	  num_true_clusters_(0) {}

bool Kmeans::readData() {
	std::ifstream input_file(file_name_);
	
	if (!input_file.is_open()) {
		std::string path_with_prefix = "../datasets/" + file_name_;
		input_file.open(path_with_prefix);
		
		if (!input_file.is_open()) {
			std::cerr << "Error opening file: " << file_name_ << " or " << path_with_prefix << "\n";
			return false;
		}
	}

	if (!(input_file >> num_of_points_ >> dimensionality_)) {
		std::cerr << "Error reading number of points and dimensionality." << "\n";
		return false;
	}

	for (int i = 0; i < num_of_points_; ++i) {
		Point point;
		for (int d = 0; d < dimensionality_; ++d) {
			double val;
			if (!(input_file >> val)) {
				std::cerr << "Error reading data point values." << "\n";
				return false;
			}
			point.addDimension(val);
		}
		dataset_.push_back(point);
	}
	input_file.close();
	return true;
}

// Reads the Phase 5 labeled data format.
// Header: N D+1 K_true (three integers).
// Each subsequent line: D doubles + one int true cluster label in [0, K_true - 1].
bool Kmeans::readLabeledData() {
	std::ifstream input_file(file_name_);

	// Try the phase5 datasets directory if the bare filename isn't found
	if (!input_file.is_open()) {
		std::string path_with_prefix = "../phase5_data_sets/" + file_name_;
		input_file.open(path_with_prefix);

		if (!input_file.is_open()) {
			std::cerr << "Error opening file: " << file_name_ << " or " << path_with_prefix << "\n";
			return false;
		}
	}

	// Read the header: N, D+1, K_true
	int dimensionality_plus_one = 0;
	if (!(input_file >> num_of_points_ >> dimensionality_plus_one >> num_true_clusters_)) {
		std::cerr << "Error reading labeled-data header (expected N D+1 K_true).\n";
		return false;
	}
	// The D+1 value includes the trailing label, so actual D = D+1 - 1
	dimensionality_ = dimensionality_plus_one - 1;

	if (dimensionality_ <= 0 || num_true_clusters_ <= 0 || num_of_points_ <= 0) {
		std::cerr << "Invalid header values: N=" << num_of_points_
		          << ", D+1=" << dimensionality_plus_one
		          << ", K_true=" << num_true_clusters_ << "\n";
		return false;
	}

	// Reserve space for one label per point
	true_labels_.assign(num_of_points_, 0);

	for (int i = 0; i < num_of_points_; ++i) {
		Point point;
		for (int d = 0; d < dimensionality_; ++d) {
			double val;
			if (!(input_file >> val)) {
				std::cerr << "Error reading data point values.\n";
				return false;
			}
			point.addDimension(val);
		}
		// Read the true cluster label that follows the D attribute values
		int label;
		if (!(input_file >> label)) {
			std::cerr << "Error reading true cluster label for point " << i << ".\n";
			return false;
		}
		if (label < 0 || label >= num_true_clusters_) {
			std::cerr << "True label " << label << " for point " << i
			          << " is outside [0, " << (num_true_clusters_ - 1) << "].\n";
			return false;
		}
		true_labels_[i] = label;
		dataset_.push_back(point);
	}
	input_file.close();
	return true;
}

void Kmeans::printData() const {
	for (const auto& point : dataset_) {
		point.print();
	}
}

void Kmeans::minmaxNormalize() {
	// Min-max normalization to [0,1] range
	// Formula: v' = (v - min(A)) / (max(A) - min(A))
	// Normalize across columns (attributes), not rows
	// https://www.geeksforgeeks.org/machine-learning/data-normalization-in-data-mining/

	// Handle empty dataset
	if (num_of_points_ == 0 || dimensionality_ == 0) {
		std::cerr << "Cannot normalize empty dataset." << "\n";
		return;
	}

	// For each attribute (dimension/column)
	for (int d = 0; d < dimensionality_; ++d) {
		// Find min and max for this attribute
		double min_val = dataset_[0].getVal(d);
		double max_val = dataset_[0].getVal(d);
		
		// Loop through all points to find min and max for this attribute
		for (int i = 1; i < num_of_points_; ++i) {
			double val = dataset_[i].getVal(d);
			if (val < min_val) min_val = val;
			if (val > max_val) max_val = val;
		}

		// Calculate range
		double range = max_val - min_val;

		// Normalize all values for this attribute.
		// Handle division by zero: if range is 0, all values are identical.
		// In this case set all normalized values to 0.0 (a valid choice in [0,1]).
		// In C++, dividing by zero for a double results in infinity or NaN (not a number) and does not crash the program,
		// but it will produce invalid results that propagate through the calculations.
		if (range == 0.0) {
			// All values are identical for this attribute
			for (int i = 0; i < num_of_points_; ++i) {
				dataset_[i].setVal(d, 0.0);
			}
		} else {
			// Apply min-max normalization: v' = (v - min) / (max - min)
			for (int i = 0; i < num_of_points_; ++i) {
				double original_val = dataset_[i].getVal(d);
				double normalized_val = (original_val - min_val) / range;
				dataset_[i].setVal(d, normalized_val);
			}
		}
	}
}

void Kmeans::dumpDataToFile(const std::string& filename) const {
	// Dump the dataset to a file for verification
	std::ofstream output_file(filename);
	
	// Check if file opened successfully
	if (!output_file.is_open()) {
		std::cerr << "Error opening file for writing: " << filename << "\n";
		return;
	}

	// Write the header (number of points and dimensionality)
	output_file << num_of_points_ << " " << dimensionality_ << "\n";

	// Write each point
	output_file << std::fixed << std::setprecision(15);
	// Loop through each point and write its dimensions to the file
	for (const auto& point : dataset_) {
		// Loop through each dimension of the point and write it to the file, separated by spaces
		for (int d = 0; d < dimensionality_; ++d) {
			output_file << point.getVal(d);
			if (d < dimensionality_ - 1) {
				output_file << " ";
			}
		}
		output_file << "\n";
	}

	output_file.close();
	std::cout << "Data dumped to: " << filename << "\n";
}

std::vector<Point> Kmeans::selectRandomSelectionCentroids() {
	// Step 1 of K-means Algorithm: Select K points as initial centroids
	// Centroids are selected uniformly at random from the dataset using C++11 <random> library
	std::random_device rd; // seed source for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
	std::uniform_int_distribution<> dis(0, num_of_points_ - 1);

	std::vector<int> selected_indices;
	std::vector<Point> centers;

	// Loop until we have found K unique centers
	while (selected_indices.size() < static_cast<size_t>(num_clusters_)) {
		// Select index uniformly at random
		int random_index = dis(gen);

		// Ensure we don't pick the same point twice
		if (!contains(selected_indices, random_index)) {
			selected_indices.push_back(random_index);
			centers.push_back(dataset_[random_index]);
		}
	}
	return centers;
}

std::vector<Point> Kmeans::selectRandomPartitionCentroids() {
	// Random Partition initialization method:
	// 1. Assign each point to a cluster selected uniformly at random
	// 2. Take the centroids of these initial clusters as the initial centers
	std::random_device rd; // seed source for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
	std::uniform_int_distribution<> dis(0, num_clusters_ - 1);
	
	// Step 1: Randomly assign each point to a cluster
	std::vector<int> assignments(num_of_points_);
	for (int i = 0; i < num_of_points_; ++i) {
		assignments[i] = dis(gen);
	}
	
	// Step 2: Compute centroid of each cluster
	// Initialize sums for each cluster and dimension
	std::vector<std::vector<double>> sums(
		num_clusters_,
		std::vector<double>(dimensionality_, 0.0)
	);
	std::vector<int> cluster_sizes(num_clusters_, 0);
	
	// Accumulate sums and counts
	for (int i = 0; i < num_of_points_; ++i) {
		// Get the cluster assignment for this point using the same random generator and distribution
		// to reduce both time and memory overhead by not storing the random cluster assignment for each point in a separate vector.
		int cluster = dis(gen);
		// Update cluster size
		cluster_sizes[cluster]++;
		// Add this point's dimensions to its cluster sum
		for (int d = 0; d < dimensionality_; ++d) {
			sums[cluster][d] += dataset_[i].getVal(d);
		}
	}
	
	// Step 3: Calculate centroids
	std::vector<Point> centers;
	// Calculate mean for each dimension to get new center, handling empty clusters by assigning a random point from the dataset
	for (int k = 0; k < num_clusters_; ++k) {
		Point center;
		// If cluster has points assigned, calculate the mean.
		// Otherwise, assign a random point from the dataset as the center.
		// The fallback point is selected once per empty cluster so all dimensions come from the same point
		int fallback_point = -1; // The index of the random point to use if this cluster is empty
		if (cluster_sizes[k] == 0) {
			std::uniform_int_distribution<> point_dis(0, num_of_points_ - 1);
			fallback_point = point_dis(gen);
		}
		for (int d = 0; d < dimensionality_; ++d) {
			if (cluster_sizes[k] > 0) {
				center.addDimension(sums[k][d] / cluster_sizes[k]);
			} else {
				center.addDimension(dataset_[fallback_point].getVal(d));
			}
		}
		centers.push_back(center);
	}
	
	return centers;
}

double Kmeans::calculateSSE(const std::vector<Point>& centroids, std::vector<int>& assignments) {
	// Assign each point to nearest centroid and calculate SSE
	for (int i = 0; i < num_of_points_; ++i) {
		// Calculate squared Euclidean distance to each centroid and find the nearest one
		double min_distance = std::numeric_limits<double>::max();
		int nearest_cluster = 0;
		
		// Loop through each centroid to find the nearest one for this point
		for (int k = 0; k < num_clusters_; ++k) {
			double squared_dist = 0.0;
			// Calculate squared distance from this point to the centroid
			for (int d = 0; d < dimensionality_; ++d) {
				double diff = dataset_[i].getVal(d) - centroids[k].getVal(d);
				squared_dist += diff * diff;
			}
			// Update nearest cluster if this centroid is closer
			if (squared_dist < min_distance) {
				min_distance = squared_dist;
				nearest_cluster = k;
			}
		}
		assignments[i] = nearest_cluster;
	}
	
	// Calculate SSE
	double sse = 0.0;
	for (int i = 0; i < num_of_points_; ++i) {
		int cluster = assignments[i];
		for (int d = 0; d < dimensionality_; ++d) {
			// Calculate squared error contribution of this point to its assigned cluster
			double diff = dataset_[i].getVal(d) - centroids[cluster].getVal(d);
			sse += diff * diff;
		}
	}
	return sse;
}

bool Kmeans::checkIrisBezdekOptimum(double current_sse) const {
	// Test condition for the Iris Bezdek dataset
	// Check if SSE is lower than the known global optimum
	if (current_sse < 78.8514) {
		std::cerr << "Lower than global optimum for Iris Bezdek dataset, something is wrong.\n";
	}

	// Perfect Clustering Check for Iris Bezdek dataset
	double rounded_sse = std::round(current_sse * 10000.0) / 10000.0;
	if (rounded_sse == 78.8514) {
		std::cout << "----\nGlobal opt (78.8514) reached\n----\n";
		return true; // Perfect clustering - signal to break
	}
	return false;
}

void Kmeans::runKmeans() {
	// Algorithm 7.1 Basic K-means Algorithm (from Cluster Analysis Basic Concepts and Algorithms)
	// Do not use pow() or sqrt()!
	// Does not handle singleton clusters (clusters with only one point, which is their center)
	// May have issues with coincident centers in datasets with duplicate points

	// Create output file in the output folder
	std::string output_file_name = "../output/output_" + file_name_;
	std::ofstream output_file(output_file_name);
	
	if (!output_file.is_open()) {
		std::cerr << "Error: Could not create output file: " << output_file_name << "\n";
	}

	// Track best run across all executions
	double best_sse = std::numeric_limits<double>::max();
	int best_run = 0;
	
	// Run the K-means algorithm for the specified number of runs
	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << "\n";
		std::cout << "-----" << "\n";
		if (output_file.is_open()) {
			output_file << "Run " << (run + 1) << "\n";
			output_file << "-----" << "\n";
		}

		// Step 1: Select K points as initial centroids
		std::vector<Point> centroids = selectRandomSelectionCentroids();
		
		// Vector to hold cluster assignments for each point
		std::vector<int> assignments(num_of_points_);

		// Step 2: Repeat until convergence
		double previous_sse = std::numeric_limits<double>::max();
		double current_sse = 0.0;

		for (int iteration = 0; iteration < max_iterations_; ++iteration) {
			// Step 3: Form K clusters by assigning each point to its closest centroid
			for (int i = 0; i < num_of_points_; ++i) {
				double min_distance = std::numeric_limits<double>::max();
				int nearest_cluster = 0;
				
				for (int k = 0; k < num_clusters_; ++k) {
					// Calculate squared Euclidean distance (not using sqrt or pow)
					double squared_dist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - centroids[k].getVal(d);
						squared_dist += diff * diff;
					}
					
					// Find the nearest cluster center
					if (squared_dist < min_distance) {
						min_distance = squared_dist;
						nearest_cluster = k;
					}
				}
				// Assign point to nearest cluster
				assignments[i] = nearest_cluster;
			}

			// Step 4: Recompute the centroid of each cluster
			// Initialize sums for each cluster and dimension
			std::vector<std::vector<double>> sums(
				num_clusters_,
				std::vector<double>(dimensionality_, 0.0)
			);
			
			// Initialize cluster sizes
			std::vector<int> cluster_sizes(num_clusters_, 0);
			
			// Accumulate sums and counts
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				cluster_sizes[cluster]++;
				
				// Add this point's dimensions to its cluster sum
				for (int d = 0; d < dimensionality_; ++d) {
					sums[cluster][d] += dataset_[i].getVal(d);
				}
			}
			
			// Compute new centroids
			std::vector<Point> new_centers;
			
			for (int k = 0; k < num_clusters_; ++k) {
				// Calculate mean for each dimension to get new center
				Point new_center;
				
				for (int d = 0; d < dimensionality_; ++d) {
					// Avoid dividing by zero
					if (cluster_sizes[k] > 0) {
						new_center.addDimension(sums[k][d] / cluster_sizes[k]);
					} else {
						// If a cluster has no points assigned, retain the old center
						new_center.addDimension(centroids[k].getVal(d));
					}
				}
				new_centers.push_back(new_center);
			}
			
			centroids = new_centers;

			// Calculate SSE (Sum of Squared Error) aka Scatter
			current_sse = 0.0;
			// Calculate the error of each data point to its assigned centroid and then compute the total SSE.
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - centroids[cluster].getVal(d);
					current_sse += diff * diff;
				}
			}
			
			std::cout << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << "\n";
			if (output_file.is_open()) {
				output_file << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << "\n";
			}

			/* Uncomment to enable Iris Bezdek dataset test
			if (checkIrisBezdekOptimum(current_sse)) {
			     break; // Perfect clustering reached
			}
			*/

			// Step 5: Check if SSE improvement is below the convergence threshold
			if (previous_sse != std::numeric_limits<double>::max()) {
				// Check if SSE hasn't changed at all (converged perfectly)
				if (current_sse == previous_sse) {
					break; // No change, converged
				}
				
				// Check relative improvement
				double relative_improvement = (previous_sse - current_sse) / previous_sse;
				if (relative_improvement < convergence_threshold_) {
					break; // Converged
				}
			}
			previous_sse = current_sse;
		}
		
		// Add newline after last iteration of the run
		std::cout << "\n";
		if (output_file.is_open()) {
			output_file << "\n";
		}
		
		// Track if this run achieved the best SSE
		if (current_sse < best_sse) {
			best_sse = current_sse;
			best_run = run + 1;
		}
	}
	
	// Display best run after all runs complete
	std::cout << "\nBest Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse << "\n";
	if (output_file.is_open()) {
		output_file << "\nBest Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse;
		output_file.close();
	}

	/* Write best run to best_runs.txt file
	std::ofstream best_runs_file("../output/best_runs.txt", std::ios::app);
	if (best_runs_file.is_open()) {
		best_runs_file << file_name_ << ": Best Run = " << best_run << ", SSE = " << std::fixed << std::setprecision(4) << best_sse << "\n";
		best_runs_file.close();
	} else {
		std::cerr << "Error: Could not open best_runs.txt file" << "\n";
	}
	*/

}

void Kmeans::runKmeansCoincident() {
	// Algorithm 7.1 Basic K-means Algorithm (from Cluster Analysis Basic Concepts and Algorithms)
	// with singleton cluster handling for coincident centers
	// Do not use pow() or sqrt()!

	// Create output file in the output folder
	std::string output_file_name = "../output/output_" + file_name_;
	std::ofstream output_file(output_file_name);
	
	if (!output_file.is_open()) {
		std::cerr << "Error: Could not create output file: " << output_file_name << "\n";
	}

	// Track best run across all executions
	double best_sse = std::numeric_limits<double>::max();
	int best_run = 0;
	
	// Run the K-means algorithm for the specified number of runs
	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << "\n";
		std::cout << "-----" << "\n";
		if (output_file.is_open()) {
			output_file << "Run " << (run + 1) << "\n";
			output_file << "-----" << "\n";
		}

		// Step 1: Select K points as initial centroids
		std::vector<Point> centroids = selectRandomSelectionCentroids();
		
		// Vector to hold cluster assignments for each point
		std::vector<int> assignments(num_of_points_);

		// Step 2: Repeat until convergence
		double previous_sse = std::numeric_limits<double>::max();
		double current_sse = 0.0;

		for (int iteration = 0; iteration < max_iterations_; ++iteration) {
			// Step 3: Form K clusters by assigning each point to its closest centroid
			for (int i = 0; i < num_of_points_; ++i) {
				double min_distance = std::numeric_limits<double>::max();
				int nearest_cluster = 0;
				
				for (int k = 0; k < num_clusters_; ++k) {
					// Calculate squared Euclidean distance (not using sqrt or pow)
					double squared_dist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - centroids[k].getVal(d);
						squared_dist += diff * diff;
					}
					
					// Find the nearest cluster center
					if (squared_dist < min_distance) {
						min_distance = squared_dist;
						nearest_cluster = k;
					}
				}
				// Assign point to nearest cluster
				assignments[i] = nearest_cluster;
			}

			// Handle singleton clusters (clusters with only one point, which is their center)
			// This prevents issues with coincident centers in datasets with duplicate points
			std::vector<int> cluster_sizes(num_clusters_, 0);
			for (int i = 0; i < num_of_points_; ++i) {
				cluster_sizes[assignments[i]]++;
			}
			
			// Find singleton clusters
			std::vector<int> singleton_clusters;
			for (int k = 0; k < num_clusters_; ++k) {
				if (cluster_sizes[k] == 1) {
					singleton_clusters.push_back(k);
				}
			}
			
			// If singleton clusters exist, reassign their centers
			if (!singleton_clusters.empty()) {
				// For each singleton cluster, find the point in non-singleton clusters
				// that contributes most to its cluster's error
				
				/*
				// Print that we have singleton clusters and are handling them
				std::cout << "Handling " << singleton_clusters.size() << " singleton cluster(s)..." << "\n";
				if (output_file.is_open()) {
					output_file << "Handling " << singleton_clusters.size() << " singleton cluster(s)..." << "\n";
				}
				*/
				

				for (int s = 0; s < static_cast<int>(singleton_clusters.size()); ++s) {
					int singleton_cluster_id = singleton_clusters[s];
					
					// Find the point with maximum contribution to error in non-singleton clusters
					double max_error_contribution = -1.0;
					int point_with_max_error = -1;
					int source_cluster = -1;
					
					for (int i = 0; i < num_of_points_; ++i) {
						int cluster = assignments[i];
						
						// Only consider points from non-singleton clusters with size > 1
						if (cluster_sizes[cluster] > 1) {
							// Calculate this point's contribution to its cluster's error
							double error_contribution = 0.0;
							for (int d = 0; d < dimensionality_; ++d) {
								double diff = dataset_[i].getVal(d) - centroids[cluster].getVal(d);
								error_contribution += diff * diff;
							}
							
							// Track point with maximum error contribution
							if (error_contribution > max_error_contribution) {
								max_error_contribution = error_contribution;
								point_with_max_error = i;
								source_cluster = cluster;
							}
						}
					}
					
					// If we found a point, make it the center of the singleton cluster
					if (point_with_max_error != -1) {
						// Copy the point's coordinates to the singleton cluster's centroid
						Point new_singleton_center;
						for (int d = 0; d < dimensionality_; ++d) {
							new_singleton_center.addDimension(dataset_[point_with_max_error].getVal(d));
						}
						centroids[singleton_cluster_id] = new_singleton_center;
						
						// Reassign this point to the singleton cluster
						assignments[point_with_max_error] = singleton_cluster_id;
						
						// Update cluster sizes
						cluster_sizes[source_cluster]--;
						cluster_sizes[singleton_cluster_id]++;
					}
				}
			}

			// Step 4: Recompute the centroid of each cluster
			// Initialize sums for each cluster and dimension
			std::vector<std::vector<double>> sums(
				num_clusters_,
				std::vector<double>(dimensionality_, 0.0)
			);
			
			// Initialize cluster sizes
			cluster_sizes.assign(num_clusters_, 0);
			
			// Accumulate sums and counts
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				cluster_sizes[cluster]++;
				
				// Add this point's dimensions to its cluster sum
				for (int d = 0; d < dimensionality_; ++d) {
					sums[cluster][d] += dataset_[i].getVal(d);
				}
			}
			
			// Compute new centroids
			std::vector<Point> new_centers;
			
			for (int k = 0; k < num_clusters_; ++k) {
				// Calculate mean for each dimension to get new center
				Point new_center;
				
				for (int d = 0; d < dimensionality_; ++d) {
					// Avoid dividing by zero
					if (cluster_sizes[k] > 0) {
						new_center.addDimension(sums[k][d] / cluster_sizes[k]);
					} else {
						// If a cluster has no points assigned, retain the old center
						new_center.addDimension(centroids[k].getVal(d));
					}
				}
				new_centers.push_back(new_center);
			}
			
			centroids = new_centers;

			// Calculate SSE (Sum of Squared Error) aka Scatter
			current_sse = 0.0;
			// Calculate the error of each data point to its assigned centroid and then compute the total SSE.
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - centroids[cluster].getVal(d);
					current_sse += diff * diff;
				}
			}
			
			std::cout << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << "\n";
			if (output_file.is_open()) {
				output_file << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << "\n";
			}

			/* Uncomment to enable Iris Bezdek dataset test
			if (checkIrisBezdekOptimum(current_sse)) {
			     break; // Perfect clustering reached
			}
			*/

			// Step 5: Check if SSE improvement is below the convergence threshold
			if (previous_sse != std::numeric_limits<double>::max()) {
				// Check if SSE hasn't changed at all (converged perfectly)
				if (current_sse == previous_sse) {
					break; // No change, converged
				}
				
				// Check relative improvement
				double relative_improvement = (previous_sse - current_sse) / previous_sse;
				if (relative_improvement < convergence_threshold_) {
					break; // Converged
				}
			}
			previous_sse = current_sse;
		}
		
		// Add newline after last iteration of the run
		std::cout << "\n";
		if (output_file.is_open()) {
			output_file << "\n";
		}
		
		// Track if this run achieved the best SSE
		if (current_sse < best_sse) {
			best_sse = current_sse;
			best_run = run + 1;
		}
	}
	
	// Display best run after all runs complete
	std::cout << "Best Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse << "\n";
	if (output_file.is_open()) {
		output_file << "Best Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse;
		output_file.close();
	}

	/*
	// Write best run to best_runs.txt file
	std::ofstream best_runs_file("../output/best_runs.txt", std::ios::app);
	if (best_runs_file.is_open()) {
		best_runs_file << file_name_ << ": Best Run = " << best_run << ", SSE = " << std::fixed << std::setprecision(4) << best_sse << "\n";
		best_runs_file.close();
	} else {
		std::cerr << "Error: Could not open best_runs.txt file" << "\n";
	}
	*/

}

void Kmeans::runKmeansWithMetrics(const std::string& initialization_method, const std::string& normalization_method) {
	/* Used to compare Random Partition and Random Selection initialization methods on the same dataset with the same parameters
	* Tracks: Initial SSE, Final SSE, and Number of Iterations
	* Initial SSE: This is the SSE value computed after the initialization phase, before the
	* clustering phase. It gives us a measure of the effectiveness of an initialization method by itself.
	* Final SSE: This is the SSE value computed after the clustering phase. 
	* It gives us a measure of the effectiveness of an initialization method when its output is refined by kmeans.
	* Note that this is the objective function of the k-means algorithm
	*/

	// Create output file in the output folder with method name in the filename
	std::string output_file_name;
	if (initialization_method == "Random Partition") {
		std::filesystem::create_directories("../output_random_partition");
		output_file_name = "../output_random_partition/output_" + file_name_;
	} else if (initialization_method == "Random Selection") {
		std::filesystem::create_directories("../output_random_selection");
		output_file_name = "../output_random_selection/output_" + file_name_;
	} else {
		std::cerr << "Error: Invalid initialization method: " << initialization_method << "\n";
		return;
	}

	std::ofstream output_file(output_file_name);
	
	if (!output_file.is_open()) {
		std::cerr << "Error: Could not create output file: " << output_file_name << "\n";
	}

	std::cout << "Initialization Method: " << initialization_method << "\n" << "\n";

	if (output_file.is_open()) {
		output_file << "Initialization Method: " << initialization_method << "\n" << "\n";
	}

	// Track best run across all executions
	double best_final_sse = std::numeric_limits<double>::max();
	double best_initial_sse = std::numeric_limits<double>::max();
	int best_iterations = std::numeric_limits<int>::max();
	int best_run = 0;
	double total_initial_sse = 0.0;
	double total_final_sse = 0.0;
	int total_iterations = 0;
	
	// Run the K-means algorithm for the specified number of runs
	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << "\n";
		std::cout << "-----" << "\n";
		if (output_file.is_open()) {
			output_file << "Run " << (run + 1) << "\n";
			output_file << "-----" << "\n";
		}

		// Step 1: Select K points as initial centroids
		std::vector<Point> centroids;
		if (initialization_method == "Random Partition") {
			centroids = selectRandomPartitionCentroids();
		} else if (initialization_method == "Random Selection") {
			centroids = selectRandomSelectionCentroids();
		} else {
			std::cerr << "Error: Invalid initialization method: " << initialization_method << "\n";
			return;
		}

		
		// Vector to hold cluster assignments for each point
		std::vector<int> assignments(num_of_points_);
		
		// Calculate Initial SSE using the helper function which assigns points to nearest centroid and computes SSE
		double initial_sse = calculateSSE(centroids, assignments);

		// Track initial SSE for this run and add to total for averaging later
		total_initial_sse += initial_sse;
		
		// Track if this is the best initial SSE so far
		if (initial_sse < best_initial_sse) {
			best_initial_sse = initial_sse;
		}
		
		// Display initial SSE for this run
		std::cout << "Initial SSE: " << std::fixed << std::setprecision(4) << initial_sse << "\n" << "\n";
		if (output_file.is_open()) {
			output_file << "Initial SSE: " << std::fixed << std::setprecision(4) << initial_sse << "\n" << "\n";
		}

		// Step 2: Repeat until convergence
		double previous_sse = std::numeric_limits<double>::max();
		double current_sse = 0.0;
		int iteration_count = 0;

		// Main k-means loop with convergence checks
		for (int iteration = 0; iteration < max_iterations_; ++iteration) {
			iteration_count++;
			
			// Step 3: Form K clusters by assigning each point to its closest centroid
			for (int i = 0; i < num_of_points_; ++i) {
				double min_distance = std::numeric_limits<double>::max();
				int nearest_cluster = 0;
				
				for (int k = 0; k < num_clusters_; ++k) {
					// Calculate squared Euclidean distance (not using sqrt or pow)
					double squared_dist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - centroids[k].getVal(d);
						squared_dist += diff * diff;
					}
					
					// Find the nearest cluster center
					if (squared_dist < min_distance) {
						min_distance = squared_dist;
						nearest_cluster = k;
					}
				}
				// Assign point to nearest cluster
				assignments[i] = nearest_cluster;
			}

			// Step 4: Recompute the centroid of each cluster
			// Initialize sums for each cluster and dimension
			std::vector<std::vector<double>> sums(
				num_clusters_,
				std::vector<double>(dimensionality_, 0.0)
			);
			
			// Initialize cluster sizes
			std::vector<int> cluster_sizes(num_clusters_, 0);
			
			// Accumulate sums and counts
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				cluster_sizes[cluster]++;
				
				// Add this point's dimensions to its cluster sum
				for (int d = 0; d < dimensionality_; ++d) {
					sums[cluster][d] += dataset_[i].getVal(d);
				}
			}
			
			// Compute new centroids
			std::vector<Point> new_centers;
			
			for (int k = 0; k < num_clusters_; ++k) {
				// Calculate mean for each dimension to get new center
				Point new_center;
				
				for (int d = 0; d < dimensionality_; ++d) {
					// Avoid dividing by zero
					if (cluster_sizes[k] > 0) {
						new_center.addDimension(sums[k][d] / cluster_sizes[k]);
					} else {
						// If a cluster has no points assigned, retain the old center
						new_center.addDimension(centroids[k].getVal(d));
					}
				}
				new_centers.push_back(new_center);
			}
			
			centroids = new_centers;

			// Calculate SSE (Sum of Squared Error) aka Scatter
			current_sse = 0.0;
			// Calculate the error of each data point to its assigned centroid and then compute the total SSE.
			for (int i = 0; i < num_of_points_; ++i) {
				int cluster = assignments[i];
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - centroids[cluster].getVal(d);
					current_sse += diff * diff;
				}
			}
			
			std::cout << "Iteration " << iteration_count << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << "\n";
			if (output_file.is_open()) {
				output_file << "Iteration " << iteration_count << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << "\n";
			}

			// Step 5: Check if SSE improvement is below the convergence threshold
			if (previous_sse != std::numeric_limits<double>::max()) {
				// Check if SSE hasn't changed at all (converged perfectly)
				if (current_sse == previous_sse) {
					break; // No change, converged
				}
				
				// Check relative improvement
				double relative_improvement = (previous_sse - current_sse) / previous_sse;
				if (relative_improvement < convergence_threshold_) {
					break; // Converged
				}
			}
			previous_sse = current_sse;
		}
		
		// Track final SSE and iterations
		total_final_sse += current_sse;
		total_iterations += iteration_count;
		
		std::cout << "Final SSE: " << std::fixed << std::setprecision(4) << current_sse << "\n";
		std::cout << "Number of Iterations: " << iteration_count << "\n" << "\n";
		if (output_file.is_open()) {
			output_file << "Final SSE: " << std::fixed << std::setprecision(4) << current_sse << "\n";
			output_file << "Number of Iterations: " << iteration_count << "\n" << "\n";
		}
		
		// Track if this run achieved the best SSE
		if (current_sse < best_final_sse) {
			best_final_sse = current_sse;
			best_run = run + 1;
		}
		
		// Track if this is the best iteration count
		if (iteration_count < best_iterations) {
			best_iterations = iteration_count;
		}
	}
	
	// Display summary statistics after all runs complete
	std::cout << "SUMMARY STATISTICS (" << initialization_method << ")" << "\n";
	std::cout << "Average Initial SSE: " << std::fixed << std::setprecision(4) << (total_initial_sse / num_of_runs_) << "\n";
	std::cout << "Average Final SSE: " << std::fixed << std::setprecision(4) << (total_final_sse / num_of_runs_) << "\n";
	std::cout << "Average Iterations: " << std::fixed << std::setprecision(2) << (static_cast<double>(total_iterations) / num_of_runs_) << "\n";
	std::cout << "Best Run: " << best_run << " with Final SSE = " << std::fixed << std::setprecision(4) << best_final_sse << "\n";
	std::cout << "\nSUMMARY: BestInitSSE=" << std::fixed << std::setprecision(4) << best_initial_sse 
	          << ", BestFinalSSE=" << std::fixed << std::setprecision(4) << best_final_sse 
	          << ", BestIter=" << best_iterations << "\n";

	
	if (output_file.is_open()) {
		output_file << "SUMMARY STATISTICS (" << initialization_method << ")" << "\n";
		output_file << "Average Initial SSE: " << std::fixed << std::setprecision(4) << (total_initial_sse / num_of_runs_) << "\n";
		output_file << "Average Final SSE: " << std::fixed << std::setprecision(4) << (total_final_sse / num_of_runs_) << "\n";
		output_file << "Average Iterations: " << std::fixed << std::setprecision(2) << (static_cast<double>(total_iterations) / num_of_runs_) << "\n";
		output_file << "Best Run: " << best_run << " with Final SSE = " << std::fixed << std::setprecision(4) << best_final_sse << "\n";
		output_file << "\nSUMMARY: BestInitSSE=" << std::fixed << std::setprecision(4) << best_initial_sse 
		            << ", BestFinalSSE=" << std::fixed << std::setprecision(4) << best_final_sse 
		            << ", BestIter=" << best_iterations << "\n";
		output_file.close();
	}
	
	// Write results to CSV file (will move to a Excel file later)
	std::filesystem::create_directories("../output_phase3");
	std::string csv_file_path = "../output_phase3/phase3_results.csv";
	bool file_exists = false;
	
	// Check if file exists by trying to open it for reading
	std::ifstream check_file(csv_file_path);
	if (check_file.good()) {
		file_exists = true;
	}
	check_file.close();
	
	// Open CSV file in append mode
	std::ofstream csv_file(csv_file_path, std::ios::app);
	if (csv_file.is_open()) {
		// Write header if file is new
		if (!file_exists) {
			csv_file << "Dataset,Normalization Method,Initialization Method,Best Initial SSE,Best Final SSE,Best Iterations" << "\n";
		}

		// Extract dataset name from the file name (remove .txt extension)
		std::string dataset_name = file_name_;
		size_t dot_pos = dataset_name.find(".txt");
		if (dot_pos != std::string::npos) {
			dataset_name = dataset_name.substr(0, dot_pos);
		}

		// Write data row
		csv_file << dataset_name << ","
		         << normalization_method << ","
		         << initialization_method << ","
		         << std::fixed << std::setprecision(4) << best_initial_sse << ","
		         << std::fixed << std::setprecision(4) << best_final_sse << ","
		         << best_iterations << "\n";

		csv_file.close();
		std::cout << "Results are written to " << csv_file_path << "\n";
	} else {
		std::cerr << "Error: Could not write to CSV file: " << csv_file_path << "\n";
	}
}

// Prints an error to stderr if the value is out of range.
void Kmeans::checkSilhouetteRange(double s, int point_index) {
	// Silhouette values must always be in [-1, 1]
	if (s < -1.0 || s > 1.0) {
		std::cerr << "Error: Silhouette value " << s
		          << " for point " << point_index
		          << " is outside [-1, 1]. There is a bug!.\n";
	}
}


// Computes the Calinski-Harabasz (CH) index for the given partition.
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", 2nd ed.
//            Section 17.3 (Relative Measures), pg 452-453
//            Eq. 17.49 Calinski-Harabasz variance ratio criterion for a given value of k
//            https://dataminingbook.info/book_html/chap17/book.html 
// Book formula: CH(K) = [tr(S_B) / (K-1)] / [tr(S_W) / (N-K)]
// Since tr(S_W) = SSE -> CH(K) = [tr(Sb) / (K-1)] / [SSE / (N-K)]
// tr(Sb) = sum over dimensions j of: sum over clusters k of: n_k * (mu_k_j - mu_j)^2
double Kmeans::computeCH(
	const std::vector<Point>& centroids,
	const std::vector<int>& assignments,
	double sse,
	int k)
{
	// Step 1: Count how many points belong to each cluster
	std::vector<int> cluster_sizes(k, 0);
	for (int i = 0; i < num_of_points_; ++i) {
		// Increment size counter for the cluster this point belongs to
		cluster_sizes[assignments[i]]++;
	}

	// Step 2: Compute the overall mean of the entire dataset for each dimension
	std::vector<double> overall_mean(dimensionality_, 0.0);
	for (int i = 0; i < num_of_points_; ++i) {
		// Accumulate the sum for each dimension
		for (int d = 0; d < dimensionality_; ++d) {
			overall_mean[d] += dataset_[i].getVal(d);
		}
	}
	// Divide each accumulated sum by N to get the mean
	for (int d = 0; d < dimensionality_; ++d) {
		overall_mean[d] /= num_of_points_;
	}

	// Step 3: Compute tr(Sb) which is the trace of the between-cluster scatter matrix.
	// Only the diagonal elements of Sb are needed.
	// Diagonal element j of Sb = sum over clusters k of: n_k * (mu_k_j - mu_j)^2
	double trace_sb = 0.0;
	for (int d = 0; d < dimensionality_; ++d) {
		// Accumulate the contribution of each cluster to diagonal element j
		for (int ki = 0; ki < k; ++ki) {
			double diff = centroids[ki].getVal(d) - overall_mean[d];
			// Multiply squared difference by cluster size (weight by n_k)
			trace_sb += cluster_sizes[ki] * diff * diff;
		}
	}

	// Step 4: tr(Sw) = SSE
	double trace_sw = sse;

	// Step 5: Compute CH(K) = [tr(Sb) / (K-1)] / [tr(Sw) / (N-K)]
	// Guard against degenerate case where tr(Sw) = 0 (all points on their centroids)
	if (trace_sw == 0.0) {
		std::cerr << "Warning: tr(Sw) = 0 for K=" << k << "; CH is undefined.\n";
		return 0.0;
	}

	// Calculate the CH index using the formula
	double ch = (trace_sb / (k - 1)) / (trace_sw / (num_of_points_ - k));

	return ch;
}


// Computes the Silhouette Width (SW) index for the given partition.
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", 2nd ed.
//            Section 17.2 (Internal Measures), pg. 446-447
//            Eq. 17.45 (per-point silhouette s_i), Eq. 17.46 (overall SC)
//            https://dataminingbook.info/book_html/chap17/book.html
//
// Book notation mapped to this implementation:
//        mu_in(x_i) = mean distance from x_i to all other points in its own cluster
//                   = a_i in my code
//                   = sum_{x_j in C_i, j!=i} ||x_i - x_j|| / (n_i - 1)
//                   - if |C_i| = 1 (singleton), mu_in is undefined; s_i = 0 by convention
//
//   mu_out_min(x_i) = mean distance from x_i to points in the nearest OTHER cluster
//                   = b_i in my code
//                   = min_{j != cluster_i} { sum_{y in C_j} ||x_i - y|| / n_j }
//
//   			 s_i = (mu_out_min(x_i) - mu_in(x_i)) / max{mu_out_min(x_i), mu_in(x_i)}
//                     in [-1, 1]: close to +1 = well-clustered, close to -1 = mis-clustered
//
//   SC (Eq. 17.46)  = (1/n) * sum_{i=1}^{n} s_i - overall silhouette width
//
// Note: ||x_i - x_j|| is Euclidean distance (with sqrt), NOT squared distance.
//       Using squared distance would invalidate the [-1, 1] range of s_i.
double Kmeans::computeSW(const std::vector<int>& assignments, int k) {
	// Step 1: Precompute cluster sizes
	std::vector<int> cluster_sizes(k, 0);
	for (int i = 0; i < num_of_points_; ++i) {
		cluster_sizes[assignments[i]]++;
	}

	// Step 2: Compute silhouette value for each point and accumulate the total
	double total_silhouette = 0.0;
	for (int i = 0; i < num_of_points_; ++i) {
		int ci = assignments[i];

		// Step 2a: If point i is the only member of its cluster, s(i) = 0 by convention
		if (cluster_sizes[ci] == 1) {
			checkSilhouetteRange(0.0, i);
			continue;
		}

		// Step 2b: Compute a(i) - mean distance to all other points in same cluster
		double sum_a = 0.0;
		for (int j = 0; j < num_of_points_; ++j) {
			// Only consider other points in the same cluster
			if (j != i && assignments[j] == ci) {
				// Euclidean distance (with sqrt, as required for silhouette)
				double sq_dist = 0.0;
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - dataset_[j].getVal(d);
					sq_dist += diff * diff;
				}
				sum_a += std::sqrt(sq_dist);
			}
		}
		// Divide by (|C_i| - 1) since we exclude point i itself
		double a_i = sum_a / (cluster_sizes[ci] - 1);

		// Step 2c: Compute b(i) - min mean distance to any other cluster
		double b_i = std::numeric_limits<double>::max();
		for (int ki = 0; ki < k; ++ki) {
			// Skip point i's own cluster
			if (ki == ci) continue;
			// Skip empty clusters (shouldn't happen after k-means, but guard anyway)
			if (cluster_sizes[ki] == 0) continue;

			// Compute mean distance from point i to all points in cluster ki
			double sum_b = 0.0;
			for (int j = 0; j < num_of_points_; ++j) {
				if (assignments[j] == ki) {
					// Euclidean distance (with sqrt)
					double sq_dist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - dataset_[j].getVal(d);
						sq_dist += diff * diff;
					}
					sum_b += std::sqrt(sq_dist);
				}
			}
			// Mean distance from i to cluster ki
			double mean_b = sum_b / cluster_sizes[ki];
			// Keep the minimum across all other clusters
			if (mean_b < b_i) {
				b_i = mean_b;
			}
		}

		// Step 2d: Compute s(i) = (b(i) - a(i)) / max(a(i), b(i))
		double max_ab = (a_i > b_i) ? a_i : b_i;
		double s_i = 0.0;
		if (max_ab > 0.0) {
			s_i = (b_i - a_i) / max_ab;
		}

		// Validate that s(i) is in [-1, 1] — any violation indicates a bug
		checkSilhouetteRange(s_i, i);

		total_silhouette += s_i;
	}

	// Step 3: SW(K) = mean silhouette over all N points
	double sw = total_silhouette / num_of_points_;

	return sw;
}

// Runs the internal validation sweep over K = 2 to Kmax, where Kmax = round(sqrt(N/2))
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", Chapter 17
//            https://dataminingbook.info/book_html/chap17/book.html
// For each K, runs k-means R times using random partition initialization,
// selects the best run (lowest final SSE), and computes CH and SW indices.
// Reports a table of results and writes a CSV file to ../output_phase4/.
void Kmeans::runInternalValidation() {
	// Step 1: Determine K range
	// Kmin is always 2 (at least 2 clusters needed for these indices)
	// Kmax = round(sqrt(N/2))
	int k_min = 2;
	int k_max = static_cast<int>(std::round(std::sqrt(num_of_points_ / 2.0)));

	std::cout << "Running internal validation: Kmin=" << k_min
	          << ", Kmax=" << k_max
	          << ", N=" << num_of_points_ << "\n\n";

	// Prepare output directory and CSV file
	std::filesystem::create_directories("../output_phase4");
	std::string base_name = file_name_;
	auto dot_pos = base_name.rfind('.');
	if (dot_pos != std::string::npos) {
		base_name = base_name.substr(0, dot_pos);
	}
	std::string csv_path = "../output_phase4/results_" + base_name + ".csv";
	std::ofstream csv_file(csv_path);
	if (!csv_file.is_open()) {
		std::cerr << "Error: Could not create output file: " << csv_path << "\n";
	}

	// Write CSV header
	if (csv_file.is_open()) {
		csv_file << "K,CH,SW\n";
	}

	// Vectors to store the CH and SW value for each K value tested
	std::vector<double> ch_values;
	std::vector<double> sw_values;
	std::vector<int> k_values;

	// Step 2: Sweep K from Kmin to Kmax
	for (int k = k_min; k <= k_max; ++k) {
		std::cout << "--- Computing K=" << k << " ---\n";

		// Set the number of clusters for this iteration of the sweep
		num_clusters_ = k;

		// Track the best run for this K
		double best_sse = std::numeric_limits<double>::max();
		std::vector<Point> best_centroids;
		std::vector<int> best_assignments(num_of_points_);

		// Step 2a: Run k-means R times and keep the run with the lowest final SSE
		for (int run = 0; run < num_of_runs_; ++run) {
			// Initialize centroids using random partition method
			std::vector<Point> centroids = selectRandomPartitionCentroids();
			std::vector<int> assignments(num_of_points_);

			// Run k-means until convergence or max iterations
			double previous_sse = std::numeric_limits<double>::max();
			double current_sse = 0.0;

			for (int iteration = 0; iteration < max_iterations_; ++iteration) {
				// Step 3: Assign each point to its nearest centroid
				for (int i = 0; i < num_of_points_; ++i) {
					double min_dist = std::numeric_limits<double>::max();
					int nearest = 0;
					// Find the centroid closest to point i (squared Euclidean distance)
					for (int ki = 0; ki < k; ++ki) {
						double sq_dist = 0.0;
						for (int d = 0; d < dimensionality_; ++d) {
							double diff = dataset_[i].getVal(d) - centroids[ki].getVal(d);
							sq_dist += diff * diff;
						}
						if (sq_dist < min_dist) {
							min_dist = sq_dist;
							nearest = ki;
						}
					}
					assignments[i] = nearest;
				}

				// Step 4: Recompute centroids as the mean of their assigned points
				std::vector<std::vector<double>> sums(k, std::vector<double>(dimensionality_, 0.0));
				std::vector<int> cluster_sizes(k, 0);
				for (int i = 0; i < num_of_points_; ++i) {
					// Accumulate sums and counts per cluster
					int ci = assignments[i];
					cluster_sizes[ci]++;
					for (int d = 0; d < dimensionality_; ++d) {
						sums[ci][d] += dataset_[i].getVal(d);
					}
				}
				std::vector<Point> new_centroids;
				for (int ki = 0; ki < k; ++ki) {
					Point new_center;
					for (int d = 0; d < dimensionality_; ++d) {
						// If cluster is empty, retain old centroid position
						if (cluster_sizes[ki] > 0) {
							new_center.addDimension(sums[ki][d] / cluster_sizes[ki]);
						} else {
							new_center.addDimension(centroids[ki].getVal(d));
						}
					}
					new_centroids.push_back(new_center);
				}
				centroids = new_centroids;

				// Compute SSE for this iteration
				current_sse = 0.0;
				for (int i = 0; i < num_of_points_; ++i) {
					int ci = assignments[i];
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - centroids[ci].getVal(d);
						current_sse += diff * diff;
					}
				}

				// Step 5: Check convergence
				// Stop if improvement is below threshold
				if (previous_sse != std::numeric_limits<double>::max()) {
					if (current_sse == previous_sse) {
						break; // No improvement (fully converged)
					}
					double relative_improvement = (previous_sse - current_sse) / previous_sse;
					if (relative_improvement < convergence_threshold_) {
						break; // Below convergence threshold
					}
				}
				previous_sse = current_sse;
			}

			// Step 2b: Keep this run's result if it produced the lowest SSE so far
			if (current_sse < best_sse) {
				best_sse = current_sse;
				best_centroids = centroids;
				best_assignments = assignments;
			}
		}

		// Step 2c: Compute CH and SW on the best run's partition
		double ch = computeCH(best_centroids, best_assignments, best_sse, k);
		double sw = computeSW(best_assignments, k);

		// Store results for this K
		k_values.push_back(k);
		ch_values.push_back(ch);
		sw_values.push_back(sw);

		// Write this K's results to the CSV file
		if (csv_file.is_open()) {
			csv_file << k << ","
			         << std::fixed << std::setprecision(4) << ch << ","
			         << std::fixed << std::setprecision(4) << sw << "\n";
		}

		std::cout << "K=" << k
		          << "  CH=" << std::fixed << std::setprecision(4) << ch
		          << "  SW=" << std::fixed << std::setprecision(4) << sw << "\n\n";
	}

	// Step 3: Find the K that maximizes CH and the K that maximizes SW
	int best_k_ch = k_min;
	int best_k_sw = k_min;
	double max_ch = ch_values[0];
	double max_sw = sw_values[0];

	for (int idx = 1; idx < static_cast<int>(k_values.size()); ++idx) {
		// CH is a maximization index so it picks the K with the highest CH
		if (ch_values[idx] > max_ch) {
			max_ch = ch_values[idx];
			best_k_ch = k_values[idx];
		}
		// SW is a maximization index so it picks the K with the highest SW
		if (sw_values[idx] > max_sw) {
			max_sw = sw_values[idx];
			best_k_sw = k_values[idx];
		}
	}

	// Step 4: Print the final results table to stdout
	std::cout << "Final Results:\n";
	std::cout << std::setw(5) << "K"
	          << std::setw(15) << "CH"
	          << std::setw(15) << "SW" << "\n";
	std::cout << std::string(35, '-') << "\n";
	for (int idx = 0; idx < static_cast<int>(k_values.size()); ++idx) {
		// Mark the estimated K for each index with an asterisk
		std::string ch_marker = (k_values[idx] == best_k_ch) ? " *" : "";
		std::string sw_marker = (k_values[idx] == best_k_sw) ? " *" : "";
		std::cout << std::setw(5) << k_values[idx]
		          << std::setw(15) << std::fixed << std::setprecision(4) << ch_values[idx] << ch_marker
		          << std::setw(15) << std::fixed << std::setprecision(4) << sw_values[idx] << sw_marker << "\n";
	}
	std::cout << "\nEstimated K (CH): " << best_k_ch << "  (CH=" << max_ch << ")\n";
	std::cout << "Estimated K (SW): " << best_k_sw << "  (SW=" << max_sw << ")\n";

	// Append summary to the CSV file
	if (csv_file.is_open()) {
		csv_file << "\nBest K (CH)," << best_k_ch << "\n";
		csv_file << "Best K (SW)," << best_k_sw;
		csv_file.close();
		std::cout << "\nResults written to " << csv_path << "\n";
	}
}

// Counts the four point-pair categories (TP, FN, FP, TN) shared by Rand and Jaccard.
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", 2nd ed.
//            Section 17.1 (External Measures), pg. 436-437
//            https://dataminingbook.info/book_html/chap17/book.html
// TP: pairs together in both C and T
// FN: pairs together in T only
// FP: pairs together in C only
// TN: pairs apart in both
static void countPairCategories(
	const std::vector<int>& assignments,
	const std::vector<int>& true_labels,
	long long& tp,
	long long& fn,
	long long& fp,
	long long& tn)
{
	tp = 0;
	fn = 0;
	fp = 0;
	tn = 0;

	const int n = static_cast<int>(assignments.size());
	// Iterate over each unordered pair (i, j) with i < j so each pair is counted once
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			bool same_c = (assignments[i] == assignments[j]);
			bool same_t = (true_labels[i] == true_labels[j]);
			if (same_c && same_t) {
				++tp;
			} else if (!same_c && same_t) {
				++fn;
			} else if (same_c && !same_t) {
				++fp;
			} else {
				++tn;
			}
		}
	}
}

// Computes the Rand statistic for the given partition vs. true_labels_.
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", 2nd ed.
//            Section 17.1 (External Measures), pg. 437
//            Eq. 17.21 Rand Statistic
//            https://dataminingbook.info/book_html/chap17/book.html
// Rand = (TP + TN) / N_pairs, where N_pairs = n(n-1)/2
// Symmetric index; a perfect clustering gives Rand = 1
double Kmeans::computeRand(const std::vector<int>& assignments) {
	long long tp = 0, fn = 0, fp = 0, tn = 0;
	countPairCategories(assignments, true_labels_, tp, fn, fp, tn);

	// Total pairs = TP + FN + FP + TN = n(n-1)/2
	long long total_pairs = tp + fn + fp + tn;
	if (total_pairs == 0) {
		// Only possible when N < 2 (no pairs exist)
		return 1.0;
	}
	return static_cast<double>(tp + tn) / static_cast<double>(total_pairs);
}

// Computes the Jaccard coefficient for the given partition vs. true_labels_.
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", 2nd ed.
//            Section 17.1 (External Measures), pg. 436
//            Eq. 17.20 Jaccard Coefficient
//            https://dataminingbook.info/book_html/chap17/book.html
// Jaccard = TP / (TP + FN + FP)
// Ignores TN, so it is asymmetric in true positives vs. true negatives
double Kmeans::computeJaccard(const std::vector<int>& assignments) {
	long long tp = 0, fn = 0, fp = 0, tn = 0;
	countPairCategories(assignments, true_labels_, tp, fn, fp, tn);

	long long denom = tp + fn + fp;
	if (denom == 0) {
		// No together-pairs in either partition (all singletons in both C and T)
		return 1.0;
	}
	return static_cast<double>(tp) / static_cast<double>(denom);
}

// Runs k-means R times at K = num_true_clusters_ using random-selection
// initialization (Phase 1 method), computes Rand and Jaccard against the
// true labels on each run, and reports the best value of each index.
// Reference: Zaki & Meira Jr., "Data Mining and Machine Learning", Chapter 17
//            https://dataminingbook.info/book_html/chap17/book.html
void Kmeans::runExternalValidation() {
	// K for Phase 5 is the true cluster count read from the data file header
	num_clusters_ = num_true_clusters_;
	const int k = num_clusters_;

	std::cout << "Running external validation: K=" << k
	          << ", N=" << num_of_points_
	          << ", R=" << num_of_runs_ << "\n\n";

	// Prepare per-dataset CSV in ../output_phase5/.
	std::filesystem::create_directories("../output_phase5");
	std::string base_name = file_name_;
	auto dot_pos = base_name.rfind('.');
	if (dot_pos != std::string::npos) {
		base_name = base_name.substr(0, dot_pos);
	}
	std::string csv_path = "../output_phase5/results_" + base_name + ".csv";
	std::ofstream csv_file(csv_path);
	if (!csv_file.is_open()) {
		std::cerr << "Error: Could not create output file: " << csv_path << "\n";
	}
	if (csv_file.is_open()) {
		csv_file << "Run,Rand,Jaccard\n";
	}

	// Track best Rand and best Jaccard independently across all R runs.
	double best_rand = -1.0;
	double best_jaccard = -1.0;
	int best_rand_run = 0;
	int best_jaccard_run = 0;

	for (int run = 0; run < num_of_runs_; ++run) {
		// Step 1: random-selection initialization (Phase 1 method).
		std::vector<Point> centroids = selectRandomSelectionCentroids();
		std::vector<int> assignments(num_of_points_);

		// Step 2: k-means assign/update loop with the convergence check from Phase 4
		double previous_sse = std::numeric_limits<double>::max();
		double current_sse = 0.0;

		for (int iteration = 0; iteration < max_iterations_; ++iteration) {
			// Assignment step: nearest centroid by squared Euclidean distance.
			for (int i = 0; i < num_of_points_; ++i) {
				double min_dist = std::numeric_limits<double>::max();
				int nearest = 0;
				for (int ki = 0; ki < k; ++ki) {
					double sq_dist = 0.0;
					for (int d = 0; d < dimensionality_; ++d) {
						double diff = dataset_[i].getVal(d) - centroids[ki].getVal(d);
						sq_dist += diff * diff;
					}
					if (sq_dist < min_dist) {
						min_dist = sq_dist;
						nearest = ki;
					}
				}
				assignments[i] = nearest;
			}

			// Update step: recompute each centroid as the mean of its points.
			std::vector<std::vector<double>> sums(k, std::vector<double>(dimensionality_, 0.0));
			std::vector<int> cluster_sizes(k, 0);
			for (int i = 0; i < num_of_points_; ++i) {
				int ci = assignments[i];
				cluster_sizes[ci]++;
				for (int d = 0; d < dimensionality_; ++d) {
					sums[ci][d] += dataset_[i].getVal(d);
				}
			}
			std::vector<Point> new_centroids;
			for (int ki = 0; ki < k; ++ki) {
				Point new_center;
				for (int d = 0; d < dimensionality_; ++d) {
					if (cluster_sizes[ki] > 0) {
						new_center.addDimension(sums[ki][d] / cluster_sizes[ki]);
					} else {
						// Empty cluster: keep the previous centroid position.
						new_center.addDimension(centroids[ki].getVal(d));
					}
				}
				new_centroids.push_back(new_center);
			}
			centroids = new_centroids;

			// SSE for this iteration (drives the convergence check only).
			current_sse = 0.0;
			for (int i = 0; i < num_of_points_; ++i) {
				int ci = assignments[i];
				for (int d = 0; d < dimensionality_; ++d) {
					double diff = dataset_[i].getVal(d) - centroids[ci].getVal(d);
					current_sse += diff * diff;
				}
			}

			// Convergence: no change, or relative improvement below threshold.
			if (previous_sse != std::numeric_limits<double>::max()) {
				if (current_sse == previous_sse) {
					break;
				}
				double relative_improvement = (previous_sse - current_sse) / previous_sse;
				if (relative_improvement < convergence_threshold_) {
					break;
				}
			}
			previous_sse = current_sse;
		}

		// Step 3: evaluate the external indices on this run's partition.
		double rand_val = computeRand(assignments);
		double jaccard_val = computeJaccard(assignments);

		if (csv_file.is_open()) {
			csv_file << (run + 1) << ","
			         << std::fixed << std::setprecision(4) << rand_val << ","
			         << std::fixed << std::setprecision(4) << jaccard_val << "\n";
		}

		// Track the best of each index independently.
		if (rand_val > best_rand) {
			best_rand = rand_val;
			best_rand_run = run + 1;
		}
		if (jaccard_val > best_jaccard) {
			best_jaccard = jaccard_val;
			best_jaccard_run = run + 1;
		}

		std::cout << "Run " << (run + 1)
		          << "  Rand=" << std::fixed << std::setprecision(4) << rand_val
		          << "  Jaccard=" << std::fixed << std::setprecision(4) << jaccard_val << "\n";
	}

	// Final per-dataset summary.
	std::cout << "\nBest Rand: "    << std::fixed << std::setprecision(4) << best_rand
	          << " (Run " << best_rand_run << ")\n";
	std::cout << "Best Jaccard: "   << std::fixed << std::setprecision(4) << best_jaccard
	          << " (Run " << best_jaccard_run << ")\n";

	if (csv_file.is_open()) {
		csv_file << "\nBest Rand," << std::fixed << std::setprecision(4) << best_rand << "\n";
		csv_file << "Best Jaccard," << std::fixed << std::setprecision(4) << best_jaccard << "\n";
		csv_file.close();
		std::cout << "Results written to " << csv_path << "\n";
	}

	// Aggregate one row per dataset into ../output_phase5/phase5_results.csv
	std::filesystem::create_directories("../output_phase5");
	std::string agg_path = "../output_phase5/phase5_results.csv";
	bool agg_exists = false;
	{
		std::ifstream check_file(agg_path);
		if (check_file.good()) {
			agg_exists = true;
		}
	}
	std::ofstream agg_file(agg_path, std::ios::app);
	if (agg_file.is_open()) {
		if (!agg_exists) {
			agg_file << "Dataset,Best Rand,Best Jaccard\n";
		}
		agg_file << base_name << ","
		         << std::fixed << std::setprecision(4) << best_rand << ","
		         << std::fixed << std::setprecision(4) << best_jaccard << "\n";
		agg_file.close();
		std::cout << "Aggregated row appended to " << agg_path << "\n";
	} else {
		std::cerr << "Error: Could not append to " << agg_path << "\n";
	}
}