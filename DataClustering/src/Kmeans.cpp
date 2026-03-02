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
	std::ifstream input_file(file_name_);
	
	if (!input_file.is_open()) {
		std::string path_with_prefix = "../datasets/" + file_name_;
		input_file.open(path_with_prefix);
		
		if (!input_file.is_open()) {
			std::cerr << "Error opening file: " << file_name_ << " or " << path_with_prefix << std::endl;
			return false;
		}
	}

	if (!(input_file >> num_of_points_ >> dimensionality_)) {
		std::cerr << "Error reading number of points and dimensionality." << std::endl;
		return false;
	}

	for (int i = 0; i < num_of_points_; ++i) {
		Point point;
		for (int d = 0; d < dimensionality_; ++d) {
			double val;
			if (!(input_file >> val)) {
				std::cerr << "Error reading data point values." << std::endl;
				return false;
			}
			point.addDimension(val);
		}
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

std::vector<Point> Kmeans::selectCentroids() {
	// Step 1 of K-means Algorithm: Select K points as initial centroids
	// Centroids are selected uniformly at random from the dataset using C++11 <random> library
	std::random_device rd; // seed source for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
	std::uniform_int_distribution<> dis(0, num_of_points_ - 1);

	std::vector<int> selected_indices;
	std::vector<Point> centers;

	// Loop until we have found K unique centers
	while (selected_indices.size() < (size_t)num_clusters_) {
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

bool Kmeans::checkIrisBezdekOptimum(double currentSSE) const {
	// Test condition for the Iris Bezdek dataset
	// Check if SSE is lower than the known global optimum
	if (currentSSE < 78.8514) {
		std::cerr << "Lower than global optimum for Iris Bezdek dataset, something is wrong." << std::endl;
	}
	
	// Perfect Clustering Check for Iris Bezdek dataset
	double rounded_sse = std::round(currentSSE * 10000.0) / 10000.0;
	if (rounded_sse == 78.8514) {
		std::cout << "----\nGlobal opt (78.8514) reached\n----" << std::endl;
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
		std::cerr << "Error: Could not create output file: " << output_file_name << std::endl;
	}

	// Track best run across all executions
	double best_sse = std::numeric_limits<double>::max();
	int best_run = 0;
	
	// Run the K-means algorithm for the specified number of runs
	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << std::endl;
		std::cout << "-----" << std::endl;
		if (output_file.is_open()) {
			output_file << "Run " << (run + 1) << std::endl;
			output_file << "-----" << std::endl;
		}

		// Step 1: Select K points as initial centroids
		std::vector<Point> centroids = selectCentroids();
		
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
			
			std::cout << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << std::endl;
			if (output_file.is_open()) {
				output_file << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << std::endl;
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
		std::cout << std::endl;
		if (output_file.is_open()) {
			output_file << std::endl;
		}
		
		// Track if this run achieved the best SSE
		if (current_sse < best_sse) {
			best_sse = current_sse;
			best_run = run + 1;
		}
	}
	
	// Display best run after all runs complete
	std::cout << "\nBest Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse << std::endl;
	if (output_file.is_open()) {
		output_file << "\nBest Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse;
		output_file.close();
	}

	/* Write best run to best_runs.txt file
	std::ofstream best_runs_file("../output/best_runs.txt", std::ios::app);
	if (best_runs_file.is_open()) {
		best_runs_file << file_name_ << ": Best Run = " << best_run << ", SSE = " << std::fixed << std::setprecision(4) << best_sse << std::endl;
		best_runs_file.close();
	} else {
		std::cerr << "Error: Could not open best_runs.txt file" << std::endl;
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
		std::cerr << "Error: Could not create output file: " << output_file_name << std::endl;
	}

	// Track best run across all executions
	double best_sse = std::numeric_limits<double>::max();
	int best_run = 0;
	
	// Run the K-means algorithm for the specified number of runs
	for (int run = 0; run < num_of_runs_; ++run) {
		std::cout << "Run " << (run + 1) << std::endl;
		std::cout << "-----" << std::endl;
		if (output_file.is_open()) {
			output_file << "Run " << (run + 1) << std::endl;
			output_file << "-----" << std::endl;
		}

		// Step 1: Select K points as initial centroids
		std::vector<Point> centroids = selectCentroids();
		
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
				std::cout << "Handling " << singleton_clusters.size() << " singleton cluster(s)..." << std::endl;
				if (output_file.is_open()) {
					output_file << "Handling " << singleton_clusters.size() << " singleton cluster(s)..." << std::endl;
				}
				*/
				

				for (int s = 0; s < (int)singleton_clusters.size(); ++s) {
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
			
			std::cout << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << std::endl;
			if (output_file.is_open()) {
				output_file << "Iteration " << (iteration + 1) << ": SSE = " << std::fixed << std::setprecision(4) << current_sse << std::endl;
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
		std::cout << std::endl;
		if (output_file.is_open()) {
			output_file << std::endl;
		}
		
		// Track if this run achieved the best SSE
		if (current_sse < best_sse) {
			best_sse = current_sse;
			best_run = run + 1;
		}
	}
	
	// Display best run after all runs complete
	std::cout << "Best Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse << std::endl;
	if (output_file.is_open()) {
		output_file << "Best Run: " << best_run << ": SSE = " << std::fixed << std::setprecision(4) << best_sse;
		output_file.close();
	}

	/*
	// Write best run to best_runs.txt file
	std::ofstream best_runs_file("../output/best_runs.txt", std::ios::app);
	if (best_runs_file.is_open()) {
		best_runs_file << file_name_ << ": Best Run = " << best_run << ", SSE = " << std::fixed << std::setprecision(4) << best_sse << std::endl;
		best_runs_file.close();
	} else {
		std::cerr << "Error: Could not open best_runs.txt file" << std::endl;
	}
	*/

}