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

std::vector<Point> Kmeans::selectRandomCentroids() {
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
		// Get the cluster assignment for this point
		int cluster = assignments[i];
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
		// The fallback point is selected once per empty cluster so all dimensions
		// come from the same point (avoids a "Frankenstein" centroid).
		int fallback_point = -1;
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
		std::vector<Point> centroids = selectRandomCentroids();
		
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
		std::vector<Point> centroids = selectRandomCentroids();
		
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
	* Number of Iterations: This is the number of times k-means iterates until reaching
	* convergence when initialized by a particular initialization method. 
	* It is an efficiency measure independent of programming language, implementation style, compiler, andCPU architecture. 
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
			centroids = selectRandomCentroids();
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
	std::filesystem::create_directories("../output_sheets");
	std::string csv_file_path = "../output_sheets/phase3_results.csv";
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