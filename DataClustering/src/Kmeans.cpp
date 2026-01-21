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
	// Seed rng
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

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
		int randomIndex = std::rand() % num_of_points_;

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
