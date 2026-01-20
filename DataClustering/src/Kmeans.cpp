#include "../include/Kmeans.h"

// Constructor implementation
Kmeans::Kmeans(
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
bool Kmeans::readData() {
	std::ifstream inputFile(fileName);
	
	// If file doesn't open, try with datasets/ prefix
	if (!inputFile.is_open()) {
		std::string pathWithPrefix = "../datasets/" + fileName;
		inputFile.open(pathWithPrefix);
		
		// If that also doesn't work, report error
		if (!inputFile.is_open()) {
			std::cerr << "Error opening file: " << fileName << " or " << pathWithPrefix << std::endl;
			return false;
		}
	}

	// Read number of data points and dimensionality from the first line and if valid, store them
	if (!(inputFile >> numOfPoints >> dimensionality)) {
		std::cerr << "Error reading number of points and dimensionality." << std::endl;
		return false;
	}

	// Print number of points and dimensionality
	// std::cout << "Number of Points: " << numOfPoints << ", Dimensionality: " << dimensionality << std::endl;

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

// Print dataset
void Kmeans::printData() const {
	for (const auto& point : dataset) {
		point.print();
	}
}

// Selects K centers uniformly at random from the existing data points
void Kmeans::selectAndPrintCenters() {
	// Seed RNG with random_device for better randomness
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	std::vector<int> selectedIndices;

	// Loop until we have found K unique centers
	while (selectedIndices.size() < (size_t)numClusters) {
		// Select index uniformly at random
		int randomIndex = std::rand() % numOfPoints;

		// Ensure we don't pick the same point twice
		if (!contains(selectedIndices, randomIndex)) {
			selectedIndices.push_back(randomIndex);

			// Print the point immediately
			dataset[randomIndex].print();
		}
	}
}
