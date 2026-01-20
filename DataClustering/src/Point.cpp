#include "../include/Point.h"

// Constructor implementation
Point::Point() {}

// Add a dimension value to the point
void Point::addDimension(double val) {
	dimensions.push_back(val);
}

// Get the value at a specific dimension
double Point::getVal(int index) const {
	return dimensions[index];
}

// Print the point
void Point::print() const {
	for (size_t i = 0; i < dimensions.size(); ++i) {
		// Print the value
		std::cout << dimensions[i];
		// Print a space only if it's not the last dimension
		if (i < dimensions.size() - 1) {
			std::cout << " ";
		}
	}
	std::cout << std::endl;
}
