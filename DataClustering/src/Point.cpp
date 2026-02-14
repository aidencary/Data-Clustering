#include "../include/Point.h"
#include <iomanip>

// Constructor implementation
Point::Point() {}

// Add a dimension value to the point
void Point::addDimension(double val) {
	dimensions_.push_back(val);
}

// Get the value at a specific dimension
double Point::getVal(int index) const {
	return dimensions_[index];
}

// Print the point
void Point::print() const {
	std::cout << std::fixed << std::setprecision(15);
	for (size_t i = 0; i < dimensions_.size(); ++i) {
		// Print the value
		std::cout << dimensions_[i];
		// Print a space only if it's not the last dimension
		if (i < dimensions_.size() - 1) {
			std::cout << " ";
		}
	}
	std::cout << std::endl;
}
