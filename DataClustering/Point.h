#pragma once

#include <vector>
#include <iostream>

class Point {
private:
	std::vector<double> dimensions;

public:
	// Constructor
	Point() {}

	// Add a dimension value to the point
	void addDimension(double val) {
		dimensions.push_back(val);
	}

	// Get the value at a specific dimension
	double getVal(int index) const {
		return dimensions[index];
	}

	// Print the point
	void print() const {
		for (size_t i = 0; i < dimensions.size(); ++i) {
			
			if (i > 0) {
				// Using the '?' ternary operator to avoid trailing space
				std::cout << dimensions[i] << (i < dimensions.size() - 1 ? " " : "");
			}
			std::cout << std::endl;
		}
	}


};