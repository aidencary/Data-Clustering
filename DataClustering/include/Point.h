#pragma once

#include <vector>
#include <iostream>

class Point {
private:
	std::vector<double> dimensions;

public:
	// Constructor
	Point();

	// Add a dimension value to the point
	void addDimension(double val);

	// Get the value at a specific dimension
	double getVal(int index) const;

	// Print the point
	void print() const;

};