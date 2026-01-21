#pragma once

#include <iostream>
#include <vector>

// Represents a point in multi-dimensional space.
class Point {
private:
	std::vector<double> dimensions_;

public:
	// Constructor
	Point();
	// Adds a dimension value to this point.
	void addDimension(double val);
	// Returns the value at the specified dimension index.
	double getVal(int index) const;
	// Prints the point to standard output.
	void print() const;
	// Prints the point to the specified output stream.
	void print(std::ostream& os) const;
};