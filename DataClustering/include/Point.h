#pragma once

#include <vector>
#include <iostream>

class Point {
private:
	std::vector<double> dimensions;

public:
	Point();
	void addDimension(double val);
	double getVal(int index) const;
	void print() const;
	void print(std::ostream& os) const;
};