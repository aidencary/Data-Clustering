#include "../include/Point.h"

Point::Point() {}

void Point::addDimension(double val) {
	dimensions_.push_back(val);
}

double Point::getVal(int index) const {
	return dimensions_[index];
}

void Point::print() const {
	for (size_t i = 0; i < dimensions_.size(); ++i) {
		std::cout << dimensions_[i];
		if (i < dimensions_.size() - 1) {
			std::cout << " ";
		}
	}
	std::cout << std::endl;
}

void Point::print(std::ostream& os) const {
	for (size_t i = 0; i < dimensions_.size(); ++i) {
		os << dimensions_[i];
		if (i < dimensions_.size() - 1) {
			os << " ";
		}
	}
	os << std::endl;
}
