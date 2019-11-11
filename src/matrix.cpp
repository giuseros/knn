#include "matrix.hpp"
#include <iostream>

namespace knn
{
Matrix::Matrix(int cols, int rows): _flat(cols*rows), _cols(cols), _rows(rows)
{
}

float Matrix::at(int x, int y) const
{
		return _flat[x + _cols*y];
}

void Matrix::set(int x, int y, float val)
{
	_flat[x + _cols*y] = val;
}

float* Matrix::data()
{
	return _flat.data();
}

int Matrix::num_cols() const
{
	return _cols;
}

int Matrix::num_rows() const
{
	return _rows;
}
} // namespace knn


void print_matrix(knn::Matrix &a)
{
	for (int j = 0; j<a.num_rows(); j++)
	{
		for (int i = 0; i < a.num_cols(); i++)
		{

			std::cout<<a.at(i, j) <<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
}
