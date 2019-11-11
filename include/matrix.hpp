#ifndef MATRIX__H__
#define MATRIX__H__

#include <vector>

namespace knn
{
class Matrix
{
public:
	Matrix(int cols, int rows);
	float at(int x, int y) const;
	void set(int x, int y, float val);
	float *data();
	int num_cols() const;
	int num_rows() const;
private:
	std::vector<float>_flat;
	int _cols;
	int _rows;
};
} // namespace knn

void print_matrix(knn::Matrix &a);


#endif
