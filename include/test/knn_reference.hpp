#ifndef KNN_REFERENCE__H__
#define KNN_REFERENCE__H__

#include "matrix.hpp"

namespace knn
{
namespace test
{
	Matrix knn_reference(const Matrix& V, const Matrix& Q, int k);

} // namespace test
} // namespace reference

#endif
