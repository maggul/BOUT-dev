
#ifndef BOUT_GVALUES_HXX
#define BOUT_GVALUES_HXX

#include "bout/metric_tensor.hxx"

using FieldMetric = MetricTensor::FieldMetric;

/// `GValues` needs renaming, when we know what the name should be
class GValues {

public:
  GValues(FieldMetric G1, FieldMetric G2, FieldMetric G3);

  explicit GValues(const Coordinates& coordinates);

  const FieldMetric& G1() const { return G1_m; }
  const FieldMetric& G2() const { return G2_m; }
  const FieldMetric& G3() const { return G3_m; }

  void setG1(const FieldMetric& G1) { G1_m = G1; }
  void setG2(const FieldMetric& G2) { G2_m = G2; }
  void setG3(const FieldMetric& G3) { G3_m = G3; }

  void communicate(Mesh* mesh);

private:
  FieldMetric G1_m, G2_m, G3_m;
};

#endif //BOUT_GVALUES_HXX
