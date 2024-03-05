
#include "bout/metricTensor.hxx"
#include "bout/output.hxx"
#include <utility>

MetricTensor::MetricTensor(FieldMetric g11, FieldMetric g22, FieldMetric g33,
                           FieldMetric g12, FieldMetric g13, FieldMetric g23)
    : g11_(std::move(g11)), g22_(std::move(g22)), g33_(std::move(g33)),
      g12_(std::move(g12)), g13_(std::move(g13)), g23_(std::move(g23)) {}

MetricTensor::MetricTensor(const BoutReal g11, const BoutReal g22, const BoutReal g33,
                           const BoutReal g12, const BoutReal g13, const BoutReal g23,
                           Mesh* mesh)
    : g11_(g11, mesh), g22_(g22, mesh), g33_(g33, mesh), g12_(g12, mesh), g13_(g13, mesh),
      g23_(g23, mesh) {}

void MetricTensor::check(int ystart) {
  // Diagonal metric components should be finite
  bout::checkFinite(g11_, "g11", "RGN_NOCORNERS");
  bout::checkFinite(g22_, "g22", "RGN_NOCORNERS");
  bout::checkFinite(g33_, "g33", "RGN_NOCORNERS");
  if (g11_.hasParallelSlices() && &g11_.ynext(1) != &g11_) {
    for (int dy = 1; dy <= ystart; ++dy) {
      for (const auto sign : {1, -1}) {
        bout::checkFinite(g11_.ynext(sign * dy), "g11.ynext",
                          fmt::format("RGN_YPAR_{:+d}", sign * dy));
        bout::checkFinite(g22_.ynext(sign * dy), "g22.ynext",
                          fmt::format("RGN_YPAR_{:+d}", sign * dy));
        bout::checkFinite(g33_.ynext(sign * dy), "g33.ynext",
                          fmt::format("RGN_YPAR_{:+d}", sign * dy));
      }
    }
  }
  // Diagonal metric components should be positive
  bout::checkPositive(g11_, "g11", "RGN_NOCORNERS");
  bout::checkPositive(g22_, "g22", "RGN_NOCORNERS");
  bout::checkPositive(g33_, "g33", "RGN_NOCORNERS");
  if (g11_.hasParallelSlices() && &g11_.ynext(1) != &g11_) {
    for (int dy = 1; dy <= ystart; ++dy) {
      for (const auto sign : {1, -1}) {
        bout::checkPositive(g11_.ynext(sign * dy), "g11.ynext",
                            fmt::format("RGN_YPAR_{:+d}", sign * dy));
        bout::checkPositive(g22_.ynext(sign * dy), "g22.ynext",
                            fmt::format("RGN_YPAR_{:+d}", sign * dy));
        bout::checkPositive(g33_.ynext(sign * dy), "g33.ynext",
                            fmt::format("RGN_YPAR_{:+d}", sign * dy));
      }
    }
  }

  // Off-diagonal metric components should be finite
  bout::checkFinite(g12_, "g12", "RGN_NOCORNERS");
  bout::checkFinite(g13_, "g13", "RGN_NOCORNERS");
  bout::checkFinite(g23_, "g23", "RGN_NOCORNERS");
  if (g23_.hasParallelSlices() && &g23_.ynext(1) != &g23_) {
    for (int dy = 1; dy <= ystart; ++dy) {
      for (const auto sign : {1, -1}) {
        bout::checkFinite(g12_.ynext(sign * dy), "g12.ynext",
                          fmt::format("RGN_YPAR_{:+d}", sign * dy));
        bout::checkFinite(g13_.ynext(sign * dy), "g13.ynext",
                          fmt::format("RGN_YPAR_{:+d}", sign * dy));
        bout::checkFinite(g23_.ynext(sign * dy), "g23.ynext",
                          fmt::format("RGN_YPAR_{:+d}", sign * dy));
      }
    }
  }
}

MetricTensor MetricTensor::inverse(const std::string& region) {

  TRACE("MetricTensor::inverse");

  // Perform inversion of g{ij} to get g^{ij}, or vice versa
  // NOTE: Currently this bit assumes that metric terms are Field2D objects

  auto a = Matrix<BoutReal>(3, 3);

  FieldMetric g_11 = emptyFrom(g11_);
  FieldMetric g_22 = emptyFrom(g22_);
  FieldMetric g_33 = emptyFrom(g33_);
  FieldMetric g_12 = emptyFrom(g12_);
  FieldMetric g_13 = emptyFrom(g13_);
  FieldMetric g_23 = emptyFrom(g23_);

  BOUT_FOR_SERIAL(i, g11_.getRegion(region)) {
    a(0, 0) = g11_[i];
    a(1, 1) = g22_[i];
    a(2, 2) = g33_[i];

    a(0, 1) = a(1, 0) = g12_[i];
    a(1, 2) = a(2, 1) = g23_[i];
    a(0, 2) = a(2, 0) = g13_[i];

    if (invert3x3(a)) {
      const auto error_message = "\tERROR: metric tensor is singular at ({:d}, {:d})\n";
      output_error.write(error_message, i.x(), i.y());
      throw BoutException(error_message);
    }

    g_11[i] = a(0, 0);
    g_22[i] = a(1, 1);
    g_33[i] = a(2, 2);
    g_12[i] = a(0, 1);
    g_13[i] = a(0, 2);
    g_23[i] = a(1, 2);
  }
  
  BoutReal maxerr;
  maxerr = BOUTMAX(max(abs((g_11 * g_11 + g_12 * g_12 + g_13 * g_13) - 1)),
                   max(abs((g_12 * g_12 + g_22 * g_22 + g_23 * g_23) - 1)),
                   max(abs((g_13 * g_13 + g_23 * g_23 + g_33 * g_33) - 1)));

  output_info.write("\tMaximum error in diagonal inversion is {:e}\n", maxerr);

  maxerr = BOUTMAX(max(abs(g_11 * g_12 + g_12 * g_22 + g_13 * g_23)),
                   max(abs(g_11 * g_13 + g_12 * g_23 + g_13 * g_33)),
                   max(abs(g_12 * g_13 + g_22 * g_23 + g_23 * g_33)));

  output_info.write("\tMaximum error in off-diagonal inversion is {:e}\n", maxerr);
  
  auto other_representation = MetricTensor(g_11, g_22, g_33, g_12, g_13, g_23);
  const auto location = g11_.getLocation();
  other_representation.setLocation(location);
  return other_representation;
}

void MetricTensor::map(
    const std::function<const FieldMetric(const FieldMetric)>& function) {

  const MetricTensor updated_metric_tensor = applyToComponents(function);

  setMetricTensor(MetricTensor(updated_metric_tensor.g11_, updated_metric_tensor.g22_,
                               updated_metric_tensor.g33_, updated_metric_tensor.g12_,
                               updated_metric_tensor.g13_, updated_metric_tensor.g23_));
}

MetricTensor MetricTensor::applyToComponents(
    const std::function<const FieldMetric(const FieldMetric)>& function) const {

  const auto components_in = std::vector<FieldMetric>{g11_, g22_, g33_, g12_, g13_, g23_};

  FieldMetric components_out[6];

  std::transform(components_in.begin(), components_in.end(), components_out, function);
  auto [g_11, g_22, g_33, g_12, g_13, g_23] = components_out;

  return MetricTensor(g_11, g_22, g_33, g_12, g_13, g_23);
}
