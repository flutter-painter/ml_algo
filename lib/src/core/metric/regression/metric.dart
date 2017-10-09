part of 'package:dart_ml/src/core/interface.dart';

abstract class RegressionMetric implements Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
}
