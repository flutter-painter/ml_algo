import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/loss_function/loss_function_factory.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

void main() {
  Float32x4Vector w;
  Float32x4Vector x;
  double y;

  group('Cross entropy loss', () {
    test('should return proper value', () {
      LossFunction crossEntropyLoss = LossFunctionFactory.CrossEntropy();

      w = new Float32x4Vector.from([0.2, 0.3, 0.4, 0.5]);
      x = new Float32x4Vector.from([1.0, 2.0, 3.0, 4.0]);
      y = 1.0;

      expect(crossEntropyLoss.loss(w.dot(x), y), null);
    }, skip: true);

    test('should return proper value', () {
      LossFunction logisticLoss = LossFunctionFactory.Logistic();

      w = new Float32x4Vector.from([1.2, 1.3, 1.4, 1.5]);
      x = new Float32x4Vector.from([1.0, 2.0, 3.0, 4.0]);
      y = 1.0;
    }, skip: true);
  });
}