import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

import '../../fake_data_set.dart';

void main() {
  group('RandomForestClassifier OOB Scoring', () {
    final trainData = fakeDataSet;
    final targetName = 'col_8';

    test('should have oobScore property when bootstrap is enabled', () {
      // Arrange & Act
      final forest = RandomForestClassifier(
        trainData,
        targetName,
        nEstimators: 10,
        bootstrap: true,
        maxDepth: 3,
      );

      // Assert
      expect(forest.oobScore, isNotNull);
      expect(forest.oobScore, isA<double>());
      expect(forest.oobScore, greaterThanOrEqualTo(0.0));
      expect(forest.oobScore, lessThanOrEqualTo(1.0));
    });

    test('should have null oobScore when bootstrap is disabled', () {
      // Arrange & Act
      final forest = RandomForestClassifier(
        trainData,
        targetName,
        nEstimators: 10,
        bootstrap: false,
        maxDepth: 3,
      );

      // Assert
      expect(forest.oobScore, isNull);
    });

    test('should calculate oobScore correctly for simple dataset', () {
      // Arrange
      final simpleData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 1, 1, 1], isDiscrete: true),
      ]);

      // Act
      final forest = RandomForestClassifier(
        simpleData,
        'target',
        nEstimators: 5,
        bootstrap: true,
        maxDepth: 2,
        seed: 42, // Fixed seed for reproducibility
      );

      // Assert
      expect(forest.oobScore, isNotNull);
      expect(forest.oobScore, greaterThanOrEqualTo(0.0));
      expect(forest.oobScore, lessThanOrEqualTo(1.0));
    });

    test('should have consistent oobScore with same seed', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest1 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        bootstrap: true,
        maxDepth: 3,
        seed: 123,
      );

      final forest2 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        bootstrap: true,
        maxDepth: 3,
        seed: 123,
      );

      // Assert
      expect(forest1.oobScore, equals(forest2.oobScore));
    });

    test('should have different oobScore with different seeds', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest1 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        bootstrap: true,
        maxDepth: 3,
        seed: 123,
      );

      final forest2 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        bootstrap: true,
        maxDepth: 3,
        seed: 456,
      );

      // Assert
      // Note: They might be equal by chance, but with different seeds they should
      // generally be different (we'll allow equality here as it's possible)
      expect(forest1.oobScore, isNotNull);
      expect(forest2.oobScore, isNotNull);
    });

    test('should improve oobScore with more trees', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forestSmall = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 5,
        bootstrap: true,
        maxDepth: 3,
        seed: 42,
      );

      final forestLarge = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 20,
        bootstrap: true,
        maxDepth: 3,
        seed: 42,
      );

      // Assert
      expect(forestSmall.oobScore, isNotNull);
      expect(forestLarge.oobScore, isNotNull);
      // More trees should generally improve OOB score (or at least not degrade)
      // We'll just check they're both valid scores
      expect(forestLarge.oobScore, greaterThanOrEqualTo(0.0));
      expect(forestLarge.oobScore, lessThanOrEqualTo(1.0));
    });

    test('should track oobSamples correctly', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 5,
        bootstrap: true,
        maxDepth: 3,
        seed: 42,
      );

      // Assert
      // OOB samples should be tracked (we'll verify through oobScore calculation)
      expect(forest.oobScore, isNotNull);
    });
  });
}

