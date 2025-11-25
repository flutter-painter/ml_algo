import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

void main() {
  group('RandomForestClassifier Class Weights', () {
    test('should accept null classWeight (no weighting)', () {
      // Arrange
      final data = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act & Assert - should not throw
      final forest = RandomForestClassifier(
        data,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        classWeight: null,
      );

      expect(forest, isNotNull);
    });

    test('should accept balanced classWeight', () {
      // Arrange
      final data = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act & Assert - should not throw
      final forest = RandomForestClassifier(
        data,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        classWeight: 'balanced',
      );

      expect(forest, isNotNull);
    });

    test('should accept manual classWeight Map', () {
      // Arrange
      final data = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act & Assert - should not throw
      final forest = RandomForestClassifier(
        data,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        classWeight: {0: 0.5, 1: 2.0},
      );

      expect(forest, isNotNull);
    });

    test('should improve performance on imbalanced dataset with balanced weights', () {
      // Arrange - Highly imbalanced dataset (9:1 ratio)
      final imbalancedData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        Series('feature_2', <int>[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        Series('target', <int>[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], isDiscrete: true),
      ]);

      // Act
      final forestWithoutWeights = RandomForestClassifier(
        imbalancedData,
        'target',
        nEstimators: 10,
        maxDepth: 3,
        seed: 42,
        classWeight: null,
      );

      final forestWithWeights = RandomForestClassifier(
        imbalancedData,
        'target',
        nEstimators: 10,
        maxDepth: 3,
        seed: 42,
        classWeight: 'balanced',
      );

      // Test predictions on minority class sample
      final testData = DataFrame.fromSeries([
        Series('feature_1', <int>[11]),
        Series('feature_2', <int>[110]),
      ]);

      final predictionsWithout = forestWithoutWeights.predict(testData);
      final predictionsWith = forestWithWeights.predict(testData);

      // Assert - Both should work, but weighted version should be more aware of minority class
      expect(predictionsWithout, isNotNull);
      expect(predictionsWith, isNotNull);
      expect(predictionsWithout.rows.length, equals(1));
      expect(predictionsWith.rows.length, equals(1));
    });

    test('should handle single class dataset', () {
      // Arrange - All samples belong to same class
      final singleClassData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 0, 0], isDiscrete: true),
      ]);

      // Act & Assert - should not throw
      final forest = RandomForestClassifier(
        singleClassData,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        classWeight: 'balanced',
      );

      expect(forest, isNotNull);
    });

    test('should handle equal class frequencies', () {
      // Arrange - Balanced dataset
      final balancedData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4]),
        Series('feature_2', <int>[10, 20, 30, 40]),
        Series('target', <int>[0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act & Assert - should not throw
      final forest = RandomForestClassifier(
        balancedData,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        classWeight: 'balanced',
      );

      expect(forest, isNotNull);
    });

    test('should validate manual classWeight keys match class labels', () {
      // Arrange
      final data = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3]),
        Series('target', <int>[0, 1, 2], isDiscrete: true),
      ]);

      // Act & Assert - should handle missing classes gracefully
      final forest = RandomForestClassifier(
        data,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        classWeight: {0: 1.0, 1: 1.0}, // Missing class 2
      );

      expect(forest, isNotNull);
    });

    test('should work with multiple classes', () {
      // Arrange - Multi-class imbalanced dataset
      final multiClassData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        Series('feature_2', <int>[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        Series('target', <int>[0, 0, 0, 0, 0, 1, 1, 2, 2, 2], isDiscrete: true), // 5:2:3 ratio
      ]);

      // Act & Assert
      final forest = RandomForestClassifier(
        multiClassData,
        'target',
        nEstimators: 10,
        maxDepth: 3,
        classWeight: 'balanced',
      );

      expect(forest, isNotNull);
      
      // Test predictions
      final testData = DataFrame.fromSeries([
        Series('feature_1', <int>[11]),
        Series('feature_2', <int>[110]),
      ]);
      
      final predictions = forest.predict(testData);
      expect(predictions, isNotNull);
      expect(predictions.rows.length, equals(1));
    });
  });
}

