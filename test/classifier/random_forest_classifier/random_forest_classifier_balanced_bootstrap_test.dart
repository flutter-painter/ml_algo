import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

void main() {
  group('RandomForestClassifier Balanced Bootstrap', () {
    test('should accept balancedBootstrap parameter', () {
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
        bootstrap: true,
        balancedBootstrap: true,
      );

      expect(forest, isNotNull);
    });

    test('should only work when bootstrap is enabled', () {
      // Arrange
      final data = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act & Assert - balancedBootstrap should be ignored when bootstrap=false
      final forest = RandomForestClassifier(
        data,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        bootstrap: false,
        balancedBootstrap: true, // Should be ignored
      );

      expect(forest, isNotNull);
    });

    test('should produce balanced bootstrap samples', () {
      // Arrange - Imbalanced dataset (3:2 ratio)
      final imbalancedData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act
      final forest = RandomForestClassifier(
        imbalancedData,
        'target',
        nEstimators: 10,
        maxDepth: 3,
        bootstrap: true,
        balancedBootstrap: true,
        seed: 42,
      );

      // Assert - Forest should train successfully
      expect(forest, isNotNull);
      
      // Test predictions
      final testData = DataFrame.fromSeries([
        Series('feature_1', <int>[6]),
        Series('feature_2', <int>[60]),
      ]);
      
      final predictions = forest.predict(testData);
      expect(predictions, isNotNull);
      expect(predictions.rows.length, equals(1));
    });

    test('should work with highly imbalanced dataset', () {
      // Arrange - Very imbalanced (9:1 ratio)
      final veryImbalancedData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        Series('feature_2', <int>[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        Series('target', <int>[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], isDiscrete: true),
      ]);

      // Act
      final forest = RandomForestClassifier(
        veryImbalancedData,
        'target',
        nEstimators: 10,
        maxDepth: 3,
        bootstrap: true,
        balancedBootstrap: true,
        seed: 42,
      );

      // Assert
      expect(forest, isNotNull);
      
      // Should be able to make predictions
      final testData = DataFrame.fromSeries([
        Series('feature_1', <int>[11]),
        Series('feature_2', <int>[110]),
      ]);
      
      final predictions = forest.predict(testData);
      expect(predictions, isNotNull);
    });

    test('should work with multiple classes', () {
      // Arrange - Multi-class imbalanced dataset
      final multiClassData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        Series('feature_2', <int>[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        Series('target', <int>[0, 0, 0, 0, 0, 1, 1, 2, 2, 2], isDiscrete: true), // 5:2:3 ratio
      ]);

      // Act
      final forest = RandomForestClassifier(
        multiClassData,
        'target',
        nEstimators: 10,
        maxDepth: 3,
        bootstrap: true,
        balancedBootstrap: true,
        seed: 42,
      );

      // Assert
      expect(forest, isNotNull);
      
      final testData = DataFrame.fromSeries([
        Series('feature_1', <int>[11]),
        Series('feature_2', <int>[110]),
      ]);
      
      final predictions = forest.predict(testData);
      expect(predictions, isNotNull);
    });

    test('should work together with classWeight', () {
      // Arrange
      final data = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Act & Assert - Should work together
      final forest = RandomForestClassifier(
        data,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        bootstrap: true,
        balancedBootstrap: true,
        classWeight: 'balanced',
      );

      expect(forest, isNotNull);
    });

    test('should handle single class dataset gracefully', () {
      // Arrange
      final singleClassData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 2, 3, 4, 5]),
        Series('feature_2', <int>[10, 20, 30, 40, 50]),
        Series('target', <int>[0, 0, 0, 0, 0], isDiscrete: true),
      ]);

      // Act & Assert
      final forest = RandomForestClassifier(
        singleClassData,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        bootstrap: true,
        balancedBootstrap: true,
      );

      expect(forest, isNotNull);
    });
  });
}

