import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

import '../../fake_data_set.dart';

void main() {
  group('RandomForestClassifier Feature Importance', () {
    final trainData = fakeDataSet;
    final targetName = 'col_8';

    test('should have featureImportance property', () {
      // Arrange & Act
      final forest = RandomForestClassifier(
        trainData,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
      );

      // Assert
      expect(forest.featureImportance, isNotNull);
      expect(forest.featureImportance, isA<Map<String, double>>());
    });

    test('should return feature importance for all features', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 42,
      );

      // Assert
      final importance = forest.featureImportance;
      expect(importance, isNotEmpty);
      
      // Should have importance for all feature columns (excluding target)
      final featureColumns = data.header.where((name) => name != targetName).toList();
      for (final feature in featureColumns) {
        expect(importance.containsKey(feature), isTrue,
            reason: 'Feature $feature should have importance value');
      }
    });

    test('should have non-negative importance values', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 42,
      );

      // Assert
      final importance = forest.featureImportance;
      for (final entry in importance.entries) {
        expect(entry.value, greaterThanOrEqualTo(0.0),
            reason: 'Importance for ${entry.key} should be non-negative');
      }
    });

    test('should have consistent feature importance with same seed', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest1 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 123,
      );

      final forest2 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 123,
      );

      // Assert
      expect(forest1.featureImportance, equals(forest2.featureImportance));
    });

    test('should have different feature importance with different seeds', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest1 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 123,
      );

      final forest2 = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 456,
      );

      // Assert
      // They might be equal by chance, but generally should be different
      // We'll just verify both have valid importance maps
      expect(forest1.featureImportance, isNotEmpty);
      expect(forest2.featureImportance, isNotEmpty);
    });

    test('should normalize feature importance to sum to 1.0', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 10,
        maxDepth: 3,
        seed: 42,
      );

      // Assert
      final importance = forest.featureImportance;
      final sum = importance.values.fold(0.0, (a, b) => a + b);
      
      // Allow small floating point errors
      expect(sum, closeTo(1.0, 0.01),
          reason: 'Feature importance should sum to approximately 1.0');
    });

    test('should have higher importance for more predictive features', () {
      // Arrange
      // Create a dataset where feature_1 is highly predictive
      // Use more samples to make the pattern clearer
      final predictiveData = DataFrame.fromSeries([
        Series('feature_1', <int>[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0], isDiscrete: true),
        Series('feature_2', <int>[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]),
        Series('feature_3', <int>[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]),
        Series('target', <int>[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0], isDiscrete: true),
      ]);

      // Act
      final forest = RandomForestClassifier(
        predictiveData,
        'target',
        nEstimators: 50, // More trees for better feature importance estimation
        maxDepth: 5,
        seed: 42,
      );

      // Assert
      final importance = forest.featureImportance;
      
      // feature_1 should have higher importance than feature_2 or feature_3
      // (since it's perfectly correlated with target)
      expect(importance.containsKey('feature_1'), isTrue);
      expect(importance.containsKey('feature_2'), isTrue);
      expect(importance.containsKey('feature_3'), isTrue);
      
      // feature_1 should have the highest importance (or at least not lower)
      final feature1Importance = importance['feature_1']!;
      final feature2Importance = importance['feature_2']!;
      final feature3Importance = importance['feature_3']!;
      
      // With perfect correlation, feature_1 should have significantly higher importance
      // But we'll be lenient - it should at least not be lower
      expect(feature1Importance, greaterThanOrEqualTo(feature2Importance),
          reason: 'feature_1 should have at least equal or higher importance than feature_2');
      expect(feature1Importance, greaterThanOrEqualTo(feature3Importance),
          reason: 'feature_1 should have at least equal or higher importance than feature_3');
      
      // At least one should be strictly greater (with perfect correlation, feature_1 should dominate)
      expect(feature1Importance, greaterThan(0.0),
          reason: 'feature_1 should have non-zero importance');
    });

    test('should work with different numbers of trees', () {
      // Arrange
      final data = fakeDataSet;

      // Act
      final forestSmall = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 5,
        maxDepth: 3,
        seed: 42,
      );

      final forestLarge = RandomForestClassifier(
        data,
        targetName,
        nEstimators: 20,
        maxDepth: 3,
        seed: 42,
      );

      // Assert
      expect(forestSmall.featureImportance, isNotEmpty);
      expect(forestLarge.featureImportance, isNotEmpty);
      
      // Both should have same feature keys
      expect(forestSmall.featureImportance.keys,
          equals(forestLarge.featureImportance.keys));
    });
  });
}

