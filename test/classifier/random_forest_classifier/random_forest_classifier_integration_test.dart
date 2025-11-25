import 'dart:io';

import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/persistence/sembast_tree_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:sembast/sembast_io.dart';
import 'package:test/test.dart';

import '../../fake_data_set.dart';

void main() {
  group('RandomForestClassifier Integration', () {
    late String testDbPath;
    late Database database;
    late SembastTreeStore store;

    setUp(() async {
      final tempDir = Directory.systemTemp;
      testDbPath =
          '${tempDir.path}/test_rf_${DateTime.now().millisecondsSinceEpoch}.db';
      database = await databaseFactoryIo.openDatabase(testDbPath);
      store = SembastTreeStore(database: database);
    });

    tearDown(() async {
      await database.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should train RandomForestClassifier without TreeStore', () async {
      // Arrange
      final trainData = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 5, // Small number for faster tests
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Assert
      expect(forest, isNotNull);
      expect(forest.nEstimators, equals(5));
      expect(forest.treeIds, isNull); // No TreeStore, so no treeIds
    });

    test('should train RandomForestClassifier with TreeStore', () async {
      // Arrange
      final trainData = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 5,
        maxDepth: 3,
        treeStore: store,
        dtype: DType.float32,
      );

      // Assert
      expect(forest, isNotNull);
      expect(forest.treeIds, isNotNull);
      expect(forest.treeIds!.length, equals(5));
    });

    test('should make predictions', () async {
      // Arrange
      final trainData = fakeDataSet;
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 5,
        maxDepth: 3,
        dtype: DType.float32,
      );

      final testData = DataFrame.fromSeries([
        Series('col_1', <int>[10, 90]),
        Series('col_2', <int>[20, 51]),
        Series('col_3', <int>[1, 0], isDiscrete: true),
        Series('col_4', <int>[0, 0], isDiscrete: true),
        Series('col_5', <int>[0, 1], isDiscrete: true),
        Series('col_6', <int>[30, 34]),
        Series('col_7', <int>[40, 31]),
      ]);

      // Act
      final predictions = forest.predict(testData);

      // Assert
      expect(predictions, isNotNull);
      expect(predictions.header, contains('col_8'));
      expect(predictions.rows.length, equals(2));
    });

    test('should make probability predictions', () async {
      // Arrange
      final trainData = fakeDataSet;
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 5,
        maxDepth: 3,
        dtype: DType.float32,
      );

      final testData = DataFrame.fromSeries([
        Series('col_1', <int>[10]),
        Series('col_2', <int>[20]),
        Series('col_3', <int>[1], isDiscrete: true),
        Series('col_4', <int>[0], isDiscrete: true),
        Series('col_5', <int>[0], isDiscrete: true),
        Series('col_6', <int>[30]),
        Series('col_7', <int>[40]),
      ]);

      // Act
      final probabilities = forest.predictProbabilities(testData);

      // Assert
      expect(probabilities, isNotNull);
      expect(probabilities.header, contains('col_8'));
      expect(probabilities.rows.length, equals(1));
    });

    test('should load trees from TreeStore when making predictions', () async {
      // Arrange
      final trainData = fakeDataSet;
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        maxDepth: 3,
        treeStore: store,
        dtype: DType.float32,
      );

      final treeIds = forest.treeIds!;
      expect(treeIds.length, equals(3));

      final testData = DataFrame.fromSeries([
        Series('col_1', <int>[10]),
        Series('col_2', <int>[20]),
        Series('col_3', <int>[1], isDiscrete: true),
        Series('col_4', <int>[0], isDiscrete: true),
        Series('col_5', <int>[0], isDiscrete: true),
        Series('col_6', <int>[30]),
        Series('col_7', <int>[40]),
      ]);

      // Act - predictions should load trees from store
      final predictions = forest.predict(testData);

      // Assert
      expect(predictions, isNotNull);
      expect(predictions.rows.length, equals(1));
    });

    test('should use bootstrap sampling when enabled', () async {
      // Arrange
      final trainData = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        bootstrap: true,
        sampleSize: 0.8,
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Assert
      expect(forest.bootstrap, isTrue);
      expect(forest.sampleSize, equals(0.8));
    });

    test('should work without bootstrap sampling', () async {
      // Arrange
      final trainData = fakeDataSet;

      // Act
      final forest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        bootstrap: false,
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Assert
      expect(forest.bootstrap, isFalse);
    });

    test('should support different voting strategies', () async {
      // Arrange
      final trainData = fakeDataSet;

      // Act
      final forestMajority = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        votingStrategy: VotingStrategy.majority,
        maxDepth: 3,
        dtype: DType.float32,
      );

      final forestWeighted = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        votingStrategy: VotingStrategy.weighted,
        maxDepth: 3,
        dtype: DType.float32,
      );

      final forestAvgProb = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        votingStrategy: VotingStrategy.averageProbabilities,
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Assert
      expect(forestMajority.votingStrategy, equals(VotingStrategy.majority));
      expect(forestWeighted.votingStrategy, equals(VotingStrategy.weighted));
      expect(forestAvgProb.votingStrategy,
          equals(VotingStrategy.averageProbabilities));
    });
  });
}
