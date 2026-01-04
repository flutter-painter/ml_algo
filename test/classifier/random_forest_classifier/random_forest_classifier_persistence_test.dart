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
  group('RandomForestClassifier Persistence', () {
    late String testDbPath;
    late Database database;
    late SembastTreeStore store;

    setUp(() async {
      final tempDir = Directory.systemTemp;
      testDbPath =
          '${tempDir.path}/test_rf_persistence_${DateTime.now().millisecondsSinceEpoch}.db';
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

    test('should save forest to store and return forest ID', () async {
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

      // Wait for tree saves to complete
      await Future.delayed(Duration(milliseconds: 100));

      // Act
      final forestId = await forest.saveToStore(store);

      // Assert
      expect(forestId, isNotNull);
      expect(forestId, isA<String>());
      expect(forestId.isNotEmpty, isTrue);
    });

    test('should save forest with custom forest ID', () async {
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

      await Future.delayed(Duration(milliseconds: 100));
      const customForestId = 'my_custom_forest_id';

      // Act
      final forestId = await forest.saveToStore(store, forestId: customForestId);

      // Assert
      expect(forestId, equals(customForestId));
    });

    test('should load forest from store', () async {
      // Arrange
      final trainData = fakeDataSet;
      final originalForest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        maxDepth: 3,
        treeStore: store,
        dtype: DType.float32,
        votingStrategy: VotingStrategy.weighted,
        bootstrap: true,
        sampleSize: 0.8,
      );

      await Future.delayed(Duration(milliseconds: 100));
      final forestId = await originalForest.saveToStore(store);

      // Act
      final loadedForest = await RandomForestClassifier.loadFromStore(
        store,
        forestId,
      );

      // Assert
      expect(loadedForest, isNotNull);
      expect(loadedForest, isA<RandomForestClassifier>());
      expect(loadedForest!.nEstimators, equals(originalForest.nEstimators));
      expect(loadedForest.bootstrap, equals(originalForest.bootstrap));
      expect(loadedForest.sampleSize, equals(originalForest.sampleSize));
      expect(loadedForest.votingStrategy, equals(originalForest.votingStrategy));
      expect(loadedForest.maxDepth, equals(originalForest.maxDepth));
      expect(loadedForest.treeIds, isNotNull);
      expect(loadedForest.treeIds!.length, equals(3));
    });

    test('should return null when loading non-existent forest', () async {
      // Act
      final loadedForest = await RandomForestClassifier.loadFromStore(
        store,
        'non_existent_forest_id',
      );

      // Assert
      expect(loadedForest, isNull);
    });

    test('should make predictions with loaded forest', () async {
      // Arrange
      final trainData = fakeDataSet;
      final originalForest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        maxDepth: 3,
        treeStore: store,
        dtype: DType.float32,
      );

      await Future.delayed(Duration(milliseconds: 100));
      final forestId = await originalForest.saveToStore(store);

      final loadedForest = await RandomForestClassifier.loadFromStore(
        store,
        forestId,
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

      // Act: Use async predict to load trees from store
      final predictions = await loadedForest!.predictAsync(testData);

      // Assert
      expect(predictions, isNotNull);
      expect(predictions.header, contains('col_8'));
      expect(predictions.rows.length, equals(1));
    });

    test('should preserve all forest configuration when loading', () async {
      // Arrange
      final trainData = fakeDataSet;
      final originalForest = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 5,
        maxDepth: 4,
        minError: 0.3,
        minSamplesCount: 2,
        treeStore: store,
        dtype: DType.float64,
        votingStrategy: VotingStrategy.averageProbabilities,
        bootstrap: false,
        seed: 42,
      );

      await Future.delayed(Duration(milliseconds: 100));
      final forestId = await originalForest.saveToStore(store);

      // Act
      final loadedForest = await RandomForestClassifier.loadFromStore(
        store,
        forestId,
      );

      // Assert
      expect(loadedForest, isNotNull);
      expect(loadedForest!.nEstimators, equals(5));
      expect(loadedForest.maxDepth, equals(4));
      expect(loadedForest.minError, equals(0.3));
      expect(loadedForest.minSamplesCount, equals(2));
      expect(loadedForest.dtype, equals(DType.float64));
      expect(loadedForest.votingStrategy,
          equals(VotingStrategy.averageProbabilities));
      expect(loadedForest.bootstrap, equals(false));
    });

    test('should work with different voting strategies', () async {
      // Arrange
      final trainData = fakeDataSet;

      final forestMajority = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        maxDepth: 3,
        treeStore: store,
        votingStrategy: VotingStrategy.majority,
        dtype: DType.float32,
      );

      final forestWeighted = RandomForestClassifier(
        trainData,
        'col_8',
        nEstimators: 3,
        maxDepth: 3,
        treeStore: store,
        votingStrategy: VotingStrategy.weighted,
        dtype: DType.float32,
      );

      await Future.delayed(Duration(milliseconds: 100));

      final idMajority = await forestMajority.saveToStore(store);
      final idWeighted = await forestWeighted.saveToStore(store);

      // Act
      final loadedMajority =
          await RandomForestClassifier.loadFromStore(store, idMajority);
      final loadedWeighted =
          await RandomForestClassifier.loadFromStore(store, idWeighted);

      // Assert
      expect(loadedMajority, isNotNull);
      expect(loadedWeighted, isNotNull);
      expect(loadedMajority!.votingStrategy, equals(VotingStrategy.majority));
      expect(loadedWeighted!.votingStrategy, equals(VotingStrategy.weighted));
    });
  });
}

