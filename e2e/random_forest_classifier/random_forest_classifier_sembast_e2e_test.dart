import 'dart:io';

import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/persistence/sembast_tree_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:sembast/sembast_io.dart';
import 'package:test/test.dart';

void main() {
  group('RandomForestClassifier Sembast E2E', () {
    late String testDbPath;
    late Database database;
    late SembastTreeStore store;

    setUp(() async {
      final tempDir = Directory.systemTemp;
      testDbPath =
          '${tempDir.path}/test_rf_e2e_${DateTime.now().millisecondsSinceEpoch}.db';
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

    test('Full workflow: Train -> Save -> Load -> Predict', () async {
      // Step 1: Create training data
      final trainData = DataFrame.fromSeries([
        Series('feature1', <double>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        Series('feature2', <double>[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
        Series('feature3', <double>[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]),
        Series('target', <int>[0, 0, 1, 1, 0, 0, 1, 1], isDiscrete: true),
      ]);

      // Step 2: Train a Random Forest with treeStore
      final originalForest = RandomForestClassifier(
        trainData,
        'target',
        nEstimators: 5,
        maxDepth: 3,
        treeStore: store,
        dtype: DType.float32,
        votingStrategy: VotingStrategy.majority,
        bootstrap: true,
        seed: 42,
      );

      // Verify forest was trained
      expect(originalForest, isNotNull);
      expect(originalForest.nEstimators, equals(5));
      expect(originalForest.treeIds, isNotNull);
      expect(originalForest.treeIds!.length, equals(5));

      // Wait for tree saves to complete
      await Future.delayed(Duration(milliseconds: 200));

      // Verify trees are in store
      for (final treeId in originalForest.treeIds!) {
        final tree = await store.loadTree(treeId);
        expect(tree, isNotNull, reason: 'Tree $treeId should be in store');
      }

      // Step 3: Save the forest
      final forestId = await originalForest.saveToStore(store);
      expect(forestId, isNotNull);
      expect(forestId.isNotEmpty, isTrue);

      // Step 4: Load the forest back
      final loadedForest = await RandomForestClassifier.loadFromStore(
        store,
        forestId,
      );

      expect(loadedForest, isNotNull);
      expect(loadedForest!.nEstimators, equals(originalForest.nEstimators));
      expect(loadedForest.maxDepth, equals(originalForest.maxDepth));
      expect(loadedForest.votingStrategy, equals(originalForest.votingStrategy));
      expect(loadedForest.treeIds, isNotNull);
      expect(loadedForest.treeIds!.length, equals(5));

      // Step 5: Make predictions with loaded forest
      // This should load trees from store automatically
      final testData = DataFrame.fromSeries([
        Series('feature1', <double>[2.5]),
        Series('feature2', <double>[25.0]),
        Series('feature3', <double>[250.0]),
      ]);

      // Use async predict to load trees from store
      final predictions = await loadedForest.predictAsync(testData);

      expect(predictions, isNotNull);
      expect(predictions.header, contains('target'));
      expect(predictions.rows.length, equals(1));

      // Step 6: Verify predictions are reasonable
      final predictedValue = predictions.rows.first.first;
      expect(predictedValue, isA<num>());
      // Should predict either 0 or 1 (binary classification)
      expect([0, 1], contains(predictedValue));

      // Step 7: Test probability predictions
      final probabilities = await loadedForest.predictProbabilitiesAsync(testData);
      expect(probabilities, isNotNull);
      expect(probabilities.header, contains('target'));
      expect(probabilities.rows.length, equals(1));
    });

    test('Full workflow with different configurations', () async {
      // Test with different voting strategy and parameters
      final trainData = DataFrame.fromSeries([
        Series('x1', <double>[1.0, 2.0, 3.0, 4.0]),
        Series('x2', <double>[5.0, 6.0, 7.0, 8.0]),
        Series('y', <int>[0, 1, 0, 1], isDiscrete: true),
      ]);

      // Train with weighted voting
      final forest = RandomForestClassifier(
        trainData,
        'y',
        nEstimators: 3,
        maxDepth: 2,
        treeStore: store,
        votingStrategy: VotingStrategy.weighted,
        bootstrap: false,
        dtype: DType.float64,
      );

      await Future.delayed(Duration(milliseconds: 200));

      // Save
      final forestId = await forest.saveToStore(store);

      // Load
      final loaded = await RandomForestClassifier.loadFromStore(store, forestId);

      expect(loaded, isNotNull);
      expect(loaded!.votingStrategy, equals(VotingStrategy.weighted));
      expect(loaded.bootstrap, equals(false));
      expect(loaded.dtype, equals(DType.float64));

      // Predict
      final testData = DataFrame.fromSeries([
        Series('x1', <double>[2.5]),
        Series('x2', <double>[6.5]),
      ]);

      final predictions = await loaded.predictAsync(testData);
      expect(predictions, isNotNull);
      expect(predictions.rows.length, equals(1));
    });

    test('Multiple forests in same database', () async {
      // Train multiple forests and verify they don't interfere
      final trainData = DataFrame.fromSeries([
        Series('f1', <double>[1.0, 2.0, 3.0]),
        Series('f2', <double>[4.0, 5.0, 6.0]),
        Series('label', <int>[0, 1, 0], isDiscrete: true),
      ]);

      final forest1 = RandomForestClassifier(
        trainData,
        'label',
        nEstimators: 2,
        maxDepth: 2,
        treeStore: store,
        seed: 1,
      );

      final forest2 = RandomForestClassifier(
        trainData,
        'label',
        nEstimators: 2,
        maxDepth: 2,
        treeStore: store,
        seed: 2,
      );

      await Future.delayed(Duration(milliseconds: 200));

      final id1 = await forest1.saveToStore(store);
      final id2 = await forest2.saveToStore(store);

      expect(id1, isNot(equals(id2)));

      final loaded1 = await RandomForestClassifier.loadFromStore(store, id1);
      final loaded2 = await RandomForestClassifier.loadFromStore(store, id2);

      expect(loaded1, isNotNull);
      expect(loaded2, isNotNull);
      expect(loaded1!.treeIds, isNot(equals(loaded2!.treeIds)));

      // Both should work independently
      final testData = DataFrame.fromSeries([
        Series('f1', <double>[2.0]),
        Series('f2', <double>[5.0]),
      ]);

      final pred1 = await loaded1.predictAsync(testData);
      final pred2 = await loaded2.predictAsync(testData);

      expect(pred1, isNotNull);
      expect(pred2, isNotNull);
    });
  });
}

