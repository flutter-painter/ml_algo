import 'dart:io';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/persistence/sembast_tree_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:sembast/sembast_io.dart';
import 'package:test/test.dart';

import '../../fake_data_set.dart';

void main() {
  group('DecisionTreeClassifier Persistence', () {
    late String testDbPath;
    late Database database;
    late SembastTreeStore store;

    setUp(() async {
      final tempDir = Directory.systemTemp;
      testDbPath =
          '${tempDir.path}/test_ml_algo_${DateTime.now().millisecondsSinceEpoch}.db';
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

    test('should save classifier to store using saveToStore method', () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Act
      final treeId = await classifier.saveToStore(store);

      // Assert
      expect(treeId, isNotNull);
      expect(treeId, isA<String>());
      expect(treeId.isNotEmpty, isTrue);
    });

    test('should save classifier with custom tree ID', () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );
      const customTreeId = 'custom_id_123';

      // Act
      final treeId = await classifier.saveToStore(store, treeId: customTreeId);

      // Assert
      expect(treeId, equals(customTreeId));
    });

    test('should load classifier from store using loadFromStore method',
        () async {
      // Arrange
      final originalClassifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        minError: 0.3,
        minSamplesCount: 1,
        dtype: DType.float32,
      );
      final treeId = await originalClassifier.saveToStore(store);

      // Act
      final loadedClassifier =
          await DecisionTreeClassifier.loadFromStore(store, treeId);

      // Assert
      expect(loadedClassifier, isNotNull);
      expect(loadedClassifier, isA<DecisionTreeClassifier>());
      final loadedImpl = loadedClassifier! as DecisionTreeClassifierImpl;
      final originalImpl = originalClassifier as DecisionTreeClassifierImpl;
      expect(
          loadedImpl.targetColumnName, equals(originalImpl.targetColumnName));
      expect(loadedClassifier.maxDepth, equals(originalClassifier.maxDepth));
      expect(loadedClassifier.minError, equals(originalClassifier.minError));
      expect(loadedClassifier.minSamplesCount,
          equals(originalClassifier.minSamplesCount));
      expect(loadedClassifier.dtype, equals(originalClassifier.dtype));
    });

    test('should return null when loading non-existent tree', () async {
      // Act
      final loadedClassifier = await DecisionTreeClassifier.loadFromStore(
        store,
        'non_existent_tree_id',
      );

      // Assert
      expect(loadedClassifier, isNull);
    });

    test('should maintain prediction accuracy after save/load', () async {
      // Arrange
      final trainData = fakeDataSet;
      final classifier = DecisionTreeClassifier(
        trainData,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Get original predictions
      final testData = DataFrame.fromSeries([
        Series('col_1', <int>[10, 90]),
        Series('col_2', <int>[20, 51]),
        Series('col_3', <int>[1, 0], isDiscrete: true),
        Series('col_4', <int>[0, 0], isDiscrete: true),
        Series('col_5', <int>[0, 1], isDiscrete: true),
        Series('col_6', <int>[30, 34]),
        Series('col_7', <int>[40, 31]),
      ]);
      final originalPredictions = classifier.predict(testData);

      // Save and load
      final treeId = await classifier.saveToStore(store);
      final loadedClassifier =
          await DecisionTreeClassifier.loadFromStore(store, treeId);

      // Act
      final loadedPredictions = loadedClassifier!.predict(testData);

      // Assert
      expect(loadedPredictions.header, equals(originalPredictions.header));
      expect(loadedPredictions.rows.length,
          equals(originalPredictions.rows.length));
      // Predictions should match (same tree structure)
      for (int i = 0; i < originalPredictions.rows.length; i++) {
        expect(
          loadedPredictions.rows.elementAt(i).first,
          equals(originalPredictions.rows.elementAt(i).first),
        );
      }
    });

    test('should work with both float32 and float64 dtypes', () async {
      // Arrange
      final classifier32 = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );
      final classifier64 = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float64,
      );

      // Act
      final treeId32 = await classifier32.saveToStore(store);
      final treeId64 = await classifier64.saveToStore(store);

      final loaded32 =
          await DecisionTreeClassifier.loadFromStore(store, treeId32);
      final loaded64 =
          await DecisionTreeClassifier.loadFromStore(store, treeId64);

      // Assert
      expect(loaded32, isNotNull);
      expect(loaded64, isNotNull);
      expect(loaded32!.dtype, equals(DType.float32));
      expect(loaded64!.dtype, equals(DType.float64));
    });

    test('should maintain backward compatibility with JSON serialization',
        () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );

      // Save to JSON (existing method)
      final jsonFile = File(
          '${Directory.systemTemp.path}/test_tree_${DateTime.now().millisecondsSinceEpoch}.json');
      await classifier.saveAsJson(jsonFile.path);

      // Save to store (new method)
      final treeId = await classifier.saveToStore(store);

      // Load from JSON
      final jsonContent = await jsonFile.readAsString();
      final loadedFromJson = DecisionTreeClassifier.fromJson(jsonContent);

      // Load from store
      final loadedFromStore =
          await DecisionTreeClassifier.loadFromStore(store, treeId);

      // Assert - both should have same properties
      final jsonImpl = loadedFromJson as DecisionTreeClassifierImpl;
      final storeImpl = loadedFromStore! as DecisionTreeClassifierImpl;
      expect(jsonImpl.targetColumnName, equals(storeImpl.targetColumnName));
      expect(loadedFromJson.maxDepth, equals(loadedFromStore.maxDepth));
      expect(loadedFromJson.minError, equals(loadedFromStore.minError));

      // Cleanup
      await jsonFile.delete();
    });
  });
}
