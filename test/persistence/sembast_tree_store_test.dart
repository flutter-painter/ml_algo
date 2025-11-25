import 'dart:io';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/persistence/sembast_tree_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:sembast/sembast_io.dart';
import 'package:test/test.dart';

import '../fake_data_set.dart';

void main() {
  group('SembastTreeStore', () {
    late String testDbPath;
    late Database database;
    late SembastTreeStore store;

    setUp(() async {
      // Create a temporary database file for testing
      final tempDir = Directory.systemTemp;
      testDbPath =
          '${tempDir.path}/test_ml_algo_${DateTime.now().millisecondsSinceEpoch}.db';
      database = await databaseFactoryIo.openDatabase(testDbPath);
      store = SembastTreeStore(database: database);
    });

    tearDown(() async {
      await database.close();
      // Clean up test database file
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should create an instance with a database', () {
      expect(store, isNotNull);
      expect(store, isA<SembastTreeStore>());
    });

    test('should save a DecisionTreeClassifier and return tree ID', () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        minError: 0.3,
        minSamplesCount: 1,
        dtype: DType.float32,
      );

      // Act
      final treeId = await store.saveTree(classifier);

      // Assert
      expect(treeId, isNotNull);
      expect(treeId, isA<String>());
      expect(treeId.isNotEmpty, isTrue);
    });

    test('should save a DecisionTreeClassifier with custom tree ID', () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );
      const customTreeId = 'my_custom_tree_id';

      // Act
      final treeId = await store.saveTree(classifier, treeId: customTreeId);

      // Assert
      expect(treeId, equals(customTreeId));
    });

    test('should load a saved DecisionTreeClassifier', () async {
      // Arrange
      final originalClassifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        minError: 0.3,
        minSamplesCount: 1,
        dtype: DType.float32,
      );
      final treeId = await store.saveTree(originalClassifier);

      // Act
      final loadedClassifier = await store.loadTree(treeId);

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
      final loadedClassifier = await store.loadTree('non_existent_tree_id');

      // Assert
      expect(loadedClassifier, isNull);
    });

    test('should delete a saved tree', () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );
      final treeId = await store.saveTree(classifier);

      // Verify it exists
      final beforeDelete = await store.loadTree(treeId);
      expect(beforeDelete, isNotNull);

      // Act
      await store.deleteTree(treeId);

      // Assert
      final afterDelete = await store.loadTree(treeId);
      expect(afterDelete, isNull);
    });

    test('should list all saved trees', () async {
      // Arrange
      final classifier1 = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );
      final classifier2 = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 5,
        dtype: DType.float64,
      );

      final treeId1 = await store.saveTree(classifier1);
      final treeId2 = await store.saveTree(classifier2);

      // Act
      final treeIds = await store.listTrees();

      // Assert
      expect(treeIds, isA<List<String>>());
      expect(treeIds.length, greaterThanOrEqualTo(2));
      expect(treeIds, contains(treeId1));
      expect(treeIds, contains(treeId2));
    });

    test('should get tree metadata', () async {
      // Arrange
      final classifier = DecisionTreeClassifier(
        fakeDataSet,
        'col_8',
        maxDepth: 3,
        minError: 0.3,
        minSamplesCount: 1,
        dtype: DType.float32,
      );
      final treeId = await store.saveTree(classifier);

      // Act
      final metadata = await store.getTreeMetadata(treeId);

      // Assert
      expect(metadata, isNotNull);
      expect(metadata, isA<Map<String, dynamic>>());
      expect(metadata!['target_column'], equals('col_8'));
      expect(metadata['max_depth'], equals(3));
      expect(metadata['min_error'], equals(0.3));
      expect(metadata['min_samples'], equals(1));
      expect(metadata['dtype'], equals('F32'));
    });

    test('should return null metadata for non-existent tree', () async {
      // Act
      final metadata = await store.getTreeMetadata('non_existent_tree_id');

      // Assert
      expect(metadata, isNull);
    });

    test('should load tree and make predictions correctly', () async {
      // Arrange
      final trainData = fakeDataSet;
      final classifier = DecisionTreeClassifier(
        trainData,
        'col_8',
        maxDepth: 3,
        dtype: DType.float32,
      );
      final treeId = await store.saveTree(classifier);

      // Act
      final loadedClassifier = await store.loadTree(treeId);
      final testData = DataFrame.fromSeries([
        Series('col_1', <int>[10, 90]),
        Series('col_2', <int>[20, 51]),
        Series('col_3', <int>[1, 0], isDiscrete: true),
        Series('col_4', <int>[0, 0], isDiscrete: true),
        Series('col_5', <int>[0, 1], isDiscrete: true),
        Series('col_6', <int>[30, 34]),
        Series('col_7', <int>[40, 31]),
      ]);
      final predictions = loadedClassifier!.predict(testData);

      // Assert
      expect(predictions, isNotNull);
      expect(predictions.header, contains('col_8'));
      expect(predictions.rows.length, equals(2));
    });
  });
}
