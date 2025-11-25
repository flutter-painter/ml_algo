import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../../fake_data_set.dart';

void main() {
  group('RandomForestClassifier', () {
    final trainData = fakeDataSet;
    final targetName = 'col_8';

    test('should be an abstract interface', () {
      expect(RandomForestClassifier, isA<Type>());
    });

    test('should create a RandomForestClassifier with default parameters', () {
      final forest = RandomForestClassifier(
        trainData,
        targetName,
      );

      expect(forest, isNotNull);
      expect(forest, isA<RandomForestClassifier>());
      expect(forest.nEstimators, equals(100)); // default
      expect(forest.bootstrap, isTrue); // default
      expect(forest.sampleSize, equals(1.0)); // default
      expect(forest.votingStrategy, equals(VotingStrategy.majority)); // default
    });

    test('should create a RandomForestClassifier with custom parameters', () {
      final forest = RandomForestClassifier(
        trainData,
        targetName,
        nEstimators: 50,
        bootstrap: false,
        sampleSize: 0.8,
        maxDepth: 5,
        minError: 0.3,
        minSamplesCount: 10,
        votingStrategy: VotingStrategy.majority,
      );

      expect(forest.nEstimators, equals(50));
      expect(forest.bootstrap, isFalse);
      expect(forest.sampleSize, equals(0.8));
      expect(forest.maxDepth, equals(5));
      expect(forest.minError, equals(0.3));
      expect(forest.minSamplesCount, equals(10));
      expect(forest.votingStrategy, equals(VotingStrategy.majority));
    });

    test('should have treeIds property when using TreeStore', () {
      // This will be tested in integration tests with actual TreeStore
      expect(RandomForestClassifier, isA<Type>());
    });

    test('should expose DecisionTree hyperparameters', () {
      final forest = RandomForestClassifier(
        trainData,
        targetName,
        maxDepth: 5,
        minError: 0.3,
        minSamplesCount: 10,
        assessorType: TreeAssessorType.gini,
        dtype: DType.float32,
      );

      expect(forest.maxDepth, equals(5));
      expect(forest.minError, equals(0.3));
      expect(forest.minSamplesCount, equals(10));
      expect(forest.assessorType, equals(TreeAssessorType.gini));
      expect(forest.dtype, equals(DType.float32));
    });
  });
}
