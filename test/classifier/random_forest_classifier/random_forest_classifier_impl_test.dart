import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier_impl.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../../fake_data_set.dart';

void main() {
  group('RandomForestClassifierImpl', () {
    final trainData = fakeDataSet;
    final targetName = 'col_8';

    test('should create instance with required parameters', () {
      final forest = RandomForestClassifierImpl(
        nEstimators: 10,
        bootstrap: true,
        sampleSize: 1.0,
        maxFeatures: 'sqrt',
        votingStrategy: VotingStrategy.majority,
        trainData: trainData,
        targetName: targetName,
        minError: 0.5,
        minSamplesCount: 1,
        maxDepth: 10,
        assessorType: TreeAssessorType.gini,
        dtype: DType.float32,
      );

      expect(forest, isNotNull);
      expect(forest.nEstimators, equals(10));
      expect(forest.targetName, equals(targetName));
    });

    test('should have null treeIds when not using TreeStore', () {
      final forest = RandomForestClassifierImpl(
        nEstimators: 3,
        bootstrap: true,
        sampleSize: 1.0,
        maxFeatures: 'sqrt',
        votingStrategy: VotingStrategy.majority,
        trainData: trainData,
        targetName: targetName,
        minError: 0.5,
        minSamplesCount: 1,
        maxDepth: 10,
        assessorType: TreeAssessorType.gini,
        dtype: DType.float32,
      );

      expect(forest.treeIds, isNull);
    });

    test('should have null treeIds when not using TreeStore', () {
      final forest = RandomForestClassifierImpl(
        nEstimators: 10,
        bootstrap: true,
        sampleSize: 1.0,
        maxFeatures: 'sqrt',
        votingStrategy: VotingStrategy.majority,
        trainData: trainData,
        targetName: targetName,
        minError: 0.5,
        minSamplesCount: 1,
        maxDepth: 10,
        assessorType: TreeAssessorType.gini,
        dtype: DType.float32,
      );

      expect(forest.treeIds, isNull);
    });
  });
}
