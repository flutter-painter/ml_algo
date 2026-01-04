import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/persistence/tree_store.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

abstract class RandomForestClassifierFactory {
  RandomForestClassifier create({
    required DataFrame trainData,
    required String targetName,
    required int nEstimators,
    required bool bootstrap,
    required double sampleSize,
    required dynamic maxFeatures,
    required VotingStrategy votingStrategy,
    TreeStore? treeStore,
    required bool parallel,
    int? seed,
    required num minError,
    required int minSamplesCount,
    required int maxDepth,
    required TreeAssessorType assessorType,
    required DType dtype,
    dynamic classWeight,
    required bool balancedBootstrap,
  });

  RandomForestClassifier fromJson(String json);

  Future<RandomForestClassifier?> loadFromStore(
    TreeStore store,
    String forestId,
  );
}
