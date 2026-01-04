import 'dart:convert';

import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier_factory.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier_impl.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/persistence/tree_store.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

class RandomForestClassifierFactoryImpl
    implements RandomForestClassifierFactory {
  @override
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
  }) {
    // Create instance and train (training happens in constructor)
    return RandomForestClassifierImpl(
      nEstimators: nEstimators,
      bootstrap: bootstrap,
      sampleSize: sampleSize,
      maxFeatures: maxFeatures,
      votingStrategy: votingStrategy,
      trainData: trainData,
      targetName: targetName,
      minError: minError,
      minSamplesCount: minSamplesCount,
      maxDepth: maxDepth,
      assessorType: assessorType,
      dtype: dtype,
      treeStore: treeStore,
      seed: seed,
      classWeight: classWeight,
      balancedBootstrap: balancedBootstrap,
      train: true, // Train during construction
    );
  }

  @override
  RandomForestClassifier fromJson(String json) {
    if (json.isEmpty) {
      throw Exception('Provided JSON object is empty');
    }

    final decodedJson = jsonDecode(json) as Map<String, dynamic>;
    return RandomForestClassifierImpl.fromJson(decodedJson);
  }

  @override
  Future<RandomForestClassifier?> loadFromStore(
    TreeStore store,
    String forestId,
  ) async {
    return RandomForestClassifierImpl.loadFromStore(store, forestId);
  }
}
