import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';

class VotingStrategyJsonConverter implements JsonConverter<VotingStrategy, String> {
  const VotingStrategyJsonConverter();

  @override
  VotingStrategy fromJson(String json) {
    switch (json) {
      case 'majority':
        return VotingStrategy.majority;
      case 'weighted':
        return VotingStrategy.weighted;
      case 'average_probabilities':
        return VotingStrategy.averageProbabilities;
      default:
        return VotingStrategy.majority;
    }
  }

  @override
  String toJson(VotingStrategy object) {
    switch (object) {
      case VotingStrategy.majority:
        return 'majority';
      case VotingStrategy.weighted:
        return 'weighted';
      case VotingStrategy.averageProbabilities:
        return 'average_probabilities';
    }
  }
}

