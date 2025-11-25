/// Voting strategies for aggregating tree predictions in Random Forest
enum VotingStrategy {
  /// Majority vote: Most common class wins
  majority,

  /// Weighted vote: Weight by tree accuracy (if available)
  weighted,

  /// Average probabilities: Average probability distributions, then pick max
  averageProbabilities,
}
