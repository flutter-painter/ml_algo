# Random Forest with Sembast Persistence

This example demonstrates how to use Random Forest Classifier with Sembast database for persistent storage of trained models.

## Overview

Random Forest is an ensemble learning method that trains multiple decision trees and aggregates their predictions. With Sembast persistence, you can:

- **Save trained forests** to a database for later use
- **Load forests** from the database without retraining
- **Load trees on-demand** when making predictions (memory-efficient)

## Key Features

1. **Tree Storage**: Individual trees are saved to the database during training
2. **Forest Persistence**: Entire forest configuration and metadata can be saved/loaded
3. **Lazy Loading**: Trees are loaded from database only when needed for predictions

## Usage Example

```dart
// 1. Train with treeStore
final store = SembastTreeStore(database: database);
final forest = RandomForestClassifier(
  trainData,
  'target',
  nEstimators: 100,
  treeStore: store,
);

// 2. Save forest
final forestId = await forest.saveToStore(store);

// 3. Later: Load forest
final loadedForest = await RandomForestClassifier.loadFromStore(store, forestId);

// 4. Make predictions (trees loaded automatically from store)
final predictions = await loadedForest!.predictAsync(testData);
```

## Running the Example

```shell
dart example/random_forest_sembast/main.dart
```

## What This Example Shows

1. **Training**: Creates a Random Forest classifier with tree persistence
2. **Saving**: Saves the entire forest configuration to Sembast database
3. **Loading**: Restores the forest from the database
4. **Predicting**: Makes predictions using trees loaded from the database
5. **Memory Efficiency**: Demonstrates how trees can be loaded on-demand

## Use Cases

- **Model Versioning**: Save different versions of your model
- **Memory Management**: Keep models on disk, load only when needed
- **Model Sharing**: Share trained models across application instances
- **Production Deployment**: Deploy pre-trained models without retraining

