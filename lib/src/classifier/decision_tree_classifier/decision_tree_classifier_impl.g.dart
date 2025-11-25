// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'decision_tree_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DecisionTreeClassifierImpl _$DecisionTreeClassifierImplFromJson(
        Map<String, dynamic> json) =>
    $checkedCreate(
      'DecisionTreeClassifierImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['E', 'S', 'D', 'DT', 'T', 'R', r'$V', 'A'],
        );
        final val = DecisionTreeClassifierImpl(
          $checkedConvert('E', (v) => v as num),
          $checkedConvert('S', (v) => (v as num).toInt()),
          $checkedConvert('D', (v) => (v as num).toInt()),
          $checkedConvert(
              'R', (v) => TreeNode.fromJson(v as Map<String, dynamic>)),
          $checkedConvert('T', (v) => v as String),
          $checkedConvert('A', (v) => fromTreeAssessorTypeJson(v as String?)),
          $checkedConvert(
              'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
          schemaVersion: $checkedConvert(
              r'$V',
              (v) =>
                  (v as num?)?.toInt() ??
                  decisionTreeClassifierJsonSchemaVersion),
        );
        return val;
      },
      fieldKeyMap: const {
        'minError': 'E',
        'minSamplesCount': 'S',
        'maxDepth': 'D',
        'treeRootNode': 'R',
        'targetColumnName': 'T',
        'assessorType': 'A',
        'dtype': 'DT',
        'schemaVersion': r'$V'
      },
    );

Map<String, dynamic> _$DecisionTreeClassifierImplToJson(
        DecisionTreeClassifierImpl instance) =>
    <String, dynamic>{
      'E': instance.minError,
      'S': instance.minSamplesCount,
      'D': instance.maxDepth,
      'DT': const DTypeJsonConverter().toJson(instance.dtype),
      'T': instance.targetColumnName,
      'R': instance.treeRootNode.toJson(),
      r'$V': instance.schemaVersion,
      'A': toTreeAssessorTypeJson(instance.assessorType),
    };
