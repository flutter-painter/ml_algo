// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeNode _$KDTreeNodeFromJson(Map<String, dynamic> json) => $checkedCreate(
      'KDTreeNode',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['I', 'L', 'R', 'P'],
        );
        final val = KDTreeNode(
          splitIndex: $checkedConvert('I', (v) => (v as num).toInt()),
          left: $checkedConvert(
              'L',
              (v) => v == null
                  ? null
                  : KDTreeNode.fromJson(v as Map<String, dynamic>)),
          right: $checkedConvert(
              'R',
              (v) => v == null
                  ? null
                  : KDTreeNode.fromJson(v as Map<String, dynamic>)),
          pointIndices: $checkedConvert(
              'P',
              (v) =>
                  (v as List<dynamic>).map((e) => (e as num).toInt()).toList()),
        );
        return val;
      },
      fieldKeyMap: const {
        'splitIndex': 'I',
        'left': 'L',
        'right': 'R',
        'pointIndices': 'P'
      },
    );

Map<String, dynamic> _$KDTreeNodeToJson(KDTreeNode instance) =>
    <String, dynamic>{
      'I': instance.splitIndex,
      if (instance.left?.toJson() case final value?) 'L': value,
      if (instance.right?.toJson() case final value?) 'R': value,
      'P': instance.pointIndices,
    };
