// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

TreeNode _$TreeNodeFromJson(Map<String, dynamic> json) => $checkedCreate(
      'TreeNode',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['CN', 'LB', 'PT', 'SV', 'SI'],
        );
        final val = TreeNode(
          $checkedConvert('PT', (v) => fromPredicateTypeJson(v as String?)),
          $checkedConvert('SV', (v) => v as num?),
          $checkedConvert('SI', (v) => (v as num?)?.toInt()),
          $checkedConvert('CN', (v) => fromTreeNodesJson(v as List?)),
          $checkedConvert(
              'LB',
              (v) => v == null
                  ? null
                  : TreeLeafLabel.fromJson(v as Map<String, dynamic>)),
        );
        return val;
      },
      fieldKeyMap: const {
        'predicateType': 'PT',
        'splitValue': 'SV',
        'splitIndex': 'SI',
        'children': 'CN',
        'label': 'LB'
      },
    );

Map<String, dynamic> _$TreeNodeToJson(TreeNode instance) => <String, dynamic>{
      if (treeNodesToJson(instance.children) case final value?) 'CN': value,
      if (instance.label?.toJson() case final value?) 'LB': value,
      if (predicateTypeToJson(instance.predicateType) case final value?)
        'PT': value,
      if (instance.splitValue case final value?) 'SV': value,
      if (instance.splitIndex case final value?) 'SI': value,
    };
