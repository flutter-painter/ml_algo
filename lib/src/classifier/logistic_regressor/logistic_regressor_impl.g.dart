// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'logistic_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LogisticRegressorImpl _$LogisticRegressorImplFromJson(
        Map<String, dynamic> json) =>
    $checkedCreate(
      'LogisticRegressorImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const [
            'O',
            'I',
            'LR',
            'D',
            'DR',
            'U',
            'L',
            'R',
            'RS',
            'B',
            'N',
            'LRT',
            'ICT',
            'IC',
            'CBC',
            'CN',
            'FI',
            'IS',
            'DT',
            'PT',
            'PL',
            'NL',
            'LF',
            'CPI',
            r'$V'
          ],
        );
        final val = LogisticRegressorImpl(
          $checkedConvert(
              'O',
              (v) => const LinearOptimizerTypeJsonConverter()
                  .fromJson(v as String)),
          $checkedConvert('I', (v) => (v as num).toInt()),
          $checkedConvert('LR', (v) => (v as num).toDouble()),
          $checkedConvert('D', (v) => (v as num).toDouble()),
          $checkedConvert('DR', (v) => (v as num).toInt()),
          $checkedConvert('U', (v) => (v as num).toDouble()),
          $checkedConvert('L', (v) => (v as num).toDouble()),
          $checkedConvert(
              'R',
              (v) => const RegularizationTypeJsonConverterNullable()
                  .fromJson(v as String?)),
          $checkedConvert('RS', (v) => (v as num?)?.toInt()),
          $checkedConvert('B', (v) => (v as num).toInt()),
          $checkedConvert('N', (v) => v as bool),
          $checkedConvert(
              'LRT',
              (v) =>
                  const LearningRateTypeJsonConverter().fromJson(v as String)),
          $checkedConvert(
              'ICT',
              (v) => const InitialCoefficientsTypeJsonConverter()
                  .fromJson(v as String)),
          $checkedConvert(
              'IC',
              (v) => v == null
                  ? null
                  : Vector.fromJson(v as Map<String, dynamic>)),
          $checkedConvert(
              'CN', (v) => (v as List<dynamic>).map((e) => e as String)),
          $checkedConvert('LF', (v) => fromLinkFunctionJson(v as String)),
          $checkedConvert('FI', (v) => v as bool),
          $checkedConvert('IS', (v) => v as num),
          $checkedConvert(
              'CBC', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
          $checkedConvert('PT', (v) => v as num),
          $checkedConvert('NL', (v) => v as num),
          $checkedConvert('PL', (v) => v as num),
          $checkedConvert('CPI',
              (v) => (v as List<dynamic>?)?.map((e) => e as num).toList()),
          $checkedConvert(
              'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
          schemaVersion: $checkedConvert(
              r'$V',
              (v) =>
                  (v as num?)?.toInt() ?? logisticRegressorJsonSchemaVersion),
        );
        return val;
      },
      fieldKeyMap: const {
        'optimizerType': 'O',
        'iterationsLimit': 'I',
        'initialLearningRate': 'LR',
        'decay': 'D',
        'dropRate': 'DR',
        'minCoefficientsUpdate': 'U',
        'lambda': 'L',
        'regularizationType': 'R',
        'randomSeed': 'RS',
        'batchSize': 'B',
        'isFittingDataNormalized': 'N',
        'learningRateType': 'LRT',
        'initialCoefficientsType': 'ICT',
        'initialCoefficients': 'IC',
        'targetNames': 'CN',
        'linkFunction': 'LF',
        'fitIntercept': 'FI',
        'interceptScale': 'IS',
        'coefficientsByClasses': 'CBC',
        'probabilityThreshold': 'PT',
        'negativeLabel': 'NL',
        'positiveLabel': 'PL',
        'costPerIteration': 'CPI',
        'dtype': 'DT',
        'schemaVersion': r'$V'
      },
    );

Map<String, dynamic> _$LogisticRegressorImplToJson(
        LogisticRegressorImpl instance) =>
    <String, dynamic>{
      'O': const LinearOptimizerTypeJsonConverter()
          .toJson(instance.optimizerType),
      'I': instance.iterationsLimit,
      'LR': instance.initialLearningRate,
      'D': instance.decay,
      'DR': instance.dropRate,
      'U': instance.minCoefficientsUpdate,
      'L': instance.lambda,
      if (const RegularizationTypeJsonConverterNullable()
              .toJson(instance.regularizationType)
          case final value?)
        'R': value,
      if (instance.randomSeed case final value?) 'RS': value,
      'B': instance.batchSize,
      'N': instance.isFittingDataNormalized,
      'LRT': const LearningRateTypeJsonConverter()
          .toJson(instance.learningRateType),
      'ICT': const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType),
      if (instance.initialCoefficients?.toJson() case final value?) 'IC': value,
      'CBC': instance.coefficientsByClasses.toJson(),
      'CN': instance.targetNames.toList(),
      'FI': instance.fitIntercept,
      'IS': instance.interceptScale,
      'DT': const DTypeJsonConverter().toJson(instance.dtype),
      'PT': instance.probabilityThreshold,
      'PL': instance.positiveLabel,
      'NL': instance.negativeLabel,
      'LF': linkFunctionToJson(instance.linkFunction),
      if (instance.costPerIteration case final value?) 'CPI': value,
      if (instance.schemaVersion case final value?) r'$V': value,
    };
