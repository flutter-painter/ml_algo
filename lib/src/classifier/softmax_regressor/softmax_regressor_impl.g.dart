// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'softmax_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SoftmaxRegressorImpl _$SoftmaxRegressorImplFromJson(
        Map<String, dynamic> json) =>
    $checkedCreate(
      'SoftmaxRegressorImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const [
            'OT',
            'IL',
            'ILR',
            'D',
            'DR',
            'MCU',
            'L',
            'RT',
            'RS',
            'BS',
            'FDN',
            'LR',
            'ICT',
            'IC',
            'CN',
            'FI',
            'IS',
            'CBC',
            'DT',
            'LF',
            'PL',
            'NL',
            'CPI',
            r'$V'
          ],
        );
        final val = SoftmaxRegressorImpl(
          $checkedConvert(
              'OT',
              (v) => const LinearOptimizerTypeJsonConverter()
                  .fromJson(v as String)),
          $checkedConvert('IL', (v) => (v as num).toInt()),
          $checkedConvert('ILR', (v) => (v as num).toDouble()),
          $checkedConvert('D', (v) => (v as num).toDouble()),
          $checkedConvert('DR', (v) => (v as num).toInt()),
          $checkedConvert('MCU', (v) => (v as num).toDouble()),
          $checkedConvert('L', (v) => (v as num).toDouble()),
          $checkedConvert(
              'RT',
              (v) => const RegularizationTypeJsonConverterNullable()
                  .fromJson(v as String?)),
          $checkedConvert('RS', (v) => (v as num?)?.toInt()),
          $checkedConvert('BS', (v) => (v as num).toInt()),
          $checkedConvert('FDN', (v) => v as bool),
          $checkedConvert(
              'LR',
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
                  : Matrix.fromJson(v as Map<String, dynamic>)),
          $checkedConvert(
              'CBC', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
          $checkedConvert(
              'CN', (v) => (v as List<dynamic>).map((e) => e as String)),
          $checkedConvert('LF', (v) => fromLinkFunctionJson(v as String)),
          $checkedConvert('FI', (v) => v as bool),
          $checkedConvert('IS', (v) => v as num),
          $checkedConvert('PL', (v) => v as num),
          $checkedConvert('NL', (v) => v as num),
          $checkedConvert('CPI',
              (v) => (v as List<dynamic>?)?.map((e) => e as num).toList()),
          $checkedConvert(
              'DT', (v) => const DTypeJsonConverter().fromJson(v as String)),
          schemaVersion: $checkedConvert(r'$V',
              (v) => (v as num?)?.toInt() ?? softmaxRegressorJsonSchemaVersion),
        );
        return val;
      },
      fieldKeyMap: const {
        'optimizerType': 'OT',
        'iterationsLimit': 'IL',
        'initialLearningRate': 'ILR',
        'decay': 'D',
        'dropRate': 'DR',
        'minCoefficientsUpdate': 'MCU',
        'lambda': 'L',
        'regularizationType': 'RT',
        'randomSeed': 'RS',
        'batchSize': 'BS',
        'isFittingDataNormalized': 'FDN',
        'learningRateType': 'LR',
        'initialCoefficientsType': 'ICT',
        'initialCoefficients': 'IC',
        'coefficientsByClasses': 'CBC',
        'targetNames': 'CN',
        'linkFunction': 'LF',
        'fitIntercept': 'FI',
        'interceptScale': 'IS',
        'positiveLabel': 'PL',
        'negativeLabel': 'NL',
        'costPerIteration': 'CPI',
        'dtype': 'DT',
        'schemaVersion': r'$V'
      },
    );

Map<String, dynamic> _$SoftmaxRegressorImplToJson(
        SoftmaxRegressorImpl instance) =>
    <String, dynamic>{
      'OT': const LinearOptimizerTypeJsonConverter()
          .toJson(instance.optimizerType),
      'IL': instance.iterationsLimit,
      'ILR': instance.initialLearningRate,
      'D': instance.decay,
      'DR': instance.dropRate,
      'MCU': instance.minCoefficientsUpdate,
      'L': instance.lambda,
      if (const RegularizationTypeJsonConverterNullable()
              .toJson(instance.regularizationType)
          case final value?)
        'RT': value,
      if (instance.randomSeed case final value?) 'RS': value,
      'BS': instance.batchSize,
      'FDN': instance.isFittingDataNormalized,
      'LR': const LearningRateTypeJsonConverter()
          .toJson(instance.learningRateType),
      'ICT': const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType),
      if (instance.initialCoefficients?.toJson() case final value?) 'IC': value,
      'CN': instance.targetNames.toList(),
      'FI': instance.fitIntercept,
      'IS': instance.interceptScale,
      'CBC': instance.coefficientsByClasses.toJson(),
      'DT': const DTypeJsonConverter().toJson(instance.dtype),
      'LF': linkFunctionToJson(instance.linkFunction),
      'PL': instance.positiveLabel,
      'NL': instance.negativeLabel,
      if (instance.costPerIteration case final value?) 'CPI': value,
      if (instance.schemaVersion case final value?) r'$V': value,
    };
