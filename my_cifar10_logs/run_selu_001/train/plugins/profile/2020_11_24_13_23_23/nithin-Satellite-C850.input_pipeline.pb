	�HM���!@�HM���!@!�HM���!@	s~�vts@s~�vts@!s~�vts@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�HM���!@ W�c#�?A��EB[� @Y�/o��?*	H�z�n@2F
Iterator::ModelS\U�]�?!���k�Q@)���F���?1��ML $K@:Preprocessing2U
Iterator::Model::ParallelMapV2�m½2o�?!�ȓeh1@)�m½2o�?1�ȓeh1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatod���?!��jCs,@)X��Iؗ?1X���]#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�AҧU��?!IuV�Vt#@)d �.���?17��B��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���uR_�?!ef=�h+@)���uR_�?1ef=�h+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!Y�n�?!$�Q�O<@)��4�ׂ~?1G�;T��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices����{?!�օ6�@)s����{?1�օ6�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/o�j�?!U�d�%@)nm�y��h?1c�I��-�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9t~�vts@#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 W�c#�? W�c#�?! W�c#�?      ��!       "      ��!       *      ��!       2	��EB[� @��EB[� @!��EB[� @:      ��!       B      ��!       J	�/o��?�/o��?!�/o��?R      ��!       Z	�/o��?�/o��?!�/o��?JCPU_ONLYYt~�vts@b 