ÚÃ	
ôÄ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8
~
Nadam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameNadam/output/bias/v
w
'Nadam/output/bias/v/Read/ReadVariableOpReadVariableOpNadam/output/bias/v*
_output_shapes
:*
dtype0

Nadam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameNadam/output/kernel/v

)Nadam/output/kernel/v/Read/ReadVariableOpReadVariableOpNadam/output/kernel/v*
_output_shapes

:*
dtype0

Nadam/hidden_layer_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_4/bias/v

/Nadam/hidden_layer_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_4/bias/v*
_output_shapes
:*
dtype0

Nadam/hidden_layer_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameNadam/hidden_layer_4/kernel/v

1Nadam/hidden_layer_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_4/kernel/v*
_output_shapes

:*
dtype0

Nadam/hidden_layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_3/bias/v

/Nadam/hidden_layer_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_3/bias/v*
_output_shapes
:*
dtype0

Nadam/hidden_layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameNadam/hidden_layer_3/kernel/v

1Nadam/hidden_layer_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_3/kernel/v*
_output_shapes

:*
dtype0

Nadam/hidden_layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_2/bias/v

/Nadam/hidden_layer_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_2/bias/v*
_output_shapes
:*
dtype0

Nadam/hidden_layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameNadam/hidden_layer_2/kernel/v

1Nadam/hidden_layer_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_2/kernel/v*
_output_shapes

:*
dtype0

Nadam/hidden_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_1/bias/v

/Nadam/hidden_layer_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_1/bias/v*
_output_shapes
:*
dtype0

Nadam/hidden_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*.
shared_nameNadam/hidden_layer_1/kernel/v

1Nadam/hidden_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_1/kernel/v*
_output_shapes

:
*
dtype0

Nadam/hidden_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/hidden_layer_0/bias/v

/Nadam/hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_0/bias/v*
_output_shapes
:
*
dtype0

Nadam/hidden_layer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*.
shared_nameNadam/hidden_layer_0/kernel/v

1Nadam/hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_0/kernel/v*
_output_shapes

:

*
dtype0
~
Nadam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameNadam/output/bias/m
w
'Nadam/output/bias/m/Read/ReadVariableOpReadVariableOpNadam/output/bias/m*
_output_shapes
:*
dtype0

Nadam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameNadam/output/kernel/m

)Nadam/output/kernel/m/Read/ReadVariableOpReadVariableOpNadam/output/kernel/m*
_output_shapes

:*
dtype0

Nadam/hidden_layer_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_4/bias/m

/Nadam/hidden_layer_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_4/bias/m*
_output_shapes
:*
dtype0

Nadam/hidden_layer_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameNadam/hidden_layer_4/kernel/m

1Nadam/hidden_layer_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_4/kernel/m*
_output_shapes

:*
dtype0

Nadam/hidden_layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_3/bias/m

/Nadam/hidden_layer_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_3/bias/m*
_output_shapes
:*
dtype0

Nadam/hidden_layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameNadam/hidden_layer_3/kernel/m

1Nadam/hidden_layer_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_3/kernel/m*
_output_shapes

:*
dtype0

Nadam/hidden_layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_2/bias/m

/Nadam/hidden_layer_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_2/bias/m*
_output_shapes
:*
dtype0

Nadam/hidden_layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameNadam/hidden_layer_2/kernel/m

1Nadam/hidden_layer_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_2/kernel/m*
_output_shapes

:*
dtype0

Nadam/hidden_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/hidden_layer_1/bias/m

/Nadam/hidden_layer_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_1/bias/m*
_output_shapes
:*
dtype0

Nadam/hidden_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*.
shared_nameNadam/hidden_layer_1/kernel/m

1Nadam/hidden_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_1/kernel/m*
_output_shapes

:
*
dtype0

Nadam/hidden_layer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/hidden_layer_0/bias/m

/Nadam/hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_0/bias/m*
_output_shapes
:
*
dtype0

Nadam/hidden_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*.
shared_nameNadam/hidden_layer_0/kernel/m

1Nadam/hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden_layer_0/kernel/m*
_output_shapes

:

*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
~
hidden_layer_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehidden_layer_4/bias
w
'hidden_layer_4/bias/Read/ReadVariableOpReadVariableOphidden_layer_4/bias*
_output_shapes
:*
dtype0

hidden_layer_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namehidden_layer_4/kernel

)hidden_layer_4/kernel/Read/ReadVariableOpReadVariableOphidden_layer_4/kernel*
_output_shapes

:*
dtype0
~
hidden_layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehidden_layer_3/bias
w
'hidden_layer_3/bias/Read/ReadVariableOpReadVariableOphidden_layer_3/bias*
_output_shapes
:*
dtype0

hidden_layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namehidden_layer_3/kernel

)hidden_layer_3/kernel/Read/ReadVariableOpReadVariableOphidden_layer_3/kernel*
_output_shapes

:*
dtype0
~
hidden_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehidden_layer_2/bias
w
'hidden_layer_2/bias/Read/ReadVariableOpReadVariableOphidden_layer_2/bias*
_output_shapes
:*
dtype0

hidden_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namehidden_layer_2/kernel

)hidden_layer_2/kernel/Read/ReadVariableOpReadVariableOphidden_layer_2/kernel*
_output_shapes

:*
dtype0
~
hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehidden_layer_1/bias
w
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
_output_shapes
:*
dtype0

hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_namehidden_layer_1/kernel

)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel*
_output_shapes

:
*
dtype0
~
hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namehidden_layer_0/bias
w
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
_output_shapes
:
*
dtype0

hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_namehidden_layer_0/kernel

)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel*
_output_shapes

:

*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1hidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biashidden_layer_3/kernelhidden_layer_3/biashidden_layer_4/kernelhidden_layer_4/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_39229

NoOpNoOp
R
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÍQ
valueÃQBÀQ B¹Q
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
¦
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
¦
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
¦
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
Z
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11*
Z
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11*
* 
°
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
* 
È
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_rate
Rmomentum_cachemmmm&m'm.m/m6m7m>m?mvvvv&v'v.v/v6v7v>v?v *

Sserving_default* 

0
1*

0
1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
e_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
e_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
e_
VARIABLE_VALUEhidden_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
e_
VARIABLE_VALUEhidden_layer_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
e_
VARIABLE_VALUEhidden_layer_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhidden_layer_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

|trace_0* 

}trace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

~0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUENadam/hidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUENadam/output/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUENadam/output/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/hidden_layer_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUENadam/output/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUENadam/output/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp)hidden_layer_2/kernel/Read/ReadVariableOp'hidden_layer_2/bias/Read/ReadVariableOp)hidden_layer_3/kernel/Read/ReadVariableOp'hidden_layer_3/bias/Read/ReadVariableOp)hidden_layer_4/kernel/Read/ReadVariableOp'hidden_layer_4/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Nadam/hidden_layer_0/kernel/m/Read/ReadVariableOp/Nadam/hidden_layer_0/bias/m/Read/ReadVariableOp1Nadam/hidden_layer_1/kernel/m/Read/ReadVariableOp/Nadam/hidden_layer_1/bias/m/Read/ReadVariableOp1Nadam/hidden_layer_2/kernel/m/Read/ReadVariableOp/Nadam/hidden_layer_2/bias/m/Read/ReadVariableOp1Nadam/hidden_layer_3/kernel/m/Read/ReadVariableOp/Nadam/hidden_layer_3/bias/m/Read/ReadVariableOp1Nadam/hidden_layer_4/kernel/m/Read/ReadVariableOp/Nadam/hidden_layer_4/bias/m/Read/ReadVariableOp)Nadam/output/kernel/m/Read/ReadVariableOp'Nadam/output/bias/m/Read/ReadVariableOp1Nadam/hidden_layer_0/kernel/v/Read/ReadVariableOp/Nadam/hidden_layer_0/bias/v/Read/ReadVariableOp1Nadam/hidden_layer_1/kernel/v/Read/ReadVariableOp/Nadam/hidden_layer_1/bias/v/Read/ReadVariableOp1Nadam/hidden_layer_2/kernel/v/Read/ReadVariableOp/Nadam/hidden_layer_2/bias/v/Read/ReadVariableOp1Nadam/hidden_layer_3/kernel/v/Read/ReadVariableOp/Nadam/hidden_layer_3/bias/v/Read/ReadVariableOp1Nadam/hidden_layer_4/kernel/v/Read/ReadVariableOp/Nadam/hidden_layer_4/bias/v/Read/ReadVariableOp)Nadam/output/kernel/v/Read/ReadVariableOp'Nadam/output/bias/v/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_39657
ë

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biashidden_layer_3/kernelhidden_layer_3/biashidden_layer_4/kernelhidden_layer_4/biasoutput/kerneloutput/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/hidden_layer_0/kernel/mNadam/hidden_layer_0/bias/mNadam/hidden_layer_1/kernel/mNadam/hidden_layer_1/bias/mNadam/hidden_layer_2/kernel/mNadam/hidden_layer_2/bias/mNadam/hidden_layer_3/kernel/mNadam/hidden_layer_3/bias/mNadam/hidden_layer_4/kernel/mNadam/hidden_layer_4/bias/mNadam/output/kernel/mNadam/output/bias/mNadam/hidden_layer_0/kernel/vNadam/hidden_layer_0/bias/vNadam/hidden_layer_1/kernel/vNadam/hidden_layer_1/bias/vNadam/hidden_layer_2/kernel/vNadam/hidden_layer_2/bias/vNadam/hidden_layer_3/kernel/vNadam/hidden_layer_3/bias/vNadam/hidden_layer_4/kernel/vNadam/hidden_layer_4/bias/vNadam/output/kernel/vNadam/output/bias/v*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_39805·
 

ú
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_38859

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
_
¥
__inference__traced_save_39657
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop4
0savev2_hidden_layer_3_kernel_read_readvariableop2
.savev2_hidden_layer_3_bias_read_readvariableop4
0savev2_hidden_layer_4_kernel_read_readvariableop2
.savev2_hidden_layer_4_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_nadam_hidden_layer_0_kernel_m_read_readvariableop:
6savev2_nadam_hidden_layer_0_bias_m_read_readvariableop<
8savev2_nadam_hidden_layer_1_kernel_m_read_readvariableop:
6savev2_nadam_hidden_layer_1_bias_m_read_readvariableop<
8savev2_nadam_hidden_layer_2_kernel_m_read_readvariableop:
6savev2_nadam_hidden_layer_2_bias_m_read_readvariableop<
8savev2_nadam_hidden_layer_3_kernel_m_read_readvariableop:
6savev2_nadam_hidden_layer_3_bias_m_read_readvariableop<
8savev2_nadam_hidden_layer_4_kernel_m_read_readvariableop:
6savev2_nadam_hidden_layer_4_bias_m_read_readvariableop4
0savev2_nadam_output_kernel_m_read_readvariableop2
.savev2_nadam_output_bias_m_read_readvariableop<
8savev2_nadam_hidden_layer_0_kernel_v_read_readvariableop:
6savev2_nadam_hidden_layer_0_bias_v_read_readvariableop<
8savev2_nadam_hidden_layer_1_kernel_v_read_readvariableop:
6savev2_nadam_hidden_layer_1_bias_v_read_readvariableop<
8savev2_nadam_hidden_layer_2_kernel_v_read_readvariableop:
6savev2_nadam_hidden_layer_2_bias_v_read_readvariableop<
8savev2_nadam_hidden_layer_3_kernel_v_read_readvariableop:
6savev2_nadam_hidden_layer_3_bias_v_read_readvariableop<
8savev2_nadam_hidden_layer_4_kernel_v_read_readvariableop:
6savev2_nadam_hidden_layer_4_bias_v_read_readvariableop4
0savev2_nadam_output_kernel_v_read_readvariableop2
.savev2_nadam_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ø
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
value÷Bô/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHË
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B à
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop0savev2_hidden_layer_3_kernel_read_readvariableop.savev2_hidden_layer_3_bias_read_readvariableop0savev2_hidden_layer_4_kernel_read_readvariableop.savev2_hidden_layer_4_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_nadam_hidden_layer_0_kernel_m_read_readvariableop6savev2_nadam_hidden_layer_0_bias_m_read_readvariableop8savev2_nadam_hidden_layer_1_kernel_m_read_readvariableop6savev2_nadam_hidden_layer_1_bias_m_read_readvariableop8savev2_nadam_hidden_layer_2_kernel_m_read_readvariableop6savev2_nadam_hidden_layer_2_bias_m_read_readvariableop8savev2_nadam_hidden_layer_3_kernel_m_read_readvariableop6savev2_nadam_hidden_layer_3_bias_m_read_readvariableop8savev2_nadam_hidden_layer_4_kernel_m_read_readvariableop6savev2_nadam_hidden_layer_4_bias_m_read_readvariableop0savev2_nadam_output_kernel_m_read_readvariableop.savev2_nadam_output_bias_m_read_readvariableop8savev2_nadam_hidden_layer_0_kernel_v_read_readvariableop6savev2_nadam_hidden_layer_0_bias_v_read_readvariableop8savev2_nadam_hidden_layer_1_kernel_v_read_readvariableop6savev2_nadam_hidden_layer_1_bias_v_read_readvariableop8savev2_nadam_hidden_layer_2_kernel_v_read_readvariableop6savev2_nadam_hidden_layer_2_bias_v_read_readvariableop8savev2_nadam_hidden_layer_3_kernel_v_read_readvariableop6savev2_nadam_hidden_layer_3_bias_v_read_readvariableop8savev2_nadam_hidden_layer_4_kernel_v_read_readvariableop6savev2_nadam_hidden_layer_4_bias_v_read_readvariableop0savev2_nadam_output_kernel_v_read_readvariableop.savev2_nadam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Í
_input_shapes»
¸: :

:
:
:::::::::: : : : : : : : : : :

:
:
::::::::::

:
:
:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:

: $

_output_shapes
:
:$% 

_output_shapes

:
: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

:: .

_output_shapes
::/

_output_shapes
: 
Ì

.__inference_hidden_layer_4_layer_call_fn_39466

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_38893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_39437

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_38825

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
9


E__inference_model_name_layer_call_and_return_conditional_losses_39377

inputs?
-hidden_layer_0_matmul_readvariableop_resource:

<
.hidden_layer_0_biasadd_readvariableop_resource:
?
-hidden_layer_1_matmul_readvariableop_resource:
<
.hidden_layer_1_biasadd_readvariableop_resource:?
-hidden_layer_2_matmul_readvariableop_resource:<
.hidden_layer_2_biasadd_readvariableop_resource:?
-hidden_layer_3_matmul_readvariableop_resource:<
.hidden_layer_3_biasadd_readvariableop_resource:?
-hidden_layer_4_matmul_readvariableop_resource:<
.hidden_layer_4_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity¢%hidden_layer_0/BiasAdd/ReadVariableOp¢$hidden_layer_0/MatMul/ReadVariableOp¢%hidden_layer_1/BiasAdd/ReadVariableOp¢$hidden_layer_1/MatMul/ReadVariableOp¢%hidden_layer_2/BiasAdd/ReadVariableOp¢$hidden_layer_2/MatMul/ReadVariableOp¢%hidden_layer_3/BiasAdd/ReadVariableOp¢$hidden_layer_3/MatMul/ReadVariableOp¢%hidden_layer_4/BiasAdd/ReadVariableOp¢$hidden_layer_4/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
hidden_layer_0/MatMulMatMulinputs,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0£
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¢
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
hidden_layer_2/MatMulMatMul!hidden_layer_1/Relu:activations:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
hidden_layer_3/MatMulMatMul!hidden_layer_2/Relu:activations:0,hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_3/BiasAddBiasAddhidden_layer_3/MatMul:product:0-hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_3/ReluReluhidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hidden_layer_4/MatMul/ReadVariableOpReadVariableOp-hidden_layer_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
hidden_layer_4/MatMulMatMul!hidden_layer_3/Relu:activations:0,hidden_layer_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_4/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_4/BiasAddBiasAddhidden_layer_4/MatMul:product:0-hidden_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_4/ReluReluhidden_layer_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output/MatMulMatMul!hidden_layer_4/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp&^hidden_layer_3/BiasAdd/ReadVariableOp%^hidden_layer_3/MatMul/ReadVariableOp&^hidden_layer_4/BiasAdd/ReadVariableOp%^hidden_layer_4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2N
%hidden_layer_3/BiasAdd/ReadVariableOp%hidden_layer_3/BiasAdd/ReadVariableOp2L
$hidden_layer_3/MatMul/ReadVariableOp$hidden_layer_3/MatMul/ReadVariableOp2N
%hidden_layer_4/BiasAdd/ReadVariableOp%hidden_layer_4/BiasAdd/ReadVariableOp2L
$hidden_layer_4/MatMul/ReadVariableOp$hidden_layer_4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì

.__inference_hidden_layer_1_layer_call_fn_39406

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_38842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_38842

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_39417

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
üC
ü
 __inference__wrapped_model_38807
input_1J
8model_name_hidden_layer_0_matmul_readvariableop_resource:

G
9model_name_hidden_layer_0_biasadd_readvariableop_resource:
J
8model_name_hidden_layer_1_matmul_readvariableop_resource:
G
9model_name_hidden_layer_1_biasadd_readvariableop_resource:J
8model_name_hidden_layer_2_matmul_readvariableop_resource:G
9model_name_hidden_layer_2_biasadd_readvariableop_resource:J
8model_name_hidden_layer_3_matmul_readvariableop_resource:G
9model_name_hidden_layer_3_biasadd_readvariableop_resource:J
8model_name_hidden_layer_4_matmul_readvariableop_resource:G
9model_name_hidden_layer_4_biasadd_readvariableop_resource:B
0model_name_output_matmul_readvariableop_resource:?
1model_name_output_biasadd_readvariableop_resource:
identity¢0model_name/hidden_layer_0/BiasAdd/ReadVariableOp¢/model_name/hidden_layer_0/MatMul/ReadVariableOp¢0model_name/hidden_layer_1/BiasAdd/ReadVariableOp¢/model_name/hidden_layer_1/MatMul/ReadVariableOp¢0model_name/hidden_layer_2/BiasAdd/ReadVariableOp¢/model_name/hidden_layer_2/MatMul/ReadVariableOp¢0model_name/hidden_layer_3/BiasAdd/ReadVariableOp¢/model_name/hidden_layer_3/MatMul/ReadVariableOp¢0model_name/hidden_layer_4/BiasAdd/ReadVariableOp¢/model_name/hidden_layer_4/MatMul/ReadVariableOp¢(model_name/output/BiasAdd/ReadVariableOp¢'model_name/output/MatMul/ReadVariableOp¨
/model_name/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8model_name_hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
 model_name/hidden_layer_0/MatMulMatMulinput_17model_name/hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
0model_name/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9model_name_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
!model_name/hidden_layer_0/BiasAddBiasAdd*model_name/hidden_layer_0/MatMul:product:08model_name/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

model_name/hidden_layer_0/ReluRelu*model_name/hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
/model_name/hidden_layer_1/MatMul/ReadVariableOpReadVariableOp8model_name_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ã
 model_name/hidden_layer_1/MatMulMatMul,model_name/hidden_layer_0/Relu:activations:07model_name/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0model_name/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp9model_name_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!model_name/hidden_layer_1/BiasAddBiasAdd*model_name/hidden_layer_1/MatMul:product:08model_name/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_name/hidden_layer_1/ReluRelu*model_name/hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/model_name/hidden_layer_2/MatMul/ReadVariableOpReadVariableOp8model_name_hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
 model_name/hidden_layer_2/MatMulMatMul,model_name/hidden_layer_1/Relu:activations:07model_name/hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0model_name/hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp9model_name_hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!model_name/hidden_layer_2/BiasAddBiasAdd*model_name/hidden_layer_2/MatMul:product:08model_name/hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_name/hidden_layer_2/ReluRelu*model_name/hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/model_name/hidden_layer_3/MatMul/ReadVariableOpReadVariableOp8model_name_hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
 model_name/hidden_layer_3/MatMulMatMul,model_name/hidden_layer_2/Relu:activations:07model_name/hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0model_name/hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp9model_name_hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!model_name/hidden_layer_3/BiasAddBiasAdd*model_name/hidden_layer_3/MatMul:product:08model_name/hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_name/hidden_layer_3/ReluRelu*model_name/hidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/model_name/hidden_layer_4/MatMul/ReadVariableOpReadVariableOp8model_name_hidden_layer_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
 model_name/hidden_layer_4/MatMulMatMul,model_name/hidden_layer_3/Relu:activations:07model_name/hidden_layer_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0model_name/hidden_layer_4/BiasAdd/ReadVariableOpReadVariableOp9model_name_hidden_layer_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!model_name/hidden_layer_4/BiasAddBiasAdd*model_name/hidden_layer_4/MatMul:product:08model_name/hidden_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_name/hidden_layer_4/ReluRelu*model_name/hidden_layer_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_name/output/MatMul/ReadVariableOpReadVariableOp0model_name_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0³
model_name/output/MatMulMatMul,model_name/hidden_layer_4/Relu:activations:0/model_name/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model_name/output/BiasAdd/ReadVariableOpReadVariableOp1model_name_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
model_name/output/BiasAddBiasAdd"model_name/output/MatMul:product:00model_name/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
IdentityIdentity"model_name/output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^model_name/hidden_layer_0/BiasAdd/ReadVariableOp0^model_name/hidden_layer_0/MatMul/ReadVariableOp1^model_name/hidden_layer_1/BiasAdd/ReadVariableOp0^model_name/hidden_layer_1/MatMul/ReadVariableOp1^model_name/hidden_layer_2/BiasAdd/ReadVariableOp0^model_name/hidden_layer_2/MatMul/ReadVariableOp1^model_name/hidden_layer_3/BiasAdd/ReadVariableOp0^model_name/hidden_layer_3/MatMul/ReadVariableOp1^model_name/hidden_layer_4/BiasAdd/ReadVariableOp0^model_name/hidden_layer_4/MatMul/ReadVariableOp)^model_name/output/BiasAdd/ReadVariableOp(^model_name/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2d
0model_name/hidden_layer_0/BiasAdd/ReadVariableOp0model_name/hidden_layer_0/BiasAdd/ReadVariableOp2b
/model_name/hidden_layer_0/MatMul/ReadVariableOp/model_name/hidden_layer_0/MatMul/ReadVariableOp2d
0model_name/hidden_layer_1/BiasAdd/ReadVariableOp0model_name/hidden_layer_1/BiasAdd/ReadVariableOp2b
/model_name/hidden_layer_1/MatMul/ReadVariableOp/model_name/hidden_layer_1/MatMul/ReadVariableOp2d
0model_name/hidden_layer_2/BiasAdd/ReadVariableOp0model_name/hidden_layer_2/BiasAdd/ReadVariableOp2b
/model_name/hidden_layer_2/MatMul/ReadVariableOp/model_name/hidden_layer_2/MatMul/ReadVariableOp2d
0model_name/hidden_layer_3/BiasAdd/ReadVariableOp0model_name/hidden_layer_3/BiasAdd/ReadVariableOp2b
/model_name/hidden_layer_3/MatMul/ReadVariableOp/model_name/hidden_layer_3/MatMul/ReadVariableOp2d
0model_name/hidden_layer_4/BiasAdd/ReadVariableOp0model_name/hidden_layer_4/BiasAdd/ReadVariableOp2b
/model_name/hidden_layer_4/MatMul/ReadVariableOp/model_name/hidden_layer_4/MatMul/ReadVariableOp2T
(model_name/output/BiasAdd/ReadVariableOp(model_name/output/BiasAdd/ReadVariableOp2R
'model_name/output/MatMul/ReadVariableOp'model_name/output/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
í

§
*__inference_model_name_layer_call_fn_39124
input_1
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_name_layer_call_and_return_conditional_losses_39068o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ì

.__inference_hidden_layer_0_layer_call_fn_39386

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_38825o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_39457

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_39397

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò"

E__inference_model_name_layer_call_and_return_conditional_losses_39158
input_1&
hidden_layer_0_39127:

"
hidden_layer_0_39129:
&
hidden_layer_1_39132:
"
hidden_layer_1_39134:&
hidden_layer_2_39137:"
hidden_layer_2_39139:&
hidden_layer_3_39142:"
hidden_layer_3_39144:&
hidden_layer_4_39147:"
hidden_layer_4_39149:
output_39152:
output_39154:
identity¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢&hidden_layer_2/StatefulPartitionedCall¢&hidden_layer_3/StatefulPartitionedCall¢&hidden_layer_4/StatefulPartitionedCall¢output/StatefulPartitionedCall
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_1hidden_layer_0_39127hidden_layer_0_39129*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_38825®
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_39132hidden_layer_1_39134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_38842®
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_39137hidden_layer_2_39139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_38859®
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_39142hidden_layer_3_39144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_38876®
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0hidden_layer_4_39147hidden_layer_4_39149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_38893
output/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0output_39152output_39154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_38909v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ò"

E__inference_model_name_layer_call_and_return_conditional_losses_39192
input_1&
hidden_layer_0_39161:

"
hidden_layer_0_39163:
&
hidden_layer_1_39166:
"
hidden_layer_1_39168:&
hidden_layer_2_39171:"
hidden_layer_2_39173:&
hidden_layer_3_39176:"
hidden_layer_3_39178:&
hidden_layer_4_39181:"
hidden_layer_4_39183:
output_39186:
output_39188:
identity¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢&hidden_layer_2/StatefulPartitionedCall¢&hidden_layer_3/StatefulPartitionedCall¢&hidden_layer_4/StatefulPartitionedCall¢output/StatefulPartitionedCall
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_1hidden_layer_0_39161hidden_layer_0_39163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_38825®
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_39166hidden_layer_1_39168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_38842®
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_39171hidden_layer_2_39173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_38859®
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_39176hidden_layer_3_39178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_38876®
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0hidden_layer_4_39181hidden_layer_4_39183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_38893
output/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0output_39186output_39188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_38909v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ê

¦
*__inference_model_name_layer_call_fn_39258

inputs
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_name_layer_call_and_return_conditional_losses_38916o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_38893

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

§
*__inference_model_name_layer_call_fn_38943
input_1
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_name_layer_call_and_return_conditional_losses_38916o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ä	
ò
A__inference_output_layer_call_and_return_conditional_losses_39496

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_38876

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï"

E__inference_model_name_layer_call_and_return_conditional_losses_38916

inputs&
hidden_layer_0_38826:

"
hidden_layer_0_38828:
&
hidden_layer_1_38843:
"
hidden_layer_1_38845:&
hidden_layer_2_38860:"
hidden_layer_2_38862:&
hidden_layer_3_38877:"
hidden_layer_3_38879:&
hidden_layer_4_38894:"
hidden_layer_4_38896:
output_38910:
output_38912:
identity¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢&hidden_layer_2/StatefulPartitionedCall¢&hidden_layer_3/StatefulPartitionedCall¢&hidden_layer_4/StatefulPartitionedCall¢output/StatefulPartitionedCall
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_38826hidden_layer_0_38828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_38825®
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_38843hidden_layer_1_38845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_38842®
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_38860hidden_layer_2_38862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_38859®
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_38877hidden_layer_3_38879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_38876®
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0hidden_layer_4_38894hidden_layer_4_38896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_38893
output/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0output_38910output_38912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_38909v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
 

ú
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_39477

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

¦
*__inference_model_name_layer_call_fn_39287

inputs
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_name_layer_call_and_return_conditional_losses_39068o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ï"

E__inference_model_name_layer_call_and_return_conditional_losses_39068

inputs&
hidden_layer_0_39037:

"
hidden_layer_0_39039:
&
hidden_layer_1_39042:
"
hidden_layer_1_39044:&
hidden_layer_2_39047:"
hidden_layer_2_39049:&
hidden_layer_3_39052:"
hidden_layer_3_39054:&
hidden_layer_4_39057:"
hidden_layer_4_39059:
output_39062:
output_39064:
identity¢&hidden_layer_0/StatefulPartitionedCall¢&hidden_layer_1/StatefulPartitionedCall¢&hidden_layer_2/StatefulPartitionedCall¢&hidden_layer_3/StatefulPartitionedCall¢&hidden_layer_4/StatefulPartitionedCall¢output/StatefulPartitionedCall
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_39037hidden_layer_0_39039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_38825®
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_39042hidden_layer_1_39044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_38842®
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_39047hidden_layer_2_39049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_38859®
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_39052hidden_layer_3_39054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_38876®
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0hidden_layer_4_39057hidden_layer_4_39059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_38893
output/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0output_39062output_39064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_38909v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì

.__inference_hidden_layer_3_layer_call_fn_39446

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_38876o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9


E__inference_model_name_layer_call_and_return_conditional_losses_39332

inputs?
-hidden_layer_0_matmul_readvariableop_resource:

<
.hidden_layer_0_biasadd_readvariableop_resource:
?
-hidden_layer_1_matmul_readvariableop_resource:
<
.hidden_layer_1_biasadd_readvariableop_resource:?
-hidden_layer_2_matmul_readvariableop_resource:<
.hidden_layer_2_biasadd_readvariableop_resource:?
-hidden_layer_3_matmul_readvariableop_resource:<
.hidden_layer_3_biasadd_readvariableop_resource:?
-hidden_layer_4_matmul_readvariableop_resource:<
.hidden_layer_4_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity¢%hidden_layer_0/BiasAdd/ReadVariableOp¢$hidden_layer_0/MatMul/ReadVariableOp¢%hidden_layer_1/BiasAdd/ReadVariableOp¢$hidden_layer_1/MatMul/ReadVariableOp¢%hidden_layer_2/BiasAdd/ReadVariableOp¢$hidden_layer_2/MatMul/ReadVariableOp¢%hidden_layer_3/BiasAdd/ReadVariableOp¢$hidden_layer_3/MatMul/ReadVariableOp¢%hidden_layer_4/BiasAdd/ReadVariableOp¢$hidden_layer_4/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
hidden_layer_0/MatMulMatMulinputs,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0£
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¢
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
hidden_layer_2/MatMulMatMul!hidden_layer_1/Relu:activations:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
hidden_layer_3/MatMulMatMul!hidden_layer_2/Relu:activations:0,hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_3/BiasAddBiasAddhidden_layer_3/MatMul:product:0-hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_3/ReluReluhidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hidden_layer_4/MatMul/ReadVariableOpReadVariableOp-hidden_layer_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
hidden_layer_4/MatMulMatMul!hidden_layer_3/Relu:activations:0,hidden_layer_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hidden_layer_4/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
hidden_layer_4/BiasAddBiasAddhidden_layer_4/MatMul:product:0-hidden_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
hidden_layer_4/ReluReluhidden_layer_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
output/MatMulMatMul!hidden_layer_4/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp&^hidden_layer_3/BiasAdd/ReadVariableOp%^hidden_layer_3/MatMul/ReadVariableOp&^hidden_layer_4/BiasAdd/ReadVariableOp%^hidden_layer_4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2N
%hidden_layer_3/BiasAdd/ReadVariableOp%hidden_layer_3/BiasAdd/ReadVariableOp2L
$hidden_layer_3/MatMul/ReadVariableOp$hidden_layer_3/MatMul/ReadVariableOp2N
%hidden_layer_4/BiasAdd/ReadVariableOp%hidden_layer_4/BiasAdd/ReadVariableOp2L
$hidden_layer_4/MatMul/ReadVariableOp$hidden_layer_4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Á

 
#__inference_signature_wrapper_39229
input_1
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_38807o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ì

.__inference_hidden_layer_2_layer_call_fn_39426

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_38859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

&__inference_output_layer_call_fn_39486

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_38909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë¹

!__inference__traced_restore_39805
file_prefix8
&assignvariableop_hidden_layer_0_kernel:

4
&assignvariableop_1_hidden_layer_0_bias:
:
(assignvariableop_2_hidden_layer_1_kernel:
4
&assignvariableop_3_hidden_layer_1_bias::
(assignvariableop_4_hidden_layer_2_kernel:4
&assignvariableop_5_hidden_layer_2_bias::
(assignvariableop_6_hidden_layer_3_kernel:4
&assignvariableop_7_hidden_layer_3_bias::
(assignvariableop_8_hidden_layer_4_kernel:4
&assignvariableop_9_hidden_layer_4_bias:3
!assignvariableop_10_output_kernel:-
assignvariableop_11_output_bias:(
assignvariableop_12_nadam_iter:	 *
 assignvariableop_13_nadam_beta_1: *
 assignvariableop_14_nadam_beta_2: )
assignvariableop_15_nadam_decay: 1
'assignvariableop_16_nadam_learning_rate: 2
(assignvariableop_17_nadam_momentum_cache: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: #
assignvariableop_20_total: #
assignvariableop_21_count: C
1assignvariableop_22_nadam_hidden_layer_0_kernel_m:

=
/assignvariableop_23_nadam_hidden_layer_0_bias_m:
C
1assignvariableop_24_nadam_hidden_layer_1_kernel_m:
=
/assignvariableop_25_nadam_hidden_layer_1_bias_m:C
1assignvariableop_26_nadam_hidden_layer_2_kernel_m:=
/assignvariableop_27_nadam_hidden_layer_2_bias_m:C
1assignvariableop_28_nadam_hidden_layer_3_kernel_m:=
/assignvariableop_29_nadam_hidden_layer_3_bias_m:C
1assignvariableop_30_nadam_hidden_layer_4_kernel_m:=
/assignvariableop_31_nadam_hidden_layer_4_bias_m:;
)assignvariableop_32_nadam_output_kernel_m:5
'assignvariableop_33_nadam_output_bias_m:C
1assignvariableop_34_nadam_hidden_layer_0_kernel_v:

=
/assignvariableop_35_nadam_hidden_layer_0_bias_v:
C
1assignvariableop_36_nadam_hidden_layer_1_kernel_v:
=
/assignvariableop_37_nadam_hidden_layer_1_bias_v:C
1assignvariableop_38_nadam_hidden_layer_2_kernel_v:=
/assignvariableop_39_nadam_hidden_layer_2_bias_v:C
1assignvariableop_40_nadam_hidden_layer_3_kernel_v:=
/assignvariableop_41_nadam_hidden_layer_3_bias_v:C
1assignvariableop_42_nadam_hidden_layer_4_kernel_v:=
/assignvariableop_43_nadam_hidden_layer_4_bias_v:;
)assignvariableop_44_nadam_output_kernel_v:5
'assignvariableop_45_nadam_output_bias_v:
identity_47¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Û
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
value÷Bô/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp(assignvariableop_6_hidden_layer_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp&assignvariableop_7_hidden_layer_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp(assignvariableop_8_hidden_layer_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp&assignvariableop_9_hidden_layer_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp!assignvariableop_10_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_nadam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_nadam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp assignvariableop_14_nadam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_nadam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_nadam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_nadam_momentum_cacheIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_22AssignVariableOp1assignvariableop_22_nadam_hidden_layer_0_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_23AssignVariableOp/assignvariableop_23_nadam_hidden_layer_0_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_24AssignVariableOp1assignvariableop_24_nadam_hidden_layer_1_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_25AssignVariableOp/assignvariableop_25_nadam_hidden_layer_1_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_26AssignVariableOp1assignvariableop_26_nadam_hidden_layer_2_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_27AssignVariableOp/assignvariableop_27_nadam_hidden_layer_2_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_28AssignVariableOp1assignvariableop_28_nadam_hidden_layer_3_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_29AssignVariableOp/assignvariableop_29_nadam_hidden_layer_3_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_30AssignVariableOp1assignvariableop_30_nadam_hidden_layer_4_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_31AssignVariableOp/assignvariableop_31_nadam_hidden_layer_4_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_nadam_output_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp'assignvariableop_33_nadam_output_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_34AssignVariableOp1assignvariableop_34_nadam_hidden_layer_0_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_35AssignVariableOp/assignvariableop_35_nadam_hidden_layer_0_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_36AssignVariableOp1assignvariableop_36_nadam_hidden_layer_1_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_37AssignVariableOp/assignvariableop_37_nadam_hidden_layer_1_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_38AssignVariableOp1assignvariableop_38_nadam_hidden_layer_2_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_39AssignVariableOp/assignvariableop_39_nadam_hidden_layer_2_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_40AssignVariableOp1assignvariableop_40_nadam_hidden_layer_3_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_41AssignVariableOp/assignvariableop_41_nadam_hidden_layer_3_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_42AssignVariableOp1assignvariableop_42_nadam_hidden_layer_4_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_43AssignVariableOp/assignvariableop_43_nadam_hidden_layer_4_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp)assignvariableop_44_nadam_output_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp'assignvariableop_45_nadam_output_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ã
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: °
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_47Identity_47:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ä	
ò
A__inference_output_layer_call_and_return_conditional_losses_38909

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*©
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
:
output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:û¨
Ð
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
»
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
»
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
»
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
v
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11"
trackable_list_wrapper
v
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ý
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32ò
*__inference_model_name_layer_call_fn_38943
*__inference_model_name_layer_call_fn_39258
*__inference_model_name_layer_call_fn_39287
*__inference_model_name_layer_call_fn_39124¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
É
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32Þ
E__inference_model_name_layer_call_and_return_conditional_losses_39332
E__inference_model_name_layer_call_and_return_conditional_losses_39377
E__inference_model_name_layer_call_and_return_conditional_losses_39158
E__inference_model_name_layer_call_and_return_conditional_losses_39192¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
ËBÈ
 __inference__wrapped_model_38807input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_rate
Rmomentum_cachemmmm&m'm.m/m6m7m>m?mvvvv&v'v.v/v6v7v>v?v "
	optimizer
,
Sserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò
Ytrace_02Õ
.__inference_hidden_layer_0_layer_call_fn_39386¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zYtrace_0

Ztrace_02ð
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_39397¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zZtrace_0
':%

2hidden_layer_0/kernel
!:
2hidden_layer_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò
`trace_02Õ
.__inference_hidden_layer_1_layer_call_fn_39406¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z`trace_0

atrace_02ð
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_39417¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0
':%
2hidden_layer_1/kernel
!:2hidden_layer_1/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ò
gtrace_02Õ
.__inference_hidden_layer_2_layer_call_fn_39426¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zgtrace_0

htrace_02ð
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_39437¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zhtrace_0
':%2hidden_layer_2/kernel
!:2hidden_layer_2/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ò
ntrace_02Õ
.__inference_hidden_layer_3_layer_call_fn_39446¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zntrace_0

otrace_02ð
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_39457¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0
':%2hidden_layer_3/kernel
!:2hidden_layer_3/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ò
utrace_02Õ
.__inference_hidden_layer_4_layer_call_fn_39466¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zutrace_0

vtrace_02ð
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_39477¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0
':%2hidden_layer_4/kernel
!:2hidden_layer_4/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ê
|trace_02Í
&__inference_output_layer_call_fn_39486¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z|trace_0

}trace_02è
A__inference_output_layer_call_and_return_conditional_losses_39496¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z}trace_0
:2output/kernel
:2output/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
*__inference_model_name_layer_call_fn_38943input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_model_name_layer_call_fn_39258inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_model_name_layer_call_fn_39287inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
*__inference_model_name_layer_call_fn_39124input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_name_layer_call_and_return_conditional_losses_39332inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_name_layer_call_and_return_conditional_losses_39377inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_name_layer_call_and_return_conditional_losses_39158input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_model_name_layer_call_and_return_conditional_losses_39192input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
ÊBÇ
#__inference_signature_wrapper_39229input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_hidden_layer_0_layer_call_fn_39386inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_39397inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_hidden_layer_1_layer_call_fn_39406inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_39417inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_hidden_layer_2_layer_call_fn_39426inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_39437inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_hidden_layer_3_layer_call_fn_39446inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_39457inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_hidden_layer_4_layer_call_fn_39466inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_39477inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_output_layer_call_fn_39486inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_output_layer_call_and_return_conditional_losses_39496inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
-:+

2Nadam/hidden_layer_0/kernel/m
':%
2Nadam/hidden_layer_0/bias/m
-:+
2Nadam/hidden_layer_1/kernel/m
':%2Nadam/hidden_layer_1/bias/m
-:+2Nadam/hidden_layer_2/kernel/m
':%2Nadam/hidden_layer_2/bias/m
-:+2Nadam/hidden_layer_3/kernel/m
':%2Nadam/hidden_layer_3/bias/m
-:+2Nadam/hidden_layer_4/kernel/m
':%2Nadam/hidden_layer_4/bias/m
%:#2Nadam/output/kernel/m
:2Nadam/output/bias/m
-:+

2Nadam/hidden_layer_0/kernel/v
':%
2Nadam/hidden_layer_0/bias/v
-:+
2Nadam/hidden_layer_1/kernel/v
':%2Nadam/hidden_layer_1/bias/v
-:+2Nadam/hidden_layer_2/kernel/v
':%2Nadam/hidden_layer_2/bias/v
-:+2Nadam/hidden_layer_3/kernel/v
':%2Nadam/hidden_layer_3/bias/v
-:+2Nadam/hidden_layer_4/kernel/v
':%2Nadam/hidden_layer_4/bias/v
%:#2Nadam/output/kernel/v
:2Nadam/output/bias/v
 __inference__wrapped_model_38807q&'./67>?0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ©
I__inference_hidden_layer_0_layer_call_and_return_conditional_losses_39397\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_hidden_layer_0_layer_call_fn_39386O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
©
I__inference_hidden_layer_1_layer_call_and_return_conditional_losses_39417\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_hidden_layer_1_layer_call_fn_39406O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_hidden_layer_2_layer_call_and_return_conditional_losses_39437\&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_hidden_layer_2_layer_call_fn_39426O&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_hidden_layer_3_layer_call_and_return_conditional_losses_39457\.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_hidden_layer_3_layer_call_fn_39446O.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_hidden_layer_4_layer_call_and_return_conditional_losses_39477\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_hidden_layer_4_layer_call_fn_39466O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¸
E__inference_model_name_layer_call_and_return_conditional_losses_39158o&'./67>?8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
E__inference_model_name_layer_call_and_return_conditional_losses_39192o&'./67>?8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_model_name_layer_call_and_return_conditional_losses_39332n&'./67>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_model_name_layer_call_and_return_conditional_losses_39377n&'./67>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_model_name_layer_call_fn_38943b&'./67>?8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_name_layer_call_fn_39124b&'./67>?8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_name_layer_call_fn_39258a&'./67>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_name_layer_call_fn_39287a&'./67>?7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_output_layer_call_and_return_conditional_losses_39496\>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_output_layer_call_fn_39486O>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
#__inference_signature_wrapper_39229|&'./67>?;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
"/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ