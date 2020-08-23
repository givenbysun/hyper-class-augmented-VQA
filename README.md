# hyper-class-augmented-VQA

Visual Question Answering (VQA) concerns providing answers to Natural Language questions about
images. Several deep neural network approaches have been proposed to model the task in an end-
to-end fashion. Whereas the task is grounded in visual processing, if the question focuses on events
described by verbs, the language understanding component becomes crucial. Our hypothesis is that
models should be aware of verb semantics, as expressed via semantic role labels, argument types,
and/or frame elements. Unfortunately, no VQA dataset exists that includes verb semantic information.
Our first contribution is a new VQA dataset (imSituVQA) that we built by taking advantage of the
imSitu annotations. The imSitu dataset consists of images manually labeled with semantic frame ele-
ments, mostly taken from FrameNet. Second, we propose a multi-task CNN-LSTM VQA model that
learns to classify the answers as well as the semantic frame elements. Our experiments show that se-
mantic frame element classification helps the VQA system avoid inconsistent responses and improves
performance. Third, we employ an automatic semantic role labeler and annotate a subset of the VQA
dataset (VQA sub ). This way, the proposed multi-task CNN-LSTM VQA model can be trained with the
VQA sub as well. The results show a slight improvement over the single-task CNN-LSTM model.
