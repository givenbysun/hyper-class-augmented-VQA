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

1. M. Alizadeh and B. Di Eugenio. (2020). Incorporating Verb Semantic Information
in Visual Question Answering through Multitask Learning Paradigm. Accepted in the
International Journal of Semantic Computing (IJSC).
https://www.aclweb.org/anthology/2020.lrec-1.678.pdf

2. M. Alizadeh and B. Di Eugenio. (2020). Augmenting Visual Question Answering with
Semantic Frame Information in a Multitask Learning Approach. In Proceedings of the
14th IEEE International Conference on Semantic Computing (ICSC), San Diego, CA,
February 3-5. (Nominated for Best Paper Award)
https://arxiv.org/pdf/2001.11673.pdf 
