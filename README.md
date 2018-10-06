# MoStfmodels

**This project is under construction.**

The implementation of neural language models with mixture of softmaxes ([_Yang, Z., Dai, Z., Salakhutdinov, R., and Cohen, W. W. (2018). Breaking the softmax bottleneck: A high-rank RNN language model. In International Conference on
Learning Representations._](https://arxiv.org/abs/1711.03953)), using TensorFlow. This project is my MSc dissertation. The offical implementation in PyTorch can be found [here](https://github.com/zihangdai/mos).

This project was developed from [TensorFlow offical tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb). In this project, I added the mixture of softmaxes to the original neural language models with non-recurrent dropout. Then I optimized the hyperparameters to improve perplexities and speed up training. I also advised some techniques that have potiential to furthur improve the models. More discussions can be found in my dissertation, [_Zihan Zhang, TensorFlow applied to Neural Network Language Models_](https://dreamlh.github.io/ZihanZhang.github.io/dissertation.pdf). Please cite this paper if you would like to use the code.

Run this project by the following command in Unix/Linux terminal:

python ptb_word_lm.py --data_path=./data/ --model=small --num_gpus=0

The hyperparameters can be changed by changing the arguments in the above command. The meaning of hyperparameters are as follows.

* Initial scale: the range of weights initialization by uniform distribution.
* Learning rate: the original learning rate from the beginning of training.
* Maximum norm: the threshold for gradient clipping.
* Number of layers.
* Hidden size: the number of hidden units.
* Batch size.
* Length of sequences.
* Stage 1: the number of epochs in the first training stage.
* Stage 2: the number of epochs in the whole training stage.
* Keeping probability: the probability of keeping a unit in dropout layers.
* Learning rate decay: the decay rate in the second training stage.
* RNN mode: build LSTM using basic or block cells.
* Number of GPUs.
* Number of components: the number of Softmaxes functions in MoS layer.
Only for MoS models.

Random Search for hyperparameters can be done by using _randomSearch.py_. The best hyperparameter configuration I found is as follows.

The other techniques discussed in my dissertation can be tried by using _ptb_new_flstm.py_ and _ptb_word_wt.py_. Further optimization of hyperparameters may lead to better performance.
