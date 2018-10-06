# MoStfmodels

The implementation of neural language models with **mixture of softmaxes** ([_Yang, Z., Dai, Z., Salakhutdinov, R., and Cohen, W. W. (2018). Breaking the softmax bottleneck: A high-rank RNN language model. In International Conference on
Learning Representations._](https://arxiv.org/abs/1711.03953)), using **TensorFlow**. This project is my MSc dissertation. The offical implementation in PyTorch can be found [here](https://github.com/zihangdai/mos).

This project was developed from [TensorFlow offical tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb). In this project, I added the mixture of softmaxes to the original neural language models with non-recurrent dropout. Then I optimized the hyperparameters to improve perplexities and speed up training. I also advised some techniques that have potiential to furthur improve the models. More discussions can be found in my dissertation, [_Zihan Zhang, TensorFlow applied to Neural Network Language Models_](https://dreamlh.github.io/ZihanZhang.github.io/dissertation.pdf). Please cite this paper if you would like to use the code.

Run this project by the following command in Unix/Linux terminal with the best hyperparameter configuration I found (i.e. the BMLG model in my dissertation):

_python3 ../mos/ptb_word_lm.py --data_path=./data/ --model=random --init_scale=0.02 --use_adam=False --learning_rate=20 --max_grad_norm=0.3 --num_layers=3 --num_steps=40 --hidden_size=900 --max_epoch=34 --max_max_epoch=100 --keep_prob=0.4411 --lr_decay=0.8 --batch_size=28 --vocab_size=10000 --rnn_mode=block --use_dynamic=False --n_experts=5 --num_gpus=1 --test_when_training=False_

The validation perplexity of this model is 79.440, which is lower than the large model (82.486) in TensorFlow tutorial. In addition, the training time of the former is 4.175 hours by one NVIDIA GTX 1060 GPU, which is shorter than the latter (5.050 hours).

The hyperparameters can be changed by changing the arguments in the above command. Random Search for hyperparameters can be done by using _randomSearch.py_. The other techniques discussed in my dissertation can be tried by using _ptb_new_flstm.py_ and _ptb_word_wt.py_. Further optimization of hyperparameters may lead to better performance.
