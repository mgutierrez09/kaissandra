# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 04:29:36 2019

@author: mgutierrez
"""
import numpy as np
from threading import Thread
import tensorflow as tf
import pickle
import time
import h5py
import datetime as dt
import os

from kaissandra.local_config import local_vars
from kaissandra.config import write_log
from kaissandra.results2 import (get_last_saved_epoch2,
                                 get_results_mg,
                                 init_results_mg_dir,
                                 get_results_meta,
                                 get_performance_meta,
                                 save_costs,
                                 init_results_dir,
                                 get_results)

class Model:
    """ Parent object reprsening a model """
    def __init__(self):
        pass
    def fit(self):
        pass
    def cv(self):
        pass
    def test(self):
        pass
    def _build_model(self):
        pass
    def compute_loss(self):
        pass
    def predict(self):
        pass
    def evaluate(self):
        pass
    def _save_graph(self, saver, sess, ID, epoch_cost, epoch, weights_directory):
        """ <DocString> """
        if os.path.exists(weights_directory+ID+"/")==False:
            os.mkdir(weights_directory+ID+"/")
        strEpoch = "{0:06d}".format(epoch)
        save_path = saver.save(sess, weights_directory+ID+"/"+
                                     strEpoch+"/"+strEpoch)
        print("Parameters saved")
        return save_path
    
    def _save_cost(self, epoch, epoch_cost, From='weights', params={}):
        """ save cost in cost.p file """
        if From=='weights':
            if 'IDweights' in params:
                ID = params['IDweights']
            else:
                ID = 'FDNN00001'
            directory = local_vars.weights_directory+ID+'/'
        elif From=='results':
            if 'IDresults' in params:
                ID = params['IDresults']
            else:
                ID = 'FDNN10001'
            directory = local_vars.results_directory+ID+'/'
        else:
            print("From: "+From)
            raise ValueError("Argument From not recognized")
        # load saved costs if exist
        if os.path.exists(directory+"cost.p"):
            cost = pickle.load( open( directory+"cost.p", "rb" ))
        else:
            cost = {}
            if not os.path.exists(directory):
                os.makedirs(directory)
        cost[str(epoch)] = epoch_cost
        pickle.dump(cost, open( directory+"cost.p", "wb" ))
        
    def _load_graph(self, sess, ID, epoch=-1, live='n'):
        """ <DocString> """
        # Create placeholders
        print(local_vars.weights_directory+ID+"/")
        if os.path.exists(local_vars.weights_directory+ID+"/"):
            # load last if epoch = -1
            if epoch == -1:
                ## TODO: from listdir get the file that matches the cost file name
                # get last saved epoch
                epoch = int(sorted(os.listdir(local_vars.weights_directory+ID+"/"))[-2])
            
            strEpoch = "{0:06d}".format(epoch)
            self._saver.restore(sess, local_vars.weights_directory+ID+"/"+
                                strEpoch+"/"+strEpoch)
            print("Parameters loaded. Epoch "+str(epoch))
            epoch += 1
            
        elif live=='n':
            #init_op = tf.global_variables_initializer()
            sess.run(tf.global_variables_initializer())
            epoch = 0
        else:
            raise ValueError("ERROR! Weights don't exist for live session")
        return epoch
            
    def save_output_fn(self, output, cost, method='pickle', tag='NNIO'):
        """ Save output """
        dirfilename = local_vars.IO_directory+tag
        if method=='pickle':
            pickle.dump({'output':output, 
                         'cost':cost}, open( dirfilename+'.p', 'wb'))
        elif method=='hdf5':
            f_IO = h5py.File(dirfilename,'w')
            # init IO data sets
            outputh5 = f_IO.create_dataset('output', output.shape, dtype=float)
            outputh5[:] = output
            f_IO.close()
            print("Output saved in disk as "+dirfilename)
            
    def load_dataset_fn(self, ID, method='pickle', tag='NNIO', 
                        setname='output', ext=''):
        """ Load output """
        dirfilename = local_vars.IO_directory+tag+ID
        if method=='pickle':
            dataset = pickle.load( open( dirfilename+ext, "rb" ))
        elif method=='hdf5':
            f_IO = h5py.File(dirfilename+ext,'r')
            dataset = f_IO[setname]
            #f_IO.close
        else:
            raise ValueError("method is not supported")
        return dataset
    
    def plot_cost(self):
        """ Plot cost over epochs """

class RNN(Model):
    """ Deep neural network model """
    def __init__(self, params={}):
        # get params
        if 'size_hidden_layer' in params:
            self.size_hidden_layer = params['size_hidden_layer']
        else:
            self.size_hidden_layer = 100
        if 'size_output_layer' in params:
            self.size_output_layer = params['size_output_layer']
        else:
            self.size_output_layer = 3
        if 'out_act_func' in params:
            self.out_act_func = params['out_act_func']
        else:
            self.out_act_func = 'tanh'
        if 'nEventsPerStat' in params:
            nEventsPerStat = params['nEventsPerStat']
        else:
            nEventsPerStat = 10000
        if 'movingWindow' in params:
            movingWindow = params['movingWindow']
        else:
            movingWindow = 1000
        if 'lB' in params:
            lB = params['lB']
        else:
            lB = int(nEventsPerStat+movingWindow*3)
        self.seq_len = int((lB-nEventsPerStat)/movingWindow+1)
        if 'L' in params: # number of layers in RNN
            self.L = params['L']
        else:
            self.L = 3   
        if 'L' in params: # number of layers in RNN
            self.L = params['L']
        else:
            self.L = 3
        if 'feature_keys_manual' in params: # number of layers in RNN
            feature_keys = params['feature_keys']
        else:
            feature_keys = [i for i in range(37)]
        if 'channels' in params:
            channels = params['channels']
        else:
            channels = [0]
        self.nFeatures = len(feature_keys)*len(channels)
        #{"RNNv" for vanilla RNN,"LSTM" for long-short time memory}
        if 'RRN_type' in params:
            self.RRN_type = params['RRN_type']
        else:
            self.RRN_type = "LSTM" 
        if 'loss_funcs' in params:
            self.loss_funcs = params['loss_funcs']
            
        else:
            self.loss_funcs = ['cross_entropy']
        if type(self.loss_funcs)==str:
            self.loss_funcs = [self.loss_funcs]
        if 'n_bits_outputs' in params:
            self.n_bits_outputs = params['n_bits_outputs']
        else:
            self.n_bits_outputs = [self.size_output_layer]
        if 'seed' in params:
            self.seed = params['seed']
        else:
            self.seed = 0
        if 'miniBatchSize' in params:
            self.miniBatchSize = params['miniBatchSize']
        else:
            self.miniBatchSize = 256
        if 'rand_mB' in params:
            self.rand_mB = params['rand_mB']
        else:
            self.rand_mB = True
        if 'IDweights' in params:
            self.IDweights = params['IDweights']
        else:
            self.IDweights = 'WRNN00001'
        if 'lR0' in params:
            lR0 = params['lR0']
        else:
            lR0 = 0.0001
        if 'optimazer_name' in params:
            optimazer_name = params['optimazer_name']
        else:
            optimazer_name = 'adam'
        if 'lR0' in params:
            lR0 = params['lR0']
        else:
            lR0 = 0.0001
        if 'beta1' in params:
            beta1 = params['beta1']
        else:
            beta1 = 0.9
        if 'beta2' in params:
            beta2 = params['beta2']
        else:
            beta2 = 0.999
        if 'epsilon' in params:
            epsilon = params['epsilon']
        else:
            epsilon = 1e-08
        self.params_optimizer = {'lR0':lR0, 'optimazer_name':optimazer_name,
                                'beta1':beta1, 'beta2':beta2, 'epsilon':epsilon}
        
    def _run_forward_prop(self, Wb):
        """ Run forward propagation function """
        def __cell():
            # https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/  \
            # python/ops/rnn_cell_impl.py
            # Tensorflow defaults to no peephole LSTM:
            # http://www.bioinf.jku.at/publications/older/2604.pdf
            if self.RRN_type =="LSTM":
                rnn_cell = tf.contrib.rnn.LSTMCell(self.size_hidden_layer)#,state_is_tuple=True
            elif self.RRN_type =="RNNv":
                rnn_cell = tf.contrib.rnn.BasicRNNCell(self.size_hidden_layer)
            return tf.contrib.rnn.DropoutWrapper(rnn_cell, 
                                                 output_keep_prob=self.dropout)
        
        cell = tf.contrib.rnn.MultiRNNCell([__cell() for _ in range(self.L)])#,state_is_tuple=True
        # Out is of dimension [batch_size, max_time, cell_state_size]
        out, self._state = tf.nn.dynamic_rnn(cell=cell, inputs=self.input, 
                                             dtype=tf.float32)
        out = tf.reshape(out, [-1, self.size_hidden_layer])
        # market change prediction
        pred_mc = tf.nn.sigmoid(tf.matmul(out, Wb["w"][0]) + Wb["b"][0])
        # market direction prediction
        pred_md = tf.nn.softmax(tf.matmul(out, Wb["w"][1]) + Wb["b"][1])
        # Market gain prediction
        pred_mg = tf.nn.softmax(tf.matmul(out, Wb["w"][2]) + Wb["b"][2])
        
        output = tf.concat([tf.reshape(pred_mc, [-1, self.seq_len, 1]),
                              tf.reshape(pred_md, [-1, self.seq_len, 2]),
                              tf.reshape(pred_mg, [-1, self.seq_len, 
                                                  self.n_bits_outputs[-1]])],2)

#        output = tf.reshape(pred_mg, [-1, self.seq_len, self.size_output_layer])
        
        return output
            
    def _build_model(self):
        """ Build DNN model """
        self.input = tf.placeholder(tf.float32, 
                                    [None, self.seq_len, self.nFeatures], 
                                    name="input")
        self.target = tf.placeholder("float", [None, self.seq_len, 
                                    self.size_output_layer], 
                                    name="target")
        
        self.dropout = tf.placeholder(tf.float32)
        # init parameters
        Wb = self._init_parameters()
        # forward prop
        self.output = self._run_forward_prop(Wb)
        
    def _tf_repeat(self, repeats):
        """
        Args:
    
        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, 
        length must be the same as the number of dimensions in input
    
        Returns:
        
        A Tensor. Has the same type as input. Has the shape of tensor.shape*repeats
        """
        repeated_tensor = self.target[:,:,:1]
        for i in range(repeats-1):
            repeated_tensor = tf.concat([repeated_tensor,self.target[:,:,:1]],2)

        return repeated_tensor
    
    def compute_loss(self, loss_funcs):
        """ Loss function """
#        loss = 0
#        bc = 0
#        for f, loss_func in enumerate(loss_funcs):
#            if loss_func=='exponential':
#                raise NotImplementedError("Exponential loss for RNN not implemented yet.") 
#            elif loss_func=='cross_entropy':
#                nb = self.n_bits_outputs[f]
#                if nb>1:
#                    loss = loss+tf.reduce_mean(-tf.reduce_sum(self.target[:,:,bc:bc+nb]*
#                                        tf.log(self.output[:,:,bc:bc+nb]),
#                                        [1, 2])/self.seq_len, name="loss_xe"+str(f))
#                else:
#                    loss = loss+tf.reduce_mean(tf.reduce_sum(self.target[:,:,bc:bc+nb]*\
#                                    -tf.log(self.output[:,:,bc:bc+nb])+
#                                    (1-self.target[:,:,bc:bc+nb])* -
#                                    tf.log(1 - self.output[:,:,bc:bc+nb]), 
#                                    [1, 2])/self.seq_len, name="loss_xe"+str(f))
#            else:
#                raise ValueError("loss_func not supported")
#            bc += nb
        if len(loss_funcs)>1:
            loss_mc = tf.reduce_sum(self.target[:,:,:1] * -tf.log(self.output[:,:,:1])+
                                    (1-self.target[:,:,:1])* -
                                    tf.log(1 - self.output[:,:,:1]), [1, 2])/self.seq_len
            # second and third bits for md (market direction). Add loss only when
            # mc is nonzero
#            loss_md = -tf.reduce_sum(self.target[:,:,1:3] * 
#                                     tf.log(self.output[:,:,1:3]), [1, 2])/self.seq_len
#            # last 5 bits for market gain output. Add loss only when
#            # mc is nonzero
#            loss_mg = -tf.reduce_sum(self.target[:,:,3:] * 
#                                  tf.log(self.output[:,:,3:]), [1, 2])/self.seq_len

            loss_md = -tf.reduce_sum(self._tf_repeat(2)*(self.target[:,:,1:3] * 
                                     tf.log(self.output[:,:,1:3])), [1, 2])/self.seq_len
            # last 5 bits for market gain output. Add loss only when
            # mc is nonzero
            loss_mg = -tf.reduce_sum(self._tf_repeat(self.n_bits_outputs[-1])*
                                 (self.target[:,:,3:] * 
                                  tf.log(self.output[:,:,3:])), [1, 2])/self.seq_len
            j = tf.reduce_mean(
                loss_mc+loss_md+loss_mg,
                name="loss_nll")
            j_mc = tf.reduce_mean(
                loss_mc,
                name="loss_mc")
            j_md = tf.reduce_mean(
                loss_md,
                name="loss_md")
            j_mg = tf.reduce_mean(
                loss_mg,
                name="loss_mg")
            return j, j_mc, j_md, j_mg
        else:
            loss_mg = -tf.reduce_sum(self.target *tf.log(self.output), [1, 2])/self.seq_len
            loss = tf.reduce_mean(loss_mg, name="loss_nll")
            return loss
    
    def fit(self, X, Y, num_epochs=100, keep_prob_dropout=1.0, log='',aloc=3000000):
        """ Fit model to trainning data """
        from tqdm import tqdm
        # directory to save weights
        weights_directory = local_vars.weights_directory
        # number of samples
        m = X.shape[0]
        # init timer
        tic = time.time()
        # init graph
        tf.reset_default_graph()
        # build model
        self._build_model()
        # init session
        with tf.Session() as sess:
            # get loss function
            loss, _, _, _ = self.compute_loss(self.loss_funcs)
            # define optimizer
            optimizer = get_optimazer(loss, params=self.params_optimizer)
            # define save object
            self._saver = tf.train.Saver(max_to_keep = None)
            # restore graph
            epoch_init = self._load_graph(sess, self.IDweights, epoch=-1)
            # get minibatches
#            minibatches = generate_mini_batches(X, Y, 
#                                                mini_batch_size=self.miniBatchSize, 
#                                                seed=self.seed, random=self.rand_mB)
            
            
            # get epochs range
            epochs = range(epoch_init, epoch_init+num_epochs)
            mess = "Fitting model from epoch "+str(epochs[0])+" till "+str(epochs[-1])
            print(mess)
            if len(log)>0:
                write_log(mess)
            exception = False
            try:
                nChunks = int(np.ceil(X.shape[0]/aloc))
                for epoch in epochs:
                    J_train = 0
                    chunk = 0
                    print("#Chunks = "+str(nChunks))
                    while chunk<nChunks:
                        Xc = X[chunk*aloc:(chunk+1)*aloc,:,:]
                        Yc = Y[chunk*aloc:(chunk+1)*aloc,:,:]
                        m = Xc.shape[0]
                        print("Chunck size: "+str(m))
                        minibatchidx = generate_mini_batch_idx(m, 
                                                mini_batch_size=self.miniBatchSize, 
                                                seed=self.seed, random=self.rand_mB)
                        n_mB = len(minibatchidx)
                        for minibatch in tqdm(minibatchidx, mininterval=10):
                            #(X_batch, Y_batch) = minibatch
                            X_batch = Xc[minibatch,:,:]
                            Y_batch = Yc[minibatch,:,:]
                            feed_dict = {self.input:X_batch, self.target:Y_batch, 
                                         self.dropout:keep_prob_dropout}
                            _, cost = sess.run([optimizer, loss], 
                                                    feed_dict=feed_dict)
                            J_train += cost
                        print("Time training Chunck {0:d} of {1:d}: {2:.0f} s"\
                              .format(chunk,nChunks-1,np.floor(time.time()-tic)))
                        chunk += 1
                    J_train= J_train/(n_mB*nChunks)
                    mess = "Cost after epoch %i: %f. Av cost %f" \
                               % (epoch, cost, J_train)
                
                print(mess)
                if len(log)>0:
                    write_log(mess)
                mess = dt.datetime.strftime(dt.datetime.now(),"%H:%M:%S")+\
                         " Total time training: "+\
                         "{0:.2f}".format(np.floor(time.time()-tic)/60)+"m"
                print(mess)
                if len(log)>0:
                    write_log(mess)
                #if save_graph:
                self._save_graph(self._saver, sess, self.IDweights, 
                                 J_train, epoch, weights_directory)
                self._save_cost(epoch, J_train, From='weights', 
                                params={'IDweights':self.IDweights})
            except KeyboardInterrupt:
                mess = "Trainning stopped due to KeyboardInterrupt exception"
                print(mess)
                if len(log)>0:
                    write_log(mess)
                exception = True
        if not exception:
            print("Model fit.")
        return self
        
    def cv(self, X, Y, DTA=[], IDresults='RRNN00001', alloc=2**10, 
           save_output=False, if_get_results=True, tag='DNNIO',
           startFrom=-1, endAt=-1, config={}, save_cost=True, log=''):
        """ Cross-validation function """
        from tqdm import tqdm
        tic = time.time()
        results_directory = local_vars.results_directory
        IDweights = self.IDweights
        loss_funcs = self.loss_funcs
        n_bits_outputs = self.n_bits_outputs
        m = Y.shape[0]
        n_chunks = int(np.ceil(m/alloc))
        # retrieve costs
        costs_dict, costs = retrieve_costs(tOt='tr', IDweights=IDweights)
        # get epochs range
        epochs = get_epochs_range(IDweights=IDweights, IDresults=IDresults, 
                                  seq_len=self.seq_len, 
                                  startFrom=startFrom, endAt=endAt)
        # init graph
        tf.reset_default_graph()
        # build model
        self._build_model()
        # init output
        output = np.array([])
        # init session
        with tf.Session() as sess:
            try:
                # get loss function
                loss, loss_mc, loss_md, loss_mg = self.compute_loss(loss_funcs)
                # load models and test them
                for it, epoch in enumerate(epochs):
                    # make sure cost in not a NaN
                    if check_nan(costs, epoch):
                        mess = "WARNING! cost=NaN. Break CV"
                        print(mess)
                        if len(log)>0:
                            write_log(mess)
                        break
                    # define save object
                    self._saver = tf.train.Saver(max_to_keep = None)
                    # load graph
                    self._load_graph(sess, IDweights, epoch=epoch)
                    mess = "Epoch "+str(epoch)+" of "+str(epochs[-1])+\
                          ". Getting output..."
                    print(mess)
                    if len(log)>0:
                        write_log(mess)
        #            J_train = self._loss.eval()
                    output = np.zeros(Y.shape)
                    J = 0
                    J_mc = 0
                    J_md = 0
                    J_mg = 0
                    for chunck in tqdm(range(n_chunks),mininterval=1):
                        #tqdm.write("Chunck "+str(chunck+1)+" of "+str(n_chunks))
                        # build input/output data
                        test_data_feed = {
                            self.input: X[chunck*alloc:(chunck+1)*alloc],
                            self.target: Y[chunck*alloc:(chunck+1)*alloc],
                            self.dropout: 1.0
                        }
#                        print("self.output.shape")
#                        print(self.output.shape)
                        cost, cost_mc, cost_md, cost_mg, output_chunck = sess.run([loss, loss_mc, 
                                                                loss_md, loss_mg, 
                                                                self.output], \
                                                               test_data_feed)
                        J += cost
                        J_mc += cost_mc
                        J_md += cost_md
                        J_mg += cost_mg
                        output[chunck*alloc:(chunck+1)*alloc] = output_chunck
                        #print(softMaxOut.shape)
                    J = J/n_chunks
                    J_mc = J_mc/n_chunks
                    J_md = J_md/n_chunks
                    J_mg = J_mg/n_chunks
                    Js = [J, J_mc, J_md, J_mg]
                    mess = "J = {0:.6f}".format(J)+" J_mc = {0:.6f}".format(J_mc)+" J_md = {0:.6f}".format(J_md)+" J_mg = {0:.6f}".format(J_mg)
                    print(mess)
                    if len(log)>0:
                        write_log(mess)
                    
                    if save_output:
                        self.save_output_fn(output, Js, 
                                            method='hdf5', tag=tag)
                    if if_get_results:
                        ## TEMPORARY: use legacy results functions
                        if len(n_bits_outputs)==1:
                            if it==0:
                                t_indexes = [str(t) if t<self.seq_len else 'mean' 
                                             for t in range(self.seq_len+1)]
                                results_filename, costs_filename, performance_filename = \
                                    init_results_mg_dir(results_directory, 
                                                        IDresults, 
                                                        self.size_output_layer,
                                                        t_indexes,
                                                        get_performance=True)
                            Thread(target=get_results_mg,
                                   args=(config, Y, output, costs_dict, epoch, 
                                           Js, costs_filename, results_filename,
                                           performance_filename),
                                   kwargs={'get_performance':True,'DTA':DTA}).start()
#                            get_results_mg(config, Y, output, costs_dict, epoch, 
#                                           J_test, costs_filename, results_filename,
#                                           performance_filename,
#                                           get_performance=True, DTA=DTA)
                        else:
                            if it==0:
                                results_filename, costs_filename = init_results_dir(results_directory, IDresults)
                            get_results(config, Y, DTA, 
                                Js, output, costs_dict, epoch, -1, results_filename,
                                costs_filename, from_var=False)
                    if save_cost:
                        self._save_cost(epoch, Js, From='results', 
                                        params={'IDresults':IDresults})
            except KeyboardInterrupt:
                mess = "CV stopped due to KeyboardInterrupt exception"
                print(mess)
                if len(log)>0:
                    write_log(mess)
        mess = "Total time for CV: "+str((time.time()-tic)/60)+" mins.\n"
        print(mess)
        if len(log)>0:
            write_log(mess)
        return output, Js
        
    def test(self):
        """  """
        raise NotImplementedError("Test function not implemented yet")
    
    def _init_parameters(self):
        """ Init weights and biases per layer """
        # Define variables for final fully connected layer.
        Wb = {}
        w_mg = tf.Variable(tf.truncated_normal([self.size_hidden_layer, 
                                                self.n_bits_outputs[-1]], 
                                                stddev=0.01), name="w_mg")
        b_mg = tf.Variable(tf.constant(0.1, shape=[self.n_bits_outputs[-1]]), 
                                                              name="b_mg")
        w_mc = tf.Variable(tf.truncated_normal([self.size_hidden_layer, 1], 
                                                stddev=0.01), name="w_mc")
        b_mc = tf.Variable(tf.constant(0.1, shape=[1]), name="b_mc")
        w_md = tf.Variable(tf.truncated_normal([self.size_hidden_layer, 2], 
                                                stddev=0.01), name="w_md")
        b_md = tf.Variable(tf.constant(0.1, shape=[2]), name="b_md")
        
        Wb["w"] = [w_mc, w_md, w_mg]
        Wb["b"] = [b_mc, b_md, b_mg]
        return Wb

    def predict(self, X):
        """ Predict output for features input X. If an interactive session
        has not yet been initialzed, this function does it. """
#        if not hasattr(self, '_live_session'):
#            self.init_interactive_session(epoch=epoch)
        test_data_feed = {
            self.input: X,
            self.dropout: 1.0
        }
        output = self._live_session.run([self.output], test_data_feed)
        return output
    
    def predict_batch(self, X, epoch, alloc=2**10):
        """ Predict output for features input X. If an interactive session
        has not yet been initialzed, this function does it. """
        from tqdm import tqdm
        # init graph
        tf.reset_default_graph()
        # build model
        self._build_model()
        IDweights = self.IDweights
        m = X.shape[0]
        output = np.zeros((X.shape[0], X.shape[1], self.size_output_layer))
        
        with tf.Session() as sess:
            # define save object
            self._saver = tf.train.Saver(max_to_keep = None)
            # load graph
            self._load_graph(sess, IDweights, epoch=epoch)
            n_chunks = int(np.ceil(m/alloc))
            
            for chunck in tqdm(range(n_chunks),mininterval=1):
                #tqdm.write("Chunck "+str(chunck+1)+" of "+str(n_chunks))
                # build input/output data
                test_data_feed = {
                        self.input: X[chunck*alloc:(chunck+1)*alloc,:,:],
                        self.dropout: 1.0
                }
#                       print("self.output.shape")
#                       print(self.output.shape)
                output_chunck = sess.run([self.output], test_data_feed)[0]
                #J_test += J_test_chuck
#                print(output_chunck.shape)
#                print(output[chunck*alloc:(chunck+1)*alloc,:,:].shape)
                output[chunck*alloc:(chunck+1)*alloc,:,:] = output_chunck
                        
#            test_data_feed = {
#                self.input: X,
#                self.dropout: 1.0
#            }
#            output = sess.run([self.output], test_data_feed)[0]
        return output
    
    def init_interactive_session(self, epoch=0):
        """ Initialize an interactive session for live predictions """
        print("Initing interactive session of "+self.IDweights+" E "+str(epoch))
        graph = tf.Graph()
        with graph.as_default():
            self._build_model()
            self._saver = tf.train.Saver(max_to_keep = None)
            self._live_session = tf.Session()
            self._load_graph(self._live_session, self.IDweights, epoch, live='y')
            
        return self

class StackedModel(Model):
    """  """
    def __init__(self, configs, model_names):
        self.model_names = model_names
        self.n_models = len(self.model_names)
        self.size_output_layer = configs[0]['size_output_layer']
        self.seq_len = configs[0]['seq_len']
        self.combine = configs[0]['combine_ts']['params_combine'][0]['alg']
        self.models = [RNN(config) for config in configs]
    
    def predict(self, X):
        """ Predict output for features input X. """
        output = np.zeros((1, self.n_models*self.seq_len, self.size_output_layer))
        # loop over models to run predict on each one
        for m, model in enumerate(self.models):
            output[:,m*self.seq_len:(m+1)*self.seq_len,:] = model.predict(X)
        if self.combine == 'mean':
            # reduce mean
            output = np.mean(output, axis=1, keepdims=True)
        return output
    
    def init_interactive_session(self, epochs=['12', '14']):
        """ Initialize the interactive sessions for live predictions """
        self.models = [model.init_interactive_session(epoch=epochs[e]) for e,model in enumerate(self.models)]
        return self

class XGB():
    """ XGB model """
    def __init__(self, params={}):
        self.params = params
        self.model = xgb.LGBMRegressor(verbosity=1)
#        self.model = lgb.LGBMRegressor(objective = "regression", 
#        boosting = "gbdt",
#        metric="auc",
#        boost_from_average=False,
#        num_threads=8,
#        learning_rate =0.0081,
#        num_leaves =13,
#        max_depth=-1,
#        feature_fraction =0.041,
#        bagging_freq =5,
#        bagging_fraction =0.331,
#        min_data_in_leaf =80,
#        min_sum_hessian_in_leaf =10.0,
#        verbosity =1,
#        num_iterations =99999999,
#        seed=44000)
    
    def fit(self, X, Y):
        """  """
        # Create LGB dataset
        #d_train = lgb.Dataset(X, label= Y)
        # fit
#        self.model = lgb.train(self.params, d_train)
        self.model.fit(X, Y)
        print("Model fitted. Score: ")
        print(self.model.score(X, Y))
        
        return self
    
    def cv(self, X, Y):
        """  """
        Y_tilde = self.model.predict(X)
        print("Prediction score: ")
        print(self.model.score(X, Y))
        print("rmsle: ")
        print(self._rmsle(Y, Y_tilde))
        return Y_tilde
    
    def predict(self, X, Y):
        """  """
        pass
    
    def _rmsle(self, Y, Y_tilde):
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(Y, Y_tilde))
    
class LGBM():
    """ XGB model """
    def __init__(self, params={}):
        import lightgbm as lgb
        self.params = params
        self.model = lgb.LGBMRegressor(verbosity=1)
#        self.model = lgb.LGBMRegressor(objective = "regression", 
#        boosting = "gbdt",
#        metric="auc",
#        boost_from_average=False,
#        num_threads=8,
#        learning_rate =0.0081,
#        num_leaves =13,
#        max_depth=-1,
#        feature_fraction =0.041,
#        bagging_freq =5,
#        bagging_fraction =0.331,
#        min_data_in_leaf =80,
#        min_sum_hessian_in_leaf =10.0,
#        verbosity =1,
#        num_iterations =99999999,
#        seed=44000)
    
    def fit(self, X, Y):
        """  """
        # Create LGB dataset
        #d_train = lgb.Dataset(X, label= Y)
        # fit
#        self.model = lgb.train(self.params, d_train)
        
        self.model.fit(X, Y)
        print("Model fitted. Score: ")
        print(self.model.score(X, Y))
        
        return self
    
    def cv(self, X, Y):
        """  """
        Y_tilde = self.model.predict(X)
        print("CV prediction score: ")
        print(self.model.score(X, Y))
        print("rmsle: ")
        print(self.rmsle(Y, Y_tilde))
        return Y_tilde
    
    def predict(self, X):
        """  """
        return self.model.predict(X)
    
    def rmsle(self, Y, Y_tilde):
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(Y, Y_tilde))

class DNN(Model):
    """ Deep neural network model """
    def __init__(self, params={}):
        # get params
        if 'size_input_layer' in params:
            self.size_input_layer = params['size_input_layer']
        else:
            self.size_input_layer = 20
        if 'size_hidden_layers' in params:
            self.size_hidden_layers = params['size_hidden_layers']
        else:
            self.size_hidden_layers = []
        self.n_hidden_layers = len(self.size_hidden_layers)
        if 'size_output_layer' in params:
            self.size_output_layer = params['size_output_layer']
        else:
            self.size_output_layer = 1
        if 'out_act_func' in params:
            self.out_act_func = params['out_act_func']
        else:
            self.out_act_func = 'tanh'
        if 'act_funcs' in params:
            self.act_funcs = params['act_funcs']+[self.out_act_func]
        else:
            self.act_funcs = ['relu' for _ in self.size_hidden_layers]+[self.out_act_func]
        assert(len(self.act_funcs)==self.n_hidden_layers+1)
        if 'loss_func' in params:
            self.loss_func = params['loss_func']
        else:
            self.loss_func = 'exponential'
        if 'seed' in params:
            self.seed = params['seed']
        else:
            self.seed = 0
        if 'miniBatchSize' in params:
            self.miniBatchSize = params['miniBatchSize']
        else:
            self.miniBatchSize = 256
        if 'IDweights' in params:
            self.IDweights = params['IDweights']
        else:
            self.IDweights = 'FDNN00001'
        if 'lR0' in params:
            lR0 = params['lR0']
        else:
            lR0 = 0.0001
        if 'optimazer_name' in params:
            optimazer_name = params['optimazer_name']
        else:
            optimazer_name = 'adam'
        if 'lR0' in params:
            lR0 = params['lR0']
        else:
            lR0 = 0.0001
        if 'beta1' in params:
            beta1 = params['beta1']
        else:
            beta1 = 0.9
        if 'beta2' in params:
            beta2 = params['beta2']
        else:
            beta2 = 0.999
        if 'epsilon' in params:
            epsilon = params['epsilon']
        else:
            epsilon = 1e-08
        self.params_optimizer = {'lR0':lR0, 'optimazer_name':optimazer_name,
                                'beta1':beta1, 'beta2':beta2, 'epsilon':epsilon}
        
    def fit(self, X, Y, num_epochs=100, keep_prob_dropout=1.0):
        """ Fit model to train data """
        from tqdm import tqdm
        weights_directory = local_vars.weights_directory
        # init timer
        tic = time.time()
        # init graph
        tf.reset_default_graph()
        # build model
        self._build_model()
        # init session
        with tf.Session() as sess:
            # get loss function
            loss = compute_loss(self, self.loss_func)
            # define optimizer
            optimizer = get_optimazer(loss, params=self.params_optimizer)
            # define save object
            self._saver = tf.train.Saver(max_to_keep = None)
            # restore graph
            epoch_init = self._load_graph(sess, self.IDweights, epoch=-1)
            # get minibatches
            minibatches = generate_mini_batches(X, Y, 
                                                mini_batch_size=self.miniBatchSize, 
                                                seed=self.seed, random=True)
            n_mB = len(minibatches)
            # get epochs range
            epochs = range(epoch_init, epoch_init+num_epochs)
            exception = False
            counter = 0
            try:
                
                for epoch in epochs:
                    J_train = 0
                    for minibatch in tqdm(minibatches, mininterval=1):
                        (X_batch, Y_batch) = minibatch
                        feed_dict = {self.input:X_batch, self.target:Y_batch, 
                                     self.dropout:keep_prob_dropout}
                        output, cost = sess.run([optimizer, loss], feed_dict=feed_dict)
                        J_train += cost
                    J_train= J_train/n_mB
                    print ("Cost after epoch %i: %f. Av cost %f" % (epoch, cost, J_train))
                    print(dt.datetime.strftime(dt.datetime.now(),"%H:%M:%S")+
                      " Total time training: "+"{0:.2f}".format(np.floor(time.time()-tic)/60)+"m")
                    self._save_graph(self._saver, sess, self.IDweights, 
                                     J_train, epoch, weights_directory)
                    self._save_cost(epoch, J_train, From='weights', 
                                    params={'IDweights':self.IDweights})
                    counter += 1
            except KeyboardInterrupt:
                print("Trainning stopped due to KeyboardInterrupt exception")
                exception = True
        if not exception:
            print("Model fit.")
        return self
        
    def cv(self, X, Y, DTA=[], IDresults='RRNN00001', alloc=2**10, 
           save_output=False, if_get_results=True, tag='RDNN',
           startFrom=-1, endAt=-1, config={}):
        """ Cross-validation function """
        from tqdm import tqdm
        tic = time.time()
        IDweights = self.IDweights
        loss_func = self.loss_func
        results_directory = local_vars.results_directory
        m = Y.shape[0]
        n_chunks = int(np.ceil(m/alloc))
        # retrieve costs
        costs_dict, costs = retrieve_costs(tOt='tr', IDweights=IDweights)
        # get epochs range
        epochs = get_epochs_range(IDweights=IDweights, IDresults=IDresults, 
                                  startFrom=startFrom, endAt=startFrom)
        # init graph
        tf.reset_default_graph()
        # build model
        self._build_model()
        # init session
        with tf.Session() as sess:
            # get loss function
            loss = compute_loss(self, loss_func)
            # load models and test them
            for it, epoch in enumerate(epochs):
                # make sure cost in not a NaN
                if check_nan(costs, epoch):
                    break
                # define save object
                self._saver = tf.train.Saver(max_to_keep = None)
                # load graph
                self._load_graph(sess, IDweights, epoch=epoch)
                print("Epoch "+str(epoch)+" of "+str(epochs[-1])+". Getting output...")
    #            J_train = self._loss.eval()
                output = np.zeros(Y.shape)
                J_test = 0
                for chunck in tqdm(range(n_chunks)):
                    #tqdm.write("Chunck "+str(chunck+1)+" of "+str(n_chunks))
                    # build input/output data
                    test_data_feed = {
                        self.input: X[chunck*alloc:(chunck+1)*alloc],
                        self.target: Y[chunck*alloc:(chunck+1)*alloc],
                        self.dropout: 1.0
                    }
                    J_test_chuck, output_chunck = sess.run([loss, self.output], test_data_feed)
                    J_test += J_test_chuck
                    output[chunck*alloc:(chunck+1)*alloc] = output_chunck
                    #print(softMaxOut.shape)
                J_test = J_test/n_chunks
                print("J_test = {0:.6f}".format(J_test))
                self._save_cost(epoch, J_test, From='results', 
                                params={'IDresults':IDresults})
                if save_output:
                    self.save_output_fn(output, J_test, method='hdf5', tag=tag)
                if if_get_results:
                    ## TEMPORARY: use legacy results functions
                    if it==0:
                        t_indexes = ['0']
                        results_filename, costs_filename, performance_filename = \
                        init_results_mg_dir(results_directory, 
                                            IDresults, 
                                            3,
                                            t_indexes,
                                            get_performance=True)
                    save_costs(costs_filename, [epoch, costs_dict[str(epoch)], J_test])
                    get_performance_meta(config, Y, output, DTA, epoch, 
                         performance_filename)
#                    get_results_meta(config, Y, output, costs_dict, epoch, 
#                                   J_test, costs_filename, results_filename,
#                                   performance_filename,
#                                   get_performance=True, DTA=DTA)
        
        print("Total time for testing: "+str((time.time()-tic)/60)+" mins.\n")
        return self
        
    def test(self):
        pass
    
    def _init_parameters(self):
        """ Init weights and biases per layer """
        Wb = {}
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        # loop over hidden layers
        s_l_1 = self.size_input_layer
        l = -1
        for l, L in enumerate(self.size_hidden_layers):
            Wb['W'+str(l)] = tf.Variable(w_initializer([s_l_1, L]))
            Wb['b'+str(l)] = tf.Variable(b_initializer([1, L]))
            s_l_1 = L
        # add last layer
        Wb['W'+str(l+1)] = tf.Variable(w_initializer([s_l_1, self.size_output_layer]))
        Wb['b'+str(l+1)] = tf.Variable(b_initializer([1, self.size_output_layer]))
#        Wb['W'+str(l+1)] = tf.Variable(tf.truncated_normal([s_l_1, self.size_output_layer], stddev=0.01), name="fc_w")
#        Wb['b'+str(l+1)] = tf.Variable(tf.constant(0.1, shape=[1, self.size_output_layer]), name="fc_b")
        return Wb
        
    def _run_forward_prop(self, Wb):
        """ Run forward propagation function """
        A = self.input
        for l, act_func in enumerate(self.act_funcs):
            W = Wb['W'+str(l)]
            b = Wb['b'+str(l)]
            Z = tf.add(tf.matmul(A, W), b)
            if act_func=='relu':
                A = tf.nn.relu(Z)
            elif act_func=='sigmoid':
                A = tf.nn.sigmoid(Z)
            elif act_func=='tanh':
                A = tf.nn.tanh(Z)
            elif act_func=='softmax':
                A = tf.nn.softmax(Z)
            else:
                raise ValueError('acti_func not supported')
#            if self.keep_prob_dropout<1 and l<self.size_hidden_layers:
#                A = tf.nn.dropout(A, self.keep_prob_dropout)
        return A
            
    def _build_model(self):
        """ Build DNN model """
        # define placeholders
        self.input = tf.placeholder(tf.float32, [None, self.size_input_layer], name='X')
        self.target = tf.placeholder(tf.float32, [None, self.size_output_layer], name='Y')
        self.dropout = tf.placeholder(tf.float32)
        # init parameters
        Wb = self._init_parameters()
        # forward prop
        self.output = self._run_forward_prop(Wb)
        # init saver obect
    
    def predict(self, X, epoch=0):
        """ Predict output for features input X. If an interactive session
        has not yet been initialzed, this function does it. """
        if not hasattr(self, '_live_session'):
            self.init_interactive_session(epoch=epoch)
        test_data_feed = {
            self.input: X,
            self.dropout: 1.0
        }
        output = self._live_session.run([self.output], test_data_feed)
        return output
    
    def init_interactive_session(self, epoch=0):
        """ Initialize an interactive session for live predictions """
        graph = tf.Graph()
        with graph.as_default():
            self._build_model()
            self._saver = tf.train.Saver(max_to_keep = None)
            self._live_session = tf.Session()
            self._load_graph(self._sess, self.IDweights, epoch)
            
        return self
    
def compute_loss(model, loss_func):
    """ Loss function """
    if loss_func=='exponential':
        loss = tf.reduce_mean(tf.exp(-model.target*model.output), name="loss_exp")
    elif loss_func=='cross_entropy':
        loss = tf.reduce_mean(model.target *-tf.log(model.output)+
                               (1-model.target)*-tf.log(1 - model.output), 
                               name="loss_xe")
    else:
        raise ValueError("loss_func not supported")
    return loss

def get_optimazer(loss, params={}):
    """  """
    if 'optimazer_name' in params:
        optimazer_name = params['optimazer_name']
    else:
        optimazer_name = 'adam'
    if 'lR0' in params:
        lR0 = params['lR0']
    else:
        lR0 = 0.0001
    if 'beta1' in params:
        beta1 = params['beta1']
    else:
        beta1 = 0.9
    if 'beta2' in params:
        beta2 = params['beta2']
    else:
        beta2 = 0.999
    if 'epsilon' in params:
        epsilon = params['epsilon']
    else:
        epsilon = 1e-08
    
    if optimazer_name=='adam':
        optimazer = tf.train.AdamOptimizer(learning_rate=lR0, 
                                           beta1=beta1, beta2=beta2,
                                           epsilon=epsilon).minimize(loss)
    else:
        raise ValueError("optimazer_name not supported")
    return optimazer

#def generate_mini_batches_v1(X, Y, mini_batch_size=64, seed=0, random=True):
#    """
#    Creates a list of random minibatches from (X, Y)
#    
#    Arguments:
#    X -- input data, of shape (input size, number of examples)
#    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
#    mini_batch_size - size of the mini-batches, integer
#    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
#    
#    Returns:
#    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
#    """
#    import math
#    m = X.shape[1]                  # number of training examples
#    mini_batches = []
#    np.random.seed(seed)
#    
#    # Step 1: Shuffle (X, Y)
#    if random:
#        perm = np.random.permutation(m)
#    else:
#        perm = np.array(range(m))
#    permutation = list(perm)
#    shuffled_X = X[:, permutation]
#    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
#
#    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
#    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
#    for k in range(0, num_complete_minibatches):
#        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
#        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
#        mini_batch = (mini_batch_X, mini_batch_Y)
#        mini_batches.append(mini_batch)
#    
#    # Handling the end case (last mini-batch < mini_batch_size)
#    if m % mini_batch_size != 0:
#        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
#        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
#        mini_batch = (mini_batch_X, mini_batch_Y)
#        mini_batches.append(mini_batch)
#    
#    return mini_batches

def generate_mini_batches(X, Y, mini_batch_size=64, seed=0, random=True):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    import math
    X_original_shape = X.shape
    Y_original_shape = Y.shape
    m = X.shape[0]                  # number of training examples
    X = X.reshape((m, -1))
    Y = Y.reshape((m, -1))
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    if random:
        perm = np.random.permutation(m)
    else:
        perm = np.array(range(m))
    permutation = list(perm)
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mB_idx = range(k * mini_batch_size, k * mini_batch_size + mini_batch_size)
        X_reshape = (len(mB_idx),)+X_original_shape[1:]
        Y_reshape = (len(mB_idx),)+Y_original_shape[1:]
        mini_batch_X = shuffled_X[mB_idx, :].reshape(X_reshape)
        mini_batch_Y = shuffled_Y[mB_idx, :].reshape(Y_reshape)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
#    print("mini_batch_X.shape")
#    print(mini_batch_X.shape)
#    print("mini_batch_Y.shape")
#    print(mini_batch_Y.shape)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mB_idx = range(num_complete_minibatches * mini_batch_size, m)
        X_reshape = (len(mB_idx),)+X_original_shape[1:]
        Y_reshape = (len(mB_idx),)+Y_original_shape[1:]
        mini_batch_X = shuffled_X[mB_idx, :].reshape(X_reshape)
        mini_batch_Y = shuffled_Y[mB_idx, :].reshape(Y_reshape)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
#    print("mini_batch_X.shape")
#    print(mini_batch_X.shape)
#    print("mini_batch_Y.shape")
#    print(mini_batch_Y.shape)
    return mini_batches

def generate_mini_batch_idx(m, mini_batch_size=64, seed=0, random=True):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    import math                # number of training examples
#    X_shape = X.reshape((m, -1))
#    Y_shape = Y.reshape((m, -1))
    mini_batch_idx = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    if random:
        perm = np.random.permutation(m)
    else:
        perm = np.array(range(m))
    permutation = list(perm)
#    shuffled_X = X[permutation, :]
#    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mB_idx = permutation[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
#        X_reshape = (len(mB_idx),)+X_original_shape[1:]
#        Y_reshape = (len(mB_idx),)+Y_original_shape[1:]
#        mini_batch_X = shuffled_X[mB_idx, :].reshape(X_reshape)
#        mini_batch_Y = shuffled_Y[mB_idx, :].reshape(Y_reshape)
#        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batch_idx.append(mB_idx)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mB_idx = permutation[num_complete_minibatches * mini_batch_size: m]
        mini_batch_idx.append(mB_idx)
    return mini_batch_idx

def check_nan(costs, epoch):
    """  """
    import math
    if type(costs)==dict:
        if math.isnan(costs[str(epoch)]):
            print("J_train=NaN BREAK!")
            return True
        else:
            return False
    elif type(costs)==type(np.array([])) or type(costs)==list:
        if math.isnan(costs[epoch]):
            print("J_train=NaN BREAK!")
            return True
        else:
            return False
    else:
        raise ValueError("costs type not known")

def get_epochs_range(IDweights='FDNN00001', IDresults='RRNN00001', seq_len=0,
                     startFrom=-1, endAt=-1):
    """ Get epochs range """
    weights_directory = local_vars.weights_directory
    results_directory = local_vars.results_directory
    if startFrom==-1:
        startFrom = get_last_saved_epoch2(results_directory, IDresults)+1
    if endAt==-1:
        directory_list = sorted(os.listdir(weights_directory+IDweights+"/"))
        endAt = int(directory_list[-2])
    # build epochs range
    epochs = range(startFrom, endAt+1)
    return epochs

def retrieve_costs(tOt='tr', IDweights='FDNN00001', IDresults='FDNN10001'):
    """  """
    if tOt=='tr':
        ID = IDweights
        directory = local_vars.weights_directory+ID+'/'
    elif tOt=='te':
        ID = IDresults
        directory = local_vars.results_directory+ID+'/'
    else:
        print("tOt: "+tOt)
        raise ValueError("Argument From not recognized")
    
#    if 'IDweights' in params:
#        IDweights = params['IDweights']
#    else:
#        IDweights = 'FDNN00001'
#    if 'weights_directory' in params:
#        weights_directory = params['weights_directory']
#    else:
#        weights_directory = local_vars.weights_directory
    print("Loading "+directory+"cost.p")
    if os.path.exists(directory+"cost.p"):
        costs = pickle.load( open( directory+"cost.p", "rb" ))
    else:
        raise ValueError("File cost.p does not exist.")
#    directory_list = sorted(os.listdir(weights_directory+IDweights+"/"))
#    print(directory_list)
#    print(filter(lambda x: 'cost' in x, directory_list))
#    costfile_id = [l for l,s in enumerate(directory_list) if 'cost' in s][0]
#    print(costfile_id)
#    print(directory_list[costfile_id])
    if type(costs)==dict:
        np_costs = np.zeros((len(costs)))
        c = 0
        for i in costs:
            np_costs[int(i)] = costs[i]
            c += 1
        #costs = np_costs
    return costs, np_costs