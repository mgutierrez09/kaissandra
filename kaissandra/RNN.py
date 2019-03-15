# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:29:31 2017

@author: mgutierrez

Module contatining all relevant classes and functions to
build a Recurrent Neural Network
"""
import numpy as np
import tensorflow as tf
import pickle
import time
import h5py
import datetime as dt
import os
from tqdm import tqdm
#import pandas as pd
#import matplotlib.pyplot as plt
from kaissandra.results import evaluate_RNN,save_best_results,get_last_saved_epoch
from kaissandra.results2 import (get_results, 
                                 get_last_saved_epoch2,
                                 init_results_dir,
                                 get_results_mg,
                                 init_results_mg_dir)
from kaissandra.local_config import local_vars

class trainData:
    def __init__(self,X,Y):
        self.data = X
        self.target = Y
        self.m = X.shape[0]
#_, seq_len, input_size = train.data.shape
#print(train.data.shape)
        

class modelRNN():
    
    def __init__(self,
                 data,
                 size_hidden_layer=400,
                 L=3,
                 size_output_layer=5,
                 keep_prob_dropout=1,
                 miniBatchSize=32,
                 outputGain=0.5,
                 commonY=3,
                 num_epochs=1000,
                 lR0=0.003,
                 version="0.0",
                 IDgraph="",
                 lID=6,
                 sess=None):
                    # tf session
        self.seq_len = int((data.lB-data.nEventsPerStat)/data.movingWindow+1)
        self._num_epochs = num_epochs
        self.size_hidden_layer = size_hidden_layer # number of hidden units per layer in RNN
        self.L = L     # number of layers in RNN
        self.nFeatures = data.nFeatures*len(data.channels)
        self.lR0 = lR0 # learning rate
        self.keep_prob_dropout= keep_prob_dropout       # probability of dropout
        self.outputGain = outputGain
        self.commonY = commonY # indicates if the output Y should contain common bits y_c0, y_c1 and y_c2.
                               # commonY={0=no common bits,1=y_c0 active,2=[y_c1,y_c2] active,3=[y_c0,y_c1,y_c2] active}
                               # if y_c0 is active, means that all outputs != 0 will contain a 1 at y_c0.
                               # if [y_c1, yc2]=10, means 
        self.miniBatchSize = miniBatchSize # batch size for each iteration
        self.size_output_layer = size_output_layer   # number of classes (in this case 5)
        self.version = version
        self.RRN_type = "LSTM" #{"RNNv" for vanilla RNN,"LSTM" for long-short time memory}
        # Defined when graph built
        self._dropout = None
        self._inputs = None
        self._target = None 
        self._pred = None
        self._error = None
        
        if IDgraph!="":
            self._launch_live_session(sess,IDgraph,lID)
        
    def _model(self):
        """ <DocString> """
        def __cell():
            # https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/  \
            # python/ops/rnn_cell_impl.py
            # Tensorflow defaults to no peephole LSTM:
            # http://www.bioinf.jku.at/publications/older/2604.pdf
            if self.RRN_type =="LSTM":
                rnn_cell = tf.contrib.rnn.LSTMCell(self.size_hidden_layer)#,state_is_tuple=True
            elif self.RRN_type =="RNNv":
                rnn_cell = tf.contrib.rnn.BasicRNNCell(self.size_hidden_layer)
            return tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self._dropout)
        
        cell = tf.contrib.rnn.MultiRNNCell([__cell() for _ in range(self.L)])#,state_is_tuple=True
        # Out is of dimension [batch_size, max_time, cell_state_size]
        out, self._state = tf.nn.dynamic_rnn(cell=cell, inputs=self._inputs, dtype=tf.float32)
        out = tf.reshape(out, [-1, self.size_hidden_layer])
        # market change prediction
        pred_mc = tf.nn.sigmoid(tf.matmul(out, self._parameters["w_lg"]) + self._parameters["b_lg"])
        # market direction prediction
        pred_md = tf.nn.softmax(tf.matmul(out, self._parameters["w_sm2"]) + self._parameters["b_sm2"])
        # Market gain prediction
        pred_mg = tf.nn.softmax(tf.matmul(out, self._parameters["w"]) + self._parameters["b"])
        # concatenate predictions
        if self.commonY==0:
            pred = tf.reshape(pred_mg, [-1, self.seq_len, 
                                        self.size_output_layer])
        elif self.commonY==3:
            pred = tf.concat([tf.reshape(pred_mc, [-1, self.seq_len, 1]),
                              tf.reshape(pred_md, [-1, self.seq_len, 2]),
                              tf.reshape(pred_mg, [-1, self.seq_len, 
                                                  self.size_output_layer])],2)
        return pred
        
    def _init_parameters(self):
        """ Define graph to run. """
        
        self._inputs = tf.placeholder(tf.float32, [None, self.seq_len, self.nFeatures], name="input")
        self._dropout = tf.placeholder(tf.float32)
        # Define variables for final fully connected layer.
        self._parameters = {}
        w_ho = tf.Variable(tf.truncated_normal([self.size_hidden_layer, 
                                                self.size_output_layer], stddev=0.01), name="fc_w")
        b_o = tf.Variable(tf.constant(0.1, shape=[self.size_output_layer]), name="fc_b")
        self._parameters["w"] = w_ho
        self._parameters["b"] = b_o
        
        # build weights for logistic regression (market change estimator)
        w_lg = tf.Variable(tf.truncated_normal([self.size_hidden_layer,1], stddev=0.01), name="wlg")
        b_lg = tf.Variable(tf.constant(0.1, shape=[1]), name="blg")
        self._parameters["w_lg"] = w_lg
        self._parameters["b_lg"] = b_lg
        
        # build weights for second softmax (direction estimator)
        w_sm2 = tf.Variable(tf.truncated_normal([self.size_hidden_layer,2], stddev=0.01), name="wsm2")
        b_sm2 = tf.Variable(tf.constant(0.1, shape=[2]), name="bsm2")
        self._parameters["w_sm2"] = w_sm2
        self._parameters["b_sm2"] = b_sm2
        #self.var = tf.get_variable("var", [1])
        self._pred = self._model()

        return None
    
    def _tf_repeat(self, repeats):
        """
        Args:
    
        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, 
        length must be the same as the number of dimensions in input
    
        Returns:
        
        A Tensor. Has the same type as input. Has the shape of tensor.shape*repeats
        """
        repeated_tensor = self._target[:,:,0:1]
        for i in range(repeats-1):
            repeated_tensor = tf.concat([repeated_tensor,self._target[:,:,0:1]],2)

        return repeated_tensor
        
    def _compute_loss(self):
        """ <DocString> """
        # Create model
        self._target = tf.placeholder("float", [None,self.seq_len, 
                                    self.size_output_layer+self.commonY], 
                                    name="target")
        # Define cross entropy loss.
        # first target bit for mc (market change)
        if self.commonY == 3:
            loss_mc = tf.reduce_sum(self._target[:,:,0:1] * -tf.log(self._pred[:,:,0:1])+
                                    (1-self._target[:,:,0:1])* -
                                    tf.log(1 - self._pred[:,:,0:1]), [1, 2])/self.seq_len
            # second and third bits for md (market direction)
            loss_md = -tf.reduce_sum(self._tf_repeat(2)*(self._target[:,:,1:3] * 
                                     tf.log(self._pred[:,:,1:3])), [1, 2])/self.seq_len
            # last 5 bits for market gain output
            loss_mg = -tf.reduce_sum(self._tf_repeat(self.size_output_layer)*
                                 (self._target[:,:,3:] * 
                                  tf.log(self._pred[:,:,3:])), [1, 2])/self.seq_len
            self._loss = tf.reduce_mean(
                loss_mc+loss_md+loss_mg,
                name="loss_nll")
        elif self.commonY == 0:
#            loss_mc = tf.reduce_sum(0 * -tf.log(self._pred[:,:,0:1])+
#                                    (1-1)* -
#                                    tf.log(1 - self._pred[:,:,0:1]), [1, 2])
#            # second and third bits for md (market direction)
#            loss_md = -tf.reduce_sum(0*(self._target[:,:,1:3] * 
#                                     tf.log(self._pred[:,:,1:3])), [1, 2])
            # last size_output_layer bits for market gain output
            loss_mg = -tf.reduce_sum(self._target *tf.log(self._pred), [1, 2])/self.seq_len
            self._loss = tf.reduce_mean(loss_mg, name="loss_nll")
        self._optim = tf.train.AdamOptimizer(self.lR0).minimize(
            self._loss,
            name="adam_optim")
        
        self._error = self._loss
        
    def _load_graph(self,ID, epoch, live=""):
        """ <DocString> """
        print("Loading graph...")
        # Create placeholders
        if os.path.exists(local_vars.weights_directory+ID+"/"):
            if epoch == -1:# load last
                # get last saved epoch
                epoch = int(sorted(os.listdir(local_vars.weights_directory+ID+"/"))[-2])
            
            strEpoch = "{0:06d}".format(epoch)
            self._saver.restore(self._sess, local_vars.weights_directory+ID+"/"+
                                strEpoch+"/"+strEpoch)
            #print("var : %s" % self.var.eval())
            print("Parameters loaded. Epoch "+str(epoch))
            
        elif live=="":
            self._init_op = tf.global_variables_initializer()
            self._sess.run(self._init_op)
        else:
            raise ValueError("Graph "+ID+"E"+str(epoch)+" does not exist. Revise IDweights and IDepoch")
        
        
        return epoch+1
    
    def save_output_fn(self, ID, output, cost, method='pickle', tag='NNIO'):
        """ Save output """
        dirfilename = local_vars.IO_directory+tag+ID
        if method=='pickle':
            pickle.dump({'output':output, 
                         'cost':cost}, open( dirfilename+'.p', 'wb'))
        elif method=='hdf5':
            f_IO = h5py.File(dirfilename,'w')
            # init IO data sets
            outputh5 = f_IO.create_dataset('output', output.shape, dtype=float)
            outputh5[:] = output
            f_IO.close()
            print("Output saved in disk")
    
    def _save_graph(self, ID, epoch_cost, epoch, weights_directory):
        """ <DocString> """
        cost = {}
        if os.path.exists(weights_directory+ID+"/")==False:
            os.mkdir(weights_directory+ID+"/")
        if os.path.exists(weights_directory+ID+"/cost.p"):
            cost = pickle.load( open( local_vars.weights_directory+ID+"/cost.p", "rb" ))
        cost[ID+str(epoch)] = epoch_cost
        pickle.dump(cost, open( weights_directory+ID+"/cost.p", "wb" ))
        strEpoch = "{0:06d}".format(epoch)
        save_path = self._saver.save(self._sess, weights_directory+ID+"/"+
                                     strEpoch+"/"+strEpoch)
        
        print("Parameters saved")
        return save_path
    
    def _launch_live_session(self, sess, IDgraph, lID):
        """ <DocString> """
        IDweights = IDgraph[:lID]
        epoch = IDgraph[lID:]
        graph = tf.Graph()
        with graph.as_default():
            self._init_parameters()
            self._saver = tf.train.Saver(max_to_keep = None)
            self._sess = tf.Session()
            self._load_graph(IDweights,int(epoch),live="y")
            
        return None
    
    def run_live_session(self, X):
        """ <DocString> """
        test_data_feed = {
            self._inputs: X,
            self._dropout: 1.0
        }
        softMaxOut = self._sess.run([self._pred], test_data_feed)[0]
        
        return softMaxOut
    
    def test(self, sess, data, IDresults, IDweights, alloc, save_results, 
             trainOrTest, filename_IO, startFrom=-1, IDIO='', data_format='', DTA=[], 
             save_journal=False, endAt=-1):
        """ 
        Test RNN network with y_c bits
        """
        import pandas as pd
        self._sess = sess
        
        self._init_parameters()
        self._compute_loss()
        self._saver = tf.train.Saver(max_to_keep = None) # define save object
        
        resultsDir = local_vars.results_directory
        
        #TR,lastSaved = loadTR(IDresults,resultsDir,saveResults,startFrom)
        if endAt==-1:
            lastTrained = int(sorted(os.listdir(local_vars.weights_directory+IDweights+"/"))[-2])
        else:
            lastTrained = endAt
        
        # load train cost function evaluations
        costs = {}
        if os.path.exists(local_vars.weights_directory+IDweights+"/cost.p"):
            costs = pickle.load( open( local_vars.weights_directory+IDweights+"/cost.p", "rb" ))
        else:
            pass
        
        # load IO info
        #filename_IO = '../RNN/IO/'+'IO_'+IDIO+'.hdf5'
        f_IO = h5py.File(filename_IO,'r')
        #X_test = f_IO['X'][:]
        Y_test = f_IO['Y'][:]
        
        n_chunks = int(np.ceil(Y_test.shape[0]/alloc))
        
        if startFrom == -1:
            startFrom = get_last_saved_epoch(resultsDir, IDresults, self.seq_len)+1
        best_ROI = 0.0
        best_ROI_profile = pd.DataFrame() # best ROI profile
        BR_ROIs = pd.DataFrame() # best results per ROI
        best_GROI = 0.0
        best_GROI_profile = pd.DataFrame()
        BR_GROIs = pd.DataFrame() # best results per ROI
        best_sharpe = 0.0
        best_sharpe_profile = pd.DataFrame() # best sharpe profile
        BR_sharpes = pd.DataFrame() # best results per sharpe ratio
        # load models and test them
        for epoch in range(startFrom,lastTrained+1):

            self._load_graph(IDweights,epoch)
            print("Epoch "+str(epoch)+" of "+str(lastTrained)+". Getting output...")
#            J_train = self._loss.eval()
            softMaxOut = np.zeros((0,self.seq_len,self.size_output_layer+self.commonY))
            for chunck in range(n_chunks):
                print("Chunck "+str(chunck+1)+" of "+str(n_chunks))
                # build input/output data
                test_data_feed = {
                    self._inputs: f_IO['X'][chunck*alloc:(chunck+1)*alloc],
                    self._target: f_IO['Y'][chunck*alloc:(chunck+1)*alloc],
                    self._dropout: 1.0
                }
                J_test, smo = self._sess.run([self._error,self._pred], test_data_feed)
                
                softMaxOut = np.append(softMaxOut,smo,axis=0)
                #print(softMaxOut.shape)
            print("Getting results")
            new_ROI, tBR_GROI, tBR_ROI, tBR_sharpe = evaluate_RNN(data, 
                                                                  self, 
                                                                  Y_test, 
                                                                  DTA, 
                                                                  IDresults, 
                                                                  IDweights, 
                                                                  J_test, 
                                                                  softMaxOut, 
                                                                  save_results,
                                                                  costs, 
                                                                  epoch, 
                                                                  resultsDir, 
                                                                  lastTrained, 
                                                                  save_journal=save_journal)
            BR_GROIs = BR_GROIs.append(tBR_GROI)
            BR_ROIs = BR_ROIs.append(tBR_ROI)
            BR_sharpes = BR_sharpes.append(tBR_sharpe)
            # update best ROI
            if new_ROI>best_ROI:
                best_ROI = new_ROI
                best_ROI_profile = tBR_ROI
            if tBR_GROI.rGROI.iloc[0]>best_GROI:
                best_GROI = tBR_GROI.rGROI.iloc[0]
                best_GROI_profile = tBR_GROI
            # update best sharpe ratio
            if tBR_sharpe.sharpe.iloc[0]>best_sharpe:
                best_sharpe = tBR_sharpe.sharpe.iloc[0]
                best_sharpe_profile = tBR_sharpe
            # save best result
            if best_ROI_profile.shape[0]>0:
                
                # save best results
                save_best_results(BR_GROIs, BR_ROIs, BR_sharpes, resultsDir, IDresults, save_results)
        # end of for epoch in range(lastSaved,lastTrained+1):
        if best_ROI_profile.shape[0]>0:
            print("Best GROI: "+str(best_GROI))
            print("Profile best GROI:")
            print(best_GROI_profile.to_string())
            print("\nBest ROI: "+str(best_ROI))
            print("Profile best ROI:")
            print(best_ROI_profile.to_string())
            print("Best Sharpe ratio: "+str(best_sharpe))
            print("Profile best Sharpe ratio:")
            print(best_sharpe_profile.to_string())
            # save best results
            #save_best_results(BR_ROIs, BR_sharpes, resultsDir, IDresults, save_results)
    
    def test2(self, sess, config, alloc, filename_IO,
             startFrom=-1, data_format='', DTA=[],  from_var=False):
        """ 
        Test RNN network with y_c bits
        """
        IDresults = config['IDresults']
        IDweights = config['IDweights']
        startFrom = config['startFrom']
        endAt = config['endAt']
        weights_directory = local_vars.weights_directory
        
        tic = time.time()
        self._sess = sess
        results_directory = local_vars.results_directory
        self._init_parameters()
        self._compute_loss()
        self._saver = tf.train.Saver(max_to_keep = None) # define save object
        
        #TR,lastSaved = loadTR(IDresults,resultsDir,saveResults,startFrom)
        if endAt==-1:
            lastTrained = int(sorted(os.listdir(weights_directory+IDweights+"/"))[-2])
        else:
            lastTrained = endAt
        
        # load train cost function evaluations
        costs = {}
        if os.path.exists(weights_directory+IDweights+"/cost.p"):
            costs = pickle.load( open( weights_directory+IDweights+"/cost.p", "rb" ))
            if 'cost_name' in config:
                cost_name = config['cost_name']
            else:
                cost_name = IDweights
        else:
            raise ValueError("File cost.p does not exist.")
        # load IO info
        #filename_IO = '../RNN/IO/'+'IO_'+IDIO+'.hdf5'
        f_IO = h5py.File(filename_IO,'r')
        #X_test = f_IO['X'][:]
        Y_test = f_IO['Y'][:]
        
        n_chunks = int(np.ceil(Y_test.shape[0]/alloc))
        
        if startFrom == -1:
            startFrom = get_last_saved_epoch2(results_directory, IDresults)+1
        import math
        if self.commonY==3:
            results_filename, costs_filename = init_results_dir(results_directory, IDresults)
        elif self.commonY==0:
            t_indexes = [str(t) if t<self.seq_len else 'mean' for t in range(self.seq_len+1)]
            results_filename, costs_filename, performance_filename = init_results_mg_dir(results_directory, 
                                                            IDresults, 
                                                            self.size_output_layer,
                                                            t_indexes,
                                                            get_performance=True)
        # load models and test them
        for epoch in range(startFrom,lastTrained+1):
            if math.isnan(costs[cost_name+str(epoch)]):
                print("J_train=NaN BREAK!")
                break
            self._load_graph(IDweights, epoch)
            print("Epoch "+str(epoch)+" of "+str(lastTrained)+". Getting output...")
#            J_train = self._loss.eval()
            softMaxOut = np.zeros((0,self.seq_len,self.size_output_layer+self.commonY))
            t_J_test = 0
            for chunck in tqdm(range(n_chunks)):
                #tqdm.write("Chunck "+str(chunck+1)+" of "+str(n_chunks))
                # build input/output data
                test_data_feed = {
                    self._inputs: f_IO['X'][chunck*alloc:(chunck+1)*alloc],
                    self._target: f_IO['Y'][chunck*alloc:(chunck+1)*alloc],
                    self._dropout: 1.0
                }
                J_test, smo = self._sess.run([self._error,self._pred], test_data_feed)
                t_J_test += J_test
                softMaxOut = np.append(softMaxOut,smo,axis=0)
                #print(softMaxOut.shape)
            t_J_test = t_J_test/n_chunks
            print("Getting results")
            if self.commonY==3:
                get_results(config, self, Y_test, DTA, 
                            t_J_test, softMaxOut, costs, epoch, lastTrained, results_filename,
                            costs_filename,
                            from_var=from_var)
            elif self.commonY==0:
                get_results_mg(config, Y_test, softMaxOut, costs, epoch, 
                               J_test, costs_filename, results_filename,
                               performance_filename,
                               get_performance=True, DTA=DTA)
        
        print("Total time for testing: "+str((time.time()-tic)/60)+" mins.\n")
            
        return None
    
    def train(self, sess, nChunks, weights_directory,
              ID='', logFile='',IDIO='', data_format='', 
              filename_IO='', aloc=2**17):
        """ Call to train.
        args: train_data, train object defined in sets.
        args: test_data, test data defined in sets.
        """
        self._sess = sess
        print("ID = "+ID)
        self._init_parameters()
        self._compute_loss()
        self._saver = tf.train.Saver(max_to_keep = None) # define save object
        #dirName = "../RNN/weights/"
        epochStart = self._load_graph(ID,-1)
        #epochStart = 0
        print("Training from epoch "+str(epochStart)+" till "+str(epochStart+self._num_epochs-1))
        # check if data format is HDF5
        if data_format=='hdf5':
            # load IO info
            #aloc = 2**17
            if filename_IO=='':
                filename_IO = '../RNN/IO/'+'IO_'+IDIO+'.hdf5'
            f_IO = h5py.File(filename_IO,'r')
            X = f_IO['X']
            Y = f_IO['Y']
            
        ticT = time.time()
        try:
            for epoch in range(epochStart,epochStart+self._num_epochs):
                
                #chunk,m_t,epoch_cost, resume = get_resume_info(dirName,ID)
                chunk = 0
                epoch_cost = 0
                m_t = 0
                t_minibatches = 0
                while chunk<nChunks:
                    # load X and Y
                    #print("Chunck {0:d} out of {1:d}".format(chunk, nChunks))
                    tic = time.time()
                    if data_format=='hdf5':
                        X_train = X[chunk*aloc:(chunk+1)*aloc,:,:]
                        #print("From "+str(chunk*aloc)+" to "+str((chunk+1)*aloc))
                        #print(X_train.shape)
                        Y_train = Y[chunk*aloc:(chunk+1)*aloc,:,:]
                    else:
                        X_train = pickle.load( open( "../RNN/IO/X"+str(chunk)+"_"+IDIO+"tr.p", "rb" ))
                        #print(X_train.shape)
                        Y_train = pickle.load( open( "../RNN/IO/Y"+str(chunk)+"_"+IDIO+"tr.p", "rb" ))
                    
                    #plt.plot(X_train[:,0,0])
                    #print("EEOOOOO")
                    train_data = trainData(X_train, Y_train)
                    n_batches = int(np.ceil(train_data.m/self.miniBatchSize))
                    
                    for mB in tqdm(range(n_batches),mininterval=1):
                        #print(self._batch_size)
                        this_mB = range(mB*self.miniBatchSize,np.min([train_data.m,(mB+1)*self.miniBatchSize]))
                        #print(this_mB)
                        batch_data = train_data.data[this_mB]
                        batch_target = train_data.target[this_mB]
                        #batch = train_data.sample(self._batch_size)
                        
                        train_data_feed = {
                            self._inputs: batch_data,
                            self._target: batch_target,
                            self._dropout: self.keep_prob_dropout
                        }
                        train_loss, _ = self._sess.run(
                            [self._loss, self._optim],
                            train_data_feed
                        )
                        epoch_cost += train_loss
                        t_minibatches += 1
                        
                    toc = time.time()
                    print("Time training Chunck {0:d} of {1:d}: {2:.0f} s".format(chunk,nChunks-1,np.floor(toc-tic)))
                    chunk = chunk+1
                    m_t += train_data.m
                    
                #num_minibatches = int(m_t / self.miniBatchSize)
                epoch_cost = epoch_cost/t_minibatches
                print ("Cost after epoch %i: %f. Av cost %f" % (epoch, train_loss, epoch_cost))
                
                print(dt.datetime.strftime(dt.datetime.now(),"%H:%M:%S")+
                  " Total time training: "+"{0:.2f}".format(np.floor(toc-ticT)/60)+"m")
    
                if ID!='':
                    self._save_graph(ID, epoch_cost, epoch, weights_directory)
            # end of for epoch in range(epochStart,epochStart+self._num_epochs):
        except KeyboardInterrupt:
            f_IO.close()
            raise KeyboardInterrupt
            
    def evaluate(self, sess, X, Y, params={}, tOt='tr'):
        """ Evaluate the model at one epoch and return weights and output """
        if 'IDweights' in params:
            IDweights = params['IDweights']
        else:
            IDweights = '0001001A'
        if 'IDresults' in params:
            IDresults = params['IDresults']
        else:
            IDresults = '1001001AC'
        if 'epoch' in params:
            epoch = params['epoch']
        else:
            epoch = 35
        if 'alloc' in params:
            alloc = params['alloc']
        else:
            alloc = 2**10
        if 'output_shape' in params:
            output_shape = params['output_shape']
        else:
            output_shape = (Y.shape[0], self.seq_len, self.size_output_layer+self.commonY)
        if 'save_output' in params:
            save_output = params['save_output']
        else:
            save_output = True
        m = output_shape[0]
        loss = 0
        n_chunks = int(np.ceil(m/alloc))
        self._sess = sess
        self._init_parameters()
        self._compute_loss()
        self._saver = tf.train.Saver(max_to_keep = None)
        self._load_graph(IDweights, epoch)
        print("Epoch "+str(epoch)+". Evaluating...")
        output = np.zeros(output_shape)
        for chunck in tqdm(range(n_chunks)):
            #tqdm.write("Chunck "+str(chunck+1)+" of "+str(n_chunks))
            # build input/output data
            test_data_feed = {
                self._inputs: X[chunck*alloc:(chunck+1)*alloc],
                self._target: Y[chunck*alloc:(chunck+1)*alloc],
                self._dropout: 1.0
            }
            c_loss, output[chunck*alloc:(chunck+1)*alloc] = sess.run([self._error,self._pred], test_data_feed)
            loss += c_loss
                #print(softMaxOut.shape)
        loss = loss/n_chunks
        print("loss = {}".format(loss))
        if save_output:
            if tOt=='tr':
                ID = IDweights
            elif tOt=='te':
                ID = IDresults
            self.save_output_fn(ID, output, loss, method='hdf5')
        return output, loss