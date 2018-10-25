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
import pandas as pd
#import matplotlib.pyplot as plt
from results import evaluate_RNN,save_best_results,get_last_saved_epoch

class trainData:
    def __init__(self,X,Y):
        self.data = X
        self.target = Y
        self.m = X.shape[0]
#_, seq_len, input_size = train.data.shape
#print(train.data.shape)


class modelRNN(object):
    
    def __init__(self,
                 data,
                 size_hidden_layer=400,
                 L=3,
                 size_output_layer=5,
                 keep_prob_dropout=0.9,
                 miniBatchSize=32,
                 outputGain=0.5,
                 commonY=0,
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
        self.nFeatures = data.nFeatures*len(data.channels)#int(data.nFeatures*data.nEventsPerStat/data.movingWindow+data.nFeaturesAuto) # size of input image, (16x8 flattened to 128)
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
        self._n_batches = None
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
        loss_mc = tf.reduce_sum(self._target[:,:,0:1] * -tf.log(self._pred[:,:,0:1])+
                                (1-self._target[:,:,0:1])* -
                                tf.log(1 - self._pred[:,:,0:1]), [1, 2])
        # second and third bits for md (market direction)
        loss_md = -tf.reduce_sum(self._tf_repeat(2)*(self._target[:,:,1:3] * 
                                 tf.log(self._pred[:,:,1:3])), [1, 2])
        # last 5 bits for market gain output
        loss_mg = -tf.reduce_sum(self._tf_repeat(self.size_output_layer)*
                                 (self._target[:,:,3:] * 
                                  tf.log(self._pred[:,:,3:])), [1, 2])
        self._loss = tf.reduce_mean(
            loss_mc+loss_md+loss_mg,
            name="loss_nll")
        self._optim = tf.train.AdamOptimizer(self.lR0).minimize(
            self._loss,
            name="adam_optim")
        
        self._error = self._loss
        
    def _load_graph(self,ID,epoch,live=""):
        """ <DocString> """
        # Create placeholders
        if os.path.exists("../RNN/weights/"+ID+"/"):
            if epoch == -1:# load last
                # get last saved epoch
                epoch = int(sorted(os.listdir("../RNN/weights/"+ID+"/"))[-2])
            
            strEpoch = "{0:06d}".format(epoch)
            self._saver.restore(self._sess, "../RNN/weights/"+ID+"/"+
                                strEpoch+"/"+strEpoch)
            #print("var : %s" % self.var.eval())
            print("Parameters loaded. Epoch "+str(epoch))
            
        elif live=="":
            self._init_op = tf.global_variables_initializer()
            self._sess.run(self._init_op)
        else:
            print("Error. Graph does not exist. Revise IDweights and IDepoch")
            error()
        
        
        return epoch+1
    
    def _save_graph(self,ID,epoch_cost,epoch):
        """ <DocString> """
        cost = {}
        if os.path.exists("../RNN/weights/"+ID+"/")==False:
            os.mkdir("../RNN/weights/"+ID+"/")
        if os.path.exists("../RNN/weights/"+ID+"/cost.p"):
            cost = pickle.load( open( "../RNN/weights/"+ID+"/cost.p", "rb" ))
        cost[ID+str(epoch)] = epoch_cost
        pickle.dump(cost, open( "../RNN/weights/"+ID+"/cost.p", "wb" ))
        strEpoch = "{0:06d}".format(epoch)
        save_path = self._saver.save(self._sess, "../RNN/weights/"+ID+"/"+
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
    
    def test(self, sess, data, IDresults, IDweights, nChunks, save_results, 
             trainOrTest, startFrom=-1, IDIO='', data_format='', DTA=[], 
             save_journal=False, endAt=-1):
        """ 
        Test RNN network with y_c bits
        """
        self._sess = sess
        
        self._init_parameters()
        self._compute_loss()
        self._saver = tf.train.Saver(max_to_keep = None) # define save object
        
        resultsDir = "../RNN/results/"
        
        #TR,lastSaved = loadTR(IDresults,resultsDir,saveResults,startFrom)
        if endAt==-1:
            lastTrained = int(sorted(os.listdir("../RNN/weights/"+IDweights+"/"))[-2])
        else:
            lastTrained = endAt
        
        # load train cost function evaluations
        costs = {}
        if os.path.exists("../RNN/weights/"+IDweights+"/cost.p"):
            costs = pickle.load( open( "../RNN/weights/"+IDweights+"/cost.p", "rb" ))
        else:
            pass
        
        # load IO info
        filename_IO = '../RNN/IO/'+'IO_'+IDIO+'.hdf5'
        f_IO = h5py.File(filename_IO,'r')
        #X_test = f_IO['X'][:]
        Y_test = f_IO['Y'][:]
        alloc = 500000
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

        return None
    
    def train(self, sess, nChunks, ID='', logFile='',IDIO='', data_format='', filename_IO='', aloc=2**17):
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
                    self._n_batches = int(np.ceil(train_data.m/self.miniBatchSize))
                    
                    for mB in range(self._n_batches):
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
                    self._save_graph(ID,epoch_cost,epoch)
            # end of for epoch in range(epochStart,epochStart+self._num_epochs):
        except KeyboardInterrupt:
            f_IO.close()
            end()
    