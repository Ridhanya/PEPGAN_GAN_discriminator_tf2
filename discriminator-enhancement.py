import tensorflow as tf
import  numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

def linear(input_, output_size):
	shape = input_.get_shape().as_list()
	if len(shape) != 2:
		raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
	if not shape[1]:
		raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
	input_size = shape[1]
	matrix = tf.Variable( [output_size, input_size], dtype=input_.dtype,name="Matrix")
	bias_term = tf.Variable( [output_size], dtype=input_.dtype,name="Bias")

	return tf.matmul(input_, tf.transpose(matrix)) + bias_term
	

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu):
	for idx in range(num_layers):
		g = f(linear(input_, size))
		t = tf.sigmoid(linear(input_, size) + bias)
		output = t * g + (1. - t) * input_
		input_ = output
	return output

def feature(feature_input,dropout_keep_prob,vocab_size,dis_emb_dim,filter_sizes,num_filters,dis_emb_dim,num_filters_total):
	w_fe=tf.random_uniform(shape=[vocab_size + 1, dis_emb_dim],minval= -1.0, maxval=1.0)
	embedded_chars = tf.nn.embedding_lookup(W_fe, Feature_input + 1)
	embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
	pooled_outputs = []
	for filter_size, num_filter in zip(filter_sizes, num_filters):
		filter_shape = [filter_size, dis_emb_dim, 1, num_filter]
		W=tf.random.truncated_normal(filter_shape, stddev=0.1)
		b=tf.constant(0.1, shape=[num_filter])
		conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID",name="conv-%s" % filter_size)
		h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-%s" % filter_size)
		pooled = tf.nn.max_pool(h,ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool-%s" % filter_size)
		pooled_outputs.append(pooled)
	h_pool = tf.concat(pooled_outputs, 3)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
	h_drop = tf.nn.dropout(h_highway,dropout_keep_prob)

	return h_drop


def classification( D_input,num_filters_total,num_classes):
	W=tf.Variable(tf.random.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
	b_d = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
	D_l2_loss += tf.nn.l2_loss(W_d)
	D_l2_loss += tf.nn.l2_loss(b_d)
	scores = tf.add(tf.matmul(D_input, W_d), b_d, name="scores")
	ypred_for_auc = tf.nn.softmax(scores)
	predictions = tf.argmax(scores, 1, name="predictions")
	return scores, predictions, ypred_for_auc


def discriminator(input_y,feature_input,sequence_length, num_classes, vocab_size,dis_emb_dim,filter_sizes, num_filters,batch_size,hidden_dim,start_token,goal_out_size,step_size,l2_reg_lambda=0.0, dropout_keep_prob=0.75):
	D_l2_loss = tf.constant(0.0)
	D_feature= FeatureExtractor(feature_input,dropout_keep_prob,vocab_size,dis_emb_dim,filter_sizes,num_filters,dis_emb_dim,num_filters_total)
	D_scores, D_predictions,ypred_for_auc = classification(D_feature,num_filters_total,num_classes)
	losses = tf.nn.softmax_cross_entropy_with_logits(logits=D_scores, labels=input_y)
	D_loss = tf.reduce_mean(losses) + l2_reg_lambda * D_l2_loss
	d_optimizer = tf.keras.optimizers.Adam( epsilon=5e-5)
	D_grads_and_vars = d_optimizer.compute_gradients(D_loss, aggregation_method=2)
	D_train_op = d_optimizer.apply_gradients(D_grads_and_vars)





