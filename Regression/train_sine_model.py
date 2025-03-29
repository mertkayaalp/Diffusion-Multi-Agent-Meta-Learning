import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import random
import numpy as np
np.random.seed(1338)
import networkx as nx
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
from sinusoid_generator import SinusoidGenerator, generate_train_dataset, generate_test_dataset
from sine_model import SineModel

tf.keras.backend.set_floatx('float64')

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def copy_model(model, x):
    copied_model = SineModel()
    copied_model.forward(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def loss_fn(y, pred_y):
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y, pred_y))

def compute_loss(model, x, y , loss_fn=loss_fn):
    logits = model.forward(x)
    mse = loss_fn(logits, y)
    return mse, logits

def compute_gradients(model, x, y, loss_fn=loss_fn):
    with tf.GradientTape() as tape:
        loss, logits = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def train_step(x, y, model, optimizer):
    tensor_x, tensor_y = np_to_tensor((x, y))
    gradients, loss = compute_gradients(model, tensor_x, tensor_y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss

def distribute_weights(models):

    weights = [model.trainable_variables for model in models]
    for layer in zip(*weights):
        new_layer = list()
        for m in range(len(models)):
            new_layer.append(tf.math.reduce_mean(tf.gather(layer, np.where(adj_mat[m])[0]), axis=0))
        for ml, model_layer in enumerate(layer):
            model_layer.assign(new_layer[ml])

def regular_train(model, train_ds, epochs=1, lr=0.001, log_steps=1000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        losses = []
        total_loss = 0
        start = time.time()
        for i, sinusoid_generator in enumerate(train_ds):
            x, y = sinusoid_generator.batch()
            loss = train_step(x, y, model, optimizer)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            losses.append(curr_loss)
            
            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {:.2f} seconds'.format(
                    i, curr_loss, log_steps, time.time() - start))
                start = time.time()
        plt.plot(losses)
        plt.title('SGD Loss Vs Time steps with respect to 10 points')
        plt.show()
    return model


def maml_train(models, num_agents, train_dataset, inter_set, strategy="ATC", epochs=5, lr_inner=0.01, log_steps=1000, test_steps=2000):
    optimizers = [tf.keras.optimizers.Adam() for _ in models]
    intermed_test_errors = []
    for epoch in range(epochs):
        losses = []
        start = time.time()
        for z in range(len(train_dataset[0])):
            if strategy == "CTA":
                distribute_weights(models)
            total_loss = 0
            for i, model in enumerate(models):
                x = train_dataset[i][z][0][0]
                y = train_dataset[i][z][0][1]
                model.forward(x)
                with tf.GradientTape() as test_tape:
                    with tf.GradientTape() as train_tape:
                        train_loss, _ = compute_loss(model, x, y)
                    gradients = train_tape.gradient(train_loss, model.trainable_variables)
                    k=0
                    model_copy = copy_model(model, x)
                    for j in range(len(model_copy.layers)):
                        model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                    tf.multiply(lr_inner, gradients[k]))
                        model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                                    tf.multiply(lr_inner, gradients[k+1]))
                        k+=2
                    x = train_dataset[i][z][1][0] #using a different batch for outer update
                    y = train_dataset[i][z][1][1]
                    test_loss, logits = compute_loss(model_copy, x, y)
                    total_loss += test_loss  # will average over agents
                gradients = test_tape.gradient(test_loss, model.trainable_variables)
                optimizers[i].apply_gradients(zip(gradients, model.trainable_variables))

            loss = total_loss / num_agents
            losses.append(loss)

            if strategy == "ATC" or "Centralized":
                distribute_weights(models)

            if z % log_steps == 0 and z > 0:
                print('Step {}: loss = {}, Time to run {} steps = {}'.format(z, loss, log_steps, time.time() - start))
                start = time.time()

            if (z+1) % test_steps == 0:
                optimizer_int = tf.keras.optimizers.SGD(learning_rate=lr_inner)
                ave_loss_tasks = 0
                ave_logits_tasks = 0
                for datatask in inter_set:
                    x = datatask[0][0]
                    y = datatask[0][1]
                    xtest = datatask[1][0]
                    ytest = datatask[1][1]
                    ave_loss = 0
                    ave_logits = 0
                    for j, model in enumerate(models):
                        copy_int = copy_model(model, x)
                        train_step(x, y, copy_int, optimizer_int)
                        loss, logits = compute_loss(copy_int, xtest, ytest)
                        ave_loss = ave_loss + loss
                        ave_logits = ave_logits + logits
                    ave_loss = ave_loss / len(models)
                    ave_logits = ave_logits / len(models)
                    ave_loss_tasks = ave_loss_tasks + ave_loss
                    ave_logits_tasks = ave_logits_tasks + ave_logits
                ave_loss_tasks = ave_loss_tasks / len(inter_set)
                ave_logits_tasks = ave_logits_tasks / len(inter_set)
                intermed_test_errors.append((z+1+epoch*len(train_dataset[0]), ave_loss_tasks))


        plt.plot(losses)
        plt.title('MAML Loss Vs Time steps with respect to 10 points')
        plt.savefig(strategy)
        plt.show()
    return models, intermed_test_errors

def plot_model_comparison_to_average(model, ds, model_name='neural network', K=10):
    
    sinu_generator = SinusoidGenerator(K=K)
    # Calculate the average prediction
    avg_pred = []
    for i, sinusoid_generator in enumerate(ds):
        x, y = sinusoid_generator.equally_spaced_samples()
        avg_pred.append(y)
    
    x, _ = sinu_generator.equally_spaced_samples()
    plt.figure()
    avg_plot, = plt.plot(x, np.mean(avg_pred, axis=0), '--')

    # Calculate model prediction
    model_pred = model.forward(tf.convert_to_tensor(x))
    model_plot, = plt.plot(x, model_pred.numpy())
    
    plt.legend([avg_plot, model_plot], ['Average', model_name])
    plt.title('Model Prediction compared to the mean of training sine waves actual Y')
    plt.show()
                

def eval_sine_test(models, optimizers, datas, dataplot, num_steps=(0, 1, 10)):

    fit_res = []
    
    # If 0 in fits we log the loss before any training
    if 0 in num_steps:
        ave_loss_tasks = 0
        ave_logits_tasks = 0
        for i in range(len(dataplot)):
            x_test = dataplot[i][0]
            y_test = dataplot[i][1]
            x = datas[i][0]
            y = datas[i][1]
            # tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))
            ave_loss = 0
            ave_logits = 0
            for model in models[i]:
                loss, logits = compute_loss(model, x_test, y_test)
                ave_loss = ave_loss + loss
                ave_logits = ave_logits + logits
            ave_loss = ave_loss/len(models[i])
            ave_logits = ave_logits/len(models[i])
            ave_loss_tasks = ave_loss_tasks + ave_loss
            ave_logits_tasks = ave_logits_tasks + ave_logits
        ave_loss_tasks = ave_loss_tasks/len(dataplot)
        ave_logits_tasks = ave_logits_tasks/len(dataplot)
        fit_res.append((0, ave_logits_tasks, ave_loss_tasks))

    for step in range(1, np.max(num_steps) + 1):
        ave_loss_tasks = 0
        ave_logits_tasks = 0
        for i in range(len(dataplot)):
            x_test = dataplot[i][0]
            y_test = dataplot[i][1]
            x = datas[i][0]
            y = datas[i][1]
            # tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))
            ave_loss = 0
            ave_logits = 0
            for j, model in enumerate(models[i]):
                train_step(x, y, model, optimizers[i][j])
                loss, logits = compute_loss(model, x_test, y_test)
                ave_loss = ave_loss + loss
                ave_logits = ave_logits + logits
            ave_loss = ave_loss/len(models[i])
            ave_logits = ave_logits/len(models[i])
            ave_loss_tasks = ave_loss_tasks + ave_loss
            ave_logits_tasks = ave_logits_tasks + ave_logits
        ave_loss_tasks = ave_loss_tasks/len(dataplot)
        ave_logits_tasks = ave_logits_tasks/len(dataplot)
        if step in num_steps:
            fit_res.append(
                (
                    step,
                    ave_logits_tasks,
                    ave_loss_tasks
                )
            )

    return fit_res


def eval_sinewave_for_test(models, final_test_dataset, datas, num_steps=(0, 1, 10), lr=0.01, plot=True, name=None):

    dataplot = []
    for sinusoid_generator in final_test_dataset:
        # Generate equally spaced samples for ploting
        x_test, y_test = np_to_tensor(sinusoid_generator.equally_spaced_samples(100))
        _dataplot = [x_test, y_test]
        dataplot.append(_dataplot)

    x = datas[0][0]

    # Copy the model so we can use the same model multiple times
    copied_models = [[copy_model(model, x) for model in models] for _ in datas]

    # Use SGD for this part of training as described in the paper
    optimizers = [[tf.keras.optimizers.SGD(learning_rate=lr) for model in models] for _ in datas]
    
    # Run training and log fit results
    fit_res = eval_sine_test(copied_models, optimizers, datas, dataplot, num_steps)
    
    return fit_res

def compare_models(noncoop, difmaml, single_agent, final_test_dataset, num_steps=list(range(10)),
                                intermediate_plot=True, marker='x', linestyle='--',figure_name = None):
    '''Comparing the loss of noncooperative, diffusion and centralized strategies.
    Fits the models for a new task (new sine wave) and then plot
    the loss of both models along `num_steps` interactions.
    Args:
        noncoop, difmaml, single_agent: Already trained MAMLs.
        num_steps: Number of steps to be logged.
        intermediate_plot: If True plots intermediate plots from
            `eval_sinewave_for_test`.
        marker: Marker used for plotting.
        linestyle: Line style used for plotting.
    '''
    datas = []
    for sinusoid_generator in final_test_dataset:
        x, y = np_to_tensor(sinusoid_generator.batch())
        _data = [x, y]
        datas.append(_data)

    # x_test, y_test = sinusoid_generator.equally_spaced_samples(100)

    fit_noncoop = eval_sinewave_for_test(noncoop, final_test_dataset, datas, num_steps=num_steps, plot=intermediate_plot,name='noncoopsine')
    fit_difmaml = eval_sinewave_for_test(difmaml, final_test_dataset, datas, num_steps=num_steps, plot=intermediate_plot,name='atcsine')
    fit_single_agent = eval_sinewave_for_test(single_agent, final_test_dataset, datas, num_steps=num_steps, plot=intermediate_plot,name='centralsine')
    
    fit_res = {'Non-cooperative': fit_noncoop, 'Diffusion': fit_difmaml, 'Centralized': fit_single_agent}

    # Plot of learning curve in test time
    plt.figure()
    legend = []
    colors = ['red', 'blue', 'yellow']
    for i, name in enumerate(fit_res):
        x = []
        y = []
        for n, _, loss in fit_res[name]:
            x.append(n)
            y.append(loss)
        plt.plot(x, y, color=colors[i], marker=marker, linestyle=linestyle)
        plt.xticks(num_steps)
        legend.append(name)
    plt.title('Test error vs number of gradient steps')
    plt.xlabel('Number of gradient steps')
    plt.ylabel('Average loss of test tasks')
    plt.legend(legend)
    plt.savefig('Test_time_comparison')
    plt.show()

if __name__ == '__main__':

    TRAINING_SIZE = 4000
   
    NUM_OF_AGENTS = 6

    ## If the graph is Erdos-Renyi: 
    # p = 0.6  # Connectivity probability
    # mygraph = nx.erdos_renyi_graph(NUM_OF_AGENTS, p, seed=None, directed=False)
    # adj_mat = nx.to_numpy_matrix(mygraph)
    # adj_mat = np.asarray(adj_mat)
    # for i in range(NUM_OF_AGENTS):
       # adj_mat[i][i] = 1

    adj_mat = np.eye(NUM_OF_AGENTS)

    # Manual creation of the graph 

    adj_mat[0][2] = 1
    adj_mat[2][0] = 1

    adj_mat[1][2] = 1
    adj_mat[2][1] = 1

    adj_mat[2][3] = 1
    adj_mat[3][2] = 1

    adj_mat[2][4] = 1
    adj_mat[4][2] = 1

    adj_mat[3][4] = 1
    adj_mat[4][3] = 1

    adj_mat[4][5] = 1
    adj_mat[5][4] = 1

    # Handling training dataset
    train_dataset_generators = generate_train_dataset(num_agents=NUM_OF_AGENTS,K=10, train_size= TRAINING_SIZE)

    train_dataset = []
    for agent_set in train_dataset_generators:
        agent_data = []
        for sinus_generator in agent_set:
            xin, yin = np_to_tensor(sinus_generator.batch())
            datain = [xin, yin]
            xout, yout = np_to_tensor(sinus_generator.batch())
            dataout = [xout, yout]
            data_task = [datain, dataout]
            agent_data.append(data_task)
        train_dataset.append(agent_data)

    # Handling validation set
    intermediate_test_generators = generate_test_dataset(test_size = 1000)
    inter_test_set = []
    for sinus_generator in intermediate_test_generators:
        xin, yin = np_to_tensor(sinus_generator.batch())
        datain = [xin, yin]
        x_test, y_test = np_to_tensor(sinus_generator.equally_spaced_samples(100))
        datatest = [x_test, y_test]
        data_task = [datain, datatest]
        inter_test_set.append(data_task)

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    modelsdif = [SineModel() for x in range(NUM_OF_AGENTS)]
    diffusion_models, dif_int_test = maml_train(modelsdif, NUM_OF_AGENTS, train_dataset, inter_test_set, strategy="ATC")

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    adj_mat = np.eye(NUM_OF_AGENTS)
    modelsno = [SineModel() for x in range(NUM_OF_AGENTS)]
    noncoop_models, noncoop_int_test = maml_train(modelsno, NUM_OF_AGENTS, train_dataset, inter_test_set, strategy="No_cooperation")

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    modelscent = [SineModel() for x in range(NUM_OF_AGENTS)]
    adj_mat = np.eye(NUM_OF_AGENTS)
    for i in range(NUM_OF_AGENTS):
        for j in range(NUM_OF_AGENTS):
            adj_mat[i][j] = 1
    central_models, central_int_test = maml_train(modelscent, NUM_OF_AGENTS, train_dataset, inter_test_set, strategy="Centralized")

    # ---------------------------------------------

    # Plotting the test error vs iteration

    int_res = {'Non-cooperative': noncoop_int_test, 'Diffusion': dif_int_test, 'Centralized': central_int_test}

    plt.figure()
    legend = []
    colors = ['red', 'blue', 'yellow']
    for i, name in enumerate(int_res):
        x = []
        y = []
        for n, loss in int_res[name]:
            x.append(n)
            y.append(loss)
        plt.plot(x, y, color=colors[i], marker='x', linestyle='--')
        #plt.xticks(num_steps)
        legend.append(name)
    plt.title('Test error vs Iteration')
    plt.xlabel('Training iteration')
    plt.ylabel('Average loss of test tasks')
    plt.legend(legend)
    plt.savefig('Training_time_comparison')
    plt.show()

    # ---------------------------------------------

    final_test_dataset = generate_test_dataset(test_size = 1000)

    compare_models(noncoop_models, diffusion_models, central_models, final_test_dataset)