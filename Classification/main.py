import os
import cv2
import sys
import random
import datetime
import numpy as np
np.random.seed(1338)
random.seed(9000)
import argparse
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
from task_generator import TaskGenerator
from meta_learner import MetaLearner

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def restore_model(model, weights_dir):
    print ('Reload weights from: {}'.format(weights_dir))
    ckpt = tf.train.Checkpoint(maml_model=model)
    latest_weights = tf.train.latest_checkpoint(weights_dir)
    ckpt.restore(latest_weights)
    return model

def copy_model(model, x):
    copied_model = MetaLearner()
    copied_model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def loss_fn(y, pred_y):
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y, pred_y))

def accuracy_fn(y, pred_y):
    accuracy = tf.keras.metrics.Accuracy()
    _ = accuracy.update_state(tf.argmax(pred_y, axis=1), tf.argmax(y, axis=1))
    return accuracy.result()

def compute_loss(model, x, y, loss_fn=loss_fn):
    _, pred_y = model(x)
    loss = loss_fn(y, pred_y)
    return loss, pred_y

def compute_gradients(model, x, y, loss_fn=loss_fn):
    with tf.GradientTape() as tape:
        _, pred = model(x)
        loss = loss_fn(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
    return grads

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def distribute_weights(models):
    # Single model training
    if len(models) == 1:
        pass

    weights = [model.trainable_variables for model in models]

    for layer in zip(*weights):
        new_layer = list()
        for m in range(len(models)):
            new_layer.append(tf.math.reduce_mean(tf.gather(layer, np.where(adj_mat[m])[0]), axis=0))
        for ml, model_layer in enumerate(layer):
            model_layer.assign(new_layer[ml])

def maml_train(models, batch_generator, strategy):
    n_way = args.n_way
    k_shot = args.k_shot
    total_batches = args.total_batches
    meta_batchsz = args.meta_batchsz
    meta_batchsz_test=args.meta_batchsz_test
    update_steps = args.update_steps
    update_steps_test = args.update_steps_test
    test_steps = args.test_steps
    # ckpt_steps = args.ckpt_steps
    print_steps = args.print_steps
    inner_lr = args.inner_lr
    meta_lr = args.meta_lr
    # ckpt_dir = args.ckpt_dir + args.dataset+'/{}way{}shot/'.format(n_way, k_shot)
    print ('Start training process of {}-way {}-shot {}-query problem'.format(args.n_way, args.k_shot, args.k_query))
    print ('{} steps, inner_lr: {}, meta_lr:{}, meta_batchsz:{}'.format(total_batches, inner_lr, meta_lr, meta_batchsz))

    meta_optimizers = [tf.keras.optimizers.Adam(learning_rate=args.meta_lr, name='meta_optimizer') for _ in models]

    # test_min_losses = []
    # test_max_accs = []

    def _maml_finetune_step(model, test_set):
        batch_loss = [0 for _ in range(meta_batchsz_test)]
        batch_acc = [0 for _ in range(meta_batchsz_test)]
        # copied_model = MetaLearner.hard_copy(model, args)
        for idx, task in enumerate(test_set):
            copied_model = MetaLearner.hard_copy(model, args)
            # Slice task to support set and query set
            support_x, support_y, query_x, query_y = task
            # Update fast weights several times
            for i in range(update_steps_test):
                # Set up inner gradient tape, watch the copied_model.inner_weights
                with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                    # we only want inner tape watch the fast weights in each update steps
                    inner_tape.watch(copied_model.inner_weights)
                    inner_loss, _ = compute_loss(copied_model, support_x, support_y)
                inner_grads = inner_tape.gradient(inner_loss, copied_model.inner_weights)
                copied_model = MetaLearner.meta_update(copied_model, args, alpha=inner_lr, grads=inner_grads)

            task_loss, task_pred = compute_loss(copied_model, query_x, query_y, loss_fn=loss_fn)
            task_acc = accuracy_fn(query_y, task_pred)
            batch_loss[idx] += task_loss
            batch_acc[idx] += task_acc
        # if saving memory is necessary, delete the copied_model
        del copied_model
        return batch_loss, batch_acc
    
    # MAML Train Step
    def _maml_train_step(model, i,  batch_set):
        # Set up recorders for every batch
        batch_loss = [0 for _ in range(meta_batchsz)]
        batch_acc = [0 for _ in range(meta_batchsz)]
        # Set up outer gradient tape, only watch model.trainable_variables. Because GradientTape only auto record trainable_variables of model. But the copied_model.inner_weights is tf.Tensor, so they won't be automatically watched
        with tf.GradientTape() as outer_tape:
            # Use the average loss over all tasks in one batch to compute gradients
            for idx, task in enumerate(batch_set):
                # Set up copied model
                copied_model = model
                # Slice task to support set and query set
                support_x, support_y, query_x, query_y = task
                # Update fast weights several times
                for i in range(update_steps):
                    # Set up inner gradient tape, watch the copied_model.inner_weights
                    with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                        # We only want inner tape watch the fast weights in each update steps
                        inner_tape.watch(copied_model.inner_weights)
                        inner_loss, _ = compute_loss(copied_model, support_x, support_y)
                    inner_grads = inner_tape.gradient(inner_loss, copied_model.inner_weights)
                    copied_model = MetaLearner.meta_update(copied_model, args, alpha=inner_lr, grads=inner_grads)
                # Compute task loss & accuracy on the query set
                task_loss, task_pred = compute_loss(copied_model, query_x, query_y, loss_fn=loss_fn)
                task_acc = accuracy_fn(query_y, task_pred)
                batch_loss[idx] += task_loss
                batch_acc[idx] += task_acc
            # Compute mean loss of the whole batch
            mean_loss = tf.reduce_mean(batch_loss)
        # Compute second order gradients
        outer_grads = outer_tape.gradient(mean_loss, model.trainable_variables)
        apply_gradients(meta_optimizers[i], outer_grads, model.trainable_variables)
        # Return result of one maml train step
        return batch_loss, batch_acc
            
    # Main loop
    start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print ('Start at {}'.format(start))
    # For each epoch update model total_batches times
    start = time.time()
    losses = []
    accs = []
    test_losses = []
    test_accs = []

    for epoch in range(NUM_OF_EPOCHS):
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        for step in range(total_batches):
            batch_loss_models = []
            batch_acc_models = []
            ## Use if comparing CTA and ATC
            #if strategy == 'CTA': 
                # distribute_weights(models)
            for i, model in enumerate(models):
                # Get a batch data
                batch_set = batch_generator.train_batch()
                # batch_generator.print_label_map()
                # Run maml train step
                batch_loss, batch_acc = _maml_train_step(model, i, batch_set)
                # Record Loss
                batch_loss_models.append(tf.reduce_mean(batch_loss).numpy())
                batch_acc_models.append(tf.reduce_mean(batch_acc).numpy())
            losses.append(np.mean(batch_loss_models))
            accs.append(np.mean(batch_acc_models))

            ## Use if comparing CTA and ATC
            #if strategy == 'ATC':
    
            distribute_weights(models) # adjust adjacency matrix
            
            # Printing train result
            if step % print_steps == 0 and step > 0:
                average_loss = losses[-1]
                average_acc = accs[-1]
                print ('[STEP. {}] Average Task Loss: {}; Average Task Accuracy: {}; Time to run {} Steps: {}'.format(step, average_loss, average_acc, print_steps, time.time()-start))
                start = time.time()
                # Uncomment to see the sampled folders of each task
                # train_ds.print_label_map()

            # Evaluate the model
            if (step+1) % test_steps == 0: # and step > 0:
                test_set = batch_generator.test_batch()
                test_loss_models = []
                test_acc_models = []
                for i, model in enumerate(models):
                    test_loss, test_acc = _maml_finetune_step(model, test_set)
                    test_loss_models.append(tf.reduce_mean(test_loss).numpy())
                    test_acc_models.append(tf.reduce_mean(test_acc).numpy())
                # Record test history
                test_losses.append(((step+1)+(epoch*total_batches),np.mean(test_loss_models)))
                test_accs.append(((step+1)+(epoch*total_batches),np.mean(test_acc_models)))
                print ('Test Loss: {}, Test Accuracy: {}'.format(test_losses[-1], test_accs[-1]))
                print ('=====================================================================')
            # Meta train step
    
    # Record training history
    os.chdir(args.his_dir)
    plt.figure()
    losses_plot, = plt.plot(losses, label = "Train Acccuracy", color='coral')
    accs_plot, = plt.plot(accs,'--',label = "Train Loss", color='royalblue')
    # accs_plot = plt.plot(accs, '--',color='blue')
    plt.legend([losses_plot, accs_plot], ['Train Loss', 'Train Accuracy'])
    plt.title('{} {} {}-Way {}-Shot MAML Training Process'.format(strategy, args.dataset, n_way, k_shot))
    plt.savefig('{} {}-{}-way-{}-shot-train.png'.format(strategy, args.dataset, n_way, k_shot))

    '''
    plt.figure()
    test_losses_plot, = plt.plot(test_losses, label="Test loss", color='coral')
    test_accs_plot, = plt.plot(test_accs, '--', label="Test accuracy", color='royalblue')
    plt.legend([test_losses_plot, test_accs_plot], ['Test Loss', 'Test Accuracy'])
    plt.title('{} {} {}-Way {}-Shot MAML Test Process'.format(strategy, args.dataset, n_way, k_shot))
    plt.savefig('{} {}-{}-way-{}-shot-test.png'.format(strategy, args.dataset, n_way, k_shot))
    '''

    train_hist = '{}-{}-{}-way{}-shot-train.txt'.format(strategy, args.dataset, n_way,k_shot)
    acc_hist = '{}-{}-{}-way{}-shot-acc.txt'.format(strategy, args.dataset, n_way,k_shot)
    test_acc_hist = '{}-{}-{}-way{}-shot-acc-test.txt'.format(strategy, args.dataset, n_way,k_shot)
    test_loss_hist = '{}-{}-{}-way{}-shot-loss-test.txt'.format(strategy, args.dataset, n_way,k_shot)

    # Save History
    f = open(train_hist, 'w')
    for i in range(len(losses)):
        f.write(str(losses[i]) + '\n')
    f.close()

    f = open(acc_hist, 'w')
    for i in range(len(accs)):
        f.write(str(accs[i]) + '\n')
    f.close()

    f = open(test_acc_hist, 'w')
    for i in range(len(test_accs)):
        f.write(str(test_accs[i]) + '\n')
    f.close()

    f = open(test_loss_hist, 'w')
    for i in range(len(test_losses)):
        f.write(str(test_losses[i]) + '\n')
    f.close()

    os.chdir(dir_path)

    return models, test_losses, test_accs

def eval_model(model, batch_generator, num_steps=None):
    if num_steps is None:
        num_steps = (0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    # Generate a batch data
    batch_set = batch_generator.test_batch()
    # Use a copy of current model
    copied_model = model
    # Initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.inner_lr)
    
    # task_losses = [0, 0, 0, 0]
    # task_accs = [0, 0, 0, 0]

    loss_res = [[] for _ in range(len(batch_set))]
    acc_res = [[] for _ in range(len(batch_set))]
    
    # Record test result
    if 0 in num_steps:
        for idx, task in enumerate(batch_set):
            support_x, support_y, query_x, query_y = task
            loss, pred = compute_loss(model, query_x, query_y)
            acc = accuracy_fn(query_y, pred)
            # task_losses[idx] += loss.numpy()
            # task_accs[idx] += acc.numpy()
            loss_res[idx].append((0, loss.numpy()))
            acc_res[idx].append((0, acc.numpy()))
        # print ('Before any update steps, test result:')
        # print ('Task losses: {}'.format(task_losses))
        # print ('Task accuracies: {}'.format(task_accs))
    # Test for each task
    for idx, task in enumerate(batch_set):
        print ('========== Task {} =========='.format(idx+1))
        support_x, support_y, query_x, query_y = task
        for step in range(1, np.max(num_steps)+1):
            with tf.GradientTape() as tape:
                loss, pred = compute_loss(model, support_x, support_y)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Test on query set
            qry_loss, qry_pred = compute_loss(model, query_x, query_y)
            qry_acc = accuracy_fn(query_y, qry_pred)
            # Record result
            if step in num_steps:
                loss_res[idx].append((step, qry_loss.numpy()))
                acc_res[idx].append((step, qry_acc.numpy()))
                print ('After {} steps update'.format(step))
                print ('Task losses: {}'.format(qry_loss.numpy()))
                print ('Task accs: {}'.format(qry_acc.numpy()))
                print ('---------------------------------')
    
    for idx in range(len(batch_set)):
        l_x=[]
        l_y=[]
        a_x = []
        a_y=[]
        # plt.subplot(2, 2, idx+1)
        plt.figure()
        for j in range(len(num_steps)):
            l_x.append(loss_res[idx][j][0])
            l_y.append(loss_res[idx][j][1])
            a_x.append(acc_res[idx][j][0])
            a_y.append(acc_res[idx][j][1])
        plt.plot(l_x, l_y, 'x', color='coral')
        plt.plot(a_x, a_y, '*', color='royalblue')
        # plt.annotate('Loss After 1 Fine Tune Step: %.2f'%l_y[1], xy=(l_x[1], l_y[1]), xytext=(l_x[1]-0.2, l_y[1]-0.2))
        # plt.annotate('Accuracy After 1 Fine Tune Step: %.2f'%a_y[1], xy=(a_x[1], a_y[1]), xytext=(a_x[1]-0.2, a_y[1]-0.2))
        plt.plot(l_x, l_y, linestyle='--', color='coral')
        plt.plot(a_x, a_y, linestyle='--', color='royalblue')
        plt.xlabel('Fine Tune Step', fontsize=12)
        plt.fill_between(a_x, [a+0.1 for a in a_y], [a-0.1 for a in a_y], facecolor='royalblue', alpha=0.3)
        legend=['Fine Tune Points','Fine Tune Points','Loss', 'Accuracy']
        plt.legend(legend)
        plt.title('Task {} Fine Tuning Process'.format(idx+1))
        plt.show()



if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--mode', type=str, help='train or test', default='train')
    # Dataset options
    argparse.add_argument('--dataset', type=str, help='Dataset used to train model', default='miniimagenet')
    # Task options
    argparse.add_argument('--n_way', type=int, help='Number of classes used in classification (e.g. 5-way classification)', default=5)
    argparse.add_argument('--k_shot', type=int, help='Number of images in support set', default=5)
    argparse.add_argument('--k_query', type=int, help='Number of images in query set(For Omniglot, equal to k_shot)', default=10)
    argparse.add_argument('--num_agents', type=int, help='Number of agents in the network', default=6)
    # argparse.add_argument('--connectivity', type=float, help='Probability of connectivity in Erdos-Renyi setting', default=0.6)
    argparse.add_argument('--strategy', type=str, help='Strategy used for diffusion(e.g ATC)', default='ATC')
    # Model options
    argparse.add_argument('--num_filters', type=int, help='Number of filters in the convolution layers (32 for MiniImagenet, 64 for Omniglot)', default=32)
    argparse.add_argument('--with_bn', type=bool, help='Turn True to add BatchNormalization layers in neural net', default=True)
    # Training options
    argparse.add_argument('--meta_batchsz', type=int, help='Number of tasks in one batch', default=2)
    argparse.add_argument('--meta_batchsz_test', type=int, help='Number of tasks in one batch, finetune step', default=25)
    argparse.add_argument('--update_steps', type=int, help='Number of inner gradient updates for each task', default=5)
    argparse.add_argument('--update_steps_test', type=int, help='Number of inner gradient updates for each task while testing', default=10)
    argparse.add_argument('--inner_lr', type=float, help='Learning rate of inner update steps, the step size alpha in the algorithm', default=0.01) # 0.1 or 0.4 for omniglot
    argparse.add_argument('--meta_lr', type=float, help='Learning rate of meta update steps, the step size beta in the algorithm', default=1e-3)
    argparse.add_argument('--total_batches', type=int, help='Total update steps for each epoch', default=1000)
    # Log options
    argparse.add_argument('--ckpt_steps', type=int, help='Number of steps for recording checkpoints', default=20000)
    argparse.add_argument('--test_steps', type=int, help='Number of steps for evaluating model', default=200)
    argparse.add_argument('--print_steps', type=int, help='Number of steps for prints result in the console', default=1)
    argparse.add_argument('--ckpt_dir', type=str, help='Path to the checkpoint directory', default='weights/')
    argparse.add_argument('--his_dir', type=str, help='Path to the training history directory', default='histories/')
    # Generate args
    args = argparse.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    NUM_OF_EPOCHS = 3
    NUM_OF_AGENTS = args.num_agents

    # p = args.connectivity
    # equal_initialization = False

    # -----------------------------------
    # Creating the adjacency matrix for diffusion manually
    adj_mat = np.eye(NUM_OF_AGENTS, dtype=int)

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

    # -----------------------------------

    # Initialize task generator
    # batch_generator = TaskGenerator(args)

    if args.mode == 'train':
        
        # model = restore_model(model, '../../weights/{}/{}way{}shot'.format(args.dataset, args.n_way, args.k_shot))
        '''
        if equal_initialization:
            # Initialize after you hard copy...
            model_0 = MetaLearner.initialize(MetaLearner(args=args))
            # Diffusion
            models_diff = [MetaLearner.hard_copy(model_0,args) for x in range(NUM_OF_AGENTS)]
            models_diff, test_losses_diff, test_accs_diff = maml_train(models_diff, batch_generator, args.strategy)
            # Non-cooperative
            models_nocoop = [MetaLearner.hard_copy(model_0,args) for x in range(NUM_OF_AGENTS)]
            models_nocoop, test_losses_nocoop, test_accs_nocoop = maml_train(models_nocoop, batch_generator, 'Non_cooperative')
            # Centralized
            NUM_OF_AGENTS = 1
            models_cent = [MetaLearner.hard_copy(model_0,args) for x in range(NUM_OF_AGENTS)]
            models_cent, test_losses_cent, test_accs_cent = maml_train(models_cent, batch_generator, 'Centralized')
        '''

        # Diffusion algorithm
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        batch_generator = TaskGenerator(args)
        models_diff = [MetaLearner.initialize(MetaLearner(args=args)) for x in range(NUM_OF_AGENTS)]
        models_diff, test_losses_diff, test_accs_diff = maml_train(models_diff, batch_generator, args.strategy)

        # Non-cooperative
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        batch_generator = TaskGenerator(args)

        adj_mat = np.eye(NUM_OF_AGENTS)
        models_nocoop = [MetaLearner.initialize(MetaLearner(args=args)) for x in range(NUM_OF_AGENTS)]
        models_nocoop, test_losses_nocoop, test_accs_nocoop = maml_train(models_nocoop, batch_generator,
                                                                             'Non_cooperative')
        # Centralized
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        batch_generator = TaskGenerator(args)
        adj_mat = np.eye(NUM_OF_AGENTS)
        for i in range(NUM_OF_AGENTS):
            for j in range(NUM_OF_AGENTS):
                adj_mat[i][j] = 1
        models_cent = [MetaLearner.initialize(MetaLearner(args=args)) for x in range(NUM_OF_AGENTS)]
        models_cent, test_losses_cent, test_accs_cent = maml_train(models_cent, batch_generator, 'Centralized')

        #Comparison of test loss

        os.chdir(args.his_dir)
        n_way = args.n_way
        k_shot = args.k_shot

        int_res = {'Non-cooperative': test_losses_nocoop, 'Diffusion': test_losses_diff, 'Centralized': test_losses_cent}

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
            # plt.xticks(num_steps)
            legend.append(name)
        plt.xlabel('Training iteration')
        plt.ylabel('Average loss on test tasks')
        plt.legend(legend)
        plt.title('Test Losses {} {}-Way {}-Shot Training Process'.format(args.dataset, n_way, k_shot))
        plt.savefig('Comparison-of-3-losses-{}-{}-way-{}-shot-test.png'.format( args.dataset, n_way, k_shot))
        plt.show()

        # Accuracy plots

        int_res = {'Non-cooperative': test_accs_nocoop, 'Diffusion': test_accs_diff, 'Centralized': test_accs_cent}

        plt.figure()
        legend = []
        colors = ['red', 'blue', 'yellow']
        for i, name in enumerate(int_res):
            x = []
            y = []
            for n, acc in int_res[name]:
                x.append(n)
                y.append(acc)
            plt.plot(x, y, color=colors[i], marker='x', linestyle='--')
            # plt.xticks(num_steps)
            legend.append(name)
        plt.xlabel('Training iteration')
        plt.ylabel('Average accuracy on test tasks')
        plt.legend(legend)
        plt.title('Test Accuracies {} {}-Way {}-Shot Training Process'.format(args.dataset, n_way, k_shot))
        plt.savefig('Comparison-of-3-accuracies-{}-{}-way-{}-shot-test.png'.format(args.dataset, n_way, k_shot))
        plt.show()

        os.chdir(dir_path)


    elif args.mode == 'test':
        model = restore_model(model, 'weights/{}/{}way{}shot'.format(args.dataset, args.n_way, args.k_shot))
        if args.dataset == 'miniimagenet':
            eval_model(model, batch_generator, num_steps=range(9))
        elif args.dataset == 'omniglot':
            eval_model(model, batch_generator, num_steps=range(9))
