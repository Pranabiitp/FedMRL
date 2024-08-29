


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)






import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


def create_clients(data_dict):
    '''
    Return a dictionary with keys as client names and values as data and label lists.
    
    Args:
        data_dict: A dictionary where keys are client names, and values are tuples of data and labels.
                    For example, {'client_1': (data_1, labels_1), 'client_2': (data_2, labels_2), ...}
    
    Returns:
        A dictionary with keys as client names and values as tuples of data and label lists.
    '''
    return data_dict






def test_model(X_test, Y_test,  model, comm_round):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
#     logits = model.predict(X_test)
    #print(logits)
    loss,accuracy=model.evaluate(X_test,Y_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, accuracy, loss))
    return accuracy, loss





def avg_weights(scaled_weight_list):
    '''Return the average of the listed scaled weights.'''
    num_clients = len(scaled_weight_list)
    
    if num_clients == 0:
        return None  # Handle the case where the list is empty
        
    avg_grad = list()
    
    # Get the sum of gradients across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) / num_clients
        avg_grad.append(layer_mean)
        
    return avg_grad





from keras.preprocessing.image import ImageDataGenerator
import numpy as np
def load_img_data(path):
    img_size = (224, 224)
    datagen = ImageDataGenerator()
   
    test_data = datagen.flow_from_directory(
        directory=path,
        target_size=img_size,
        class_mode='categorical',  # Set class_mode to 'categorical' for one-hot encoding
        batch_size=32,
        shuffle=False,  # Set shuffle to False to maintain order
#         classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Specify the classes explicitly
    )

    # Load images and one-hot encode labels into numpy arrays
    images, one_hot_labels = [], []
    for batch in test_data:
        images.extend(batch[0])  # Load images from the batch
        one_hot_labels.extend(batch[1])  # Extract one-hot encoded labels from the batch
        if len(images) >= len(test_data.filenames):
            break

    return np.array(images), np.array(one_hot_labels)

train1,label1=load_img_data('client_1')
train2,label2=load_img_data('client_2')
train3,label3=load_img_data('client_3')
train4,label4=load_img_data('client_4')


# In[6]:


#load test data
test=np.load("test.npy")
one_hot_labels=np.load("one_hot_labels.npy")
print("import sucessfull")


# In[25]:


test=test/255
train1=train1/255
train2=train2/255
train3=train3/255
train4=train4/255


# In[3]:


label=one_hot_labels


# In[4]:


client_data1 = {
    'client1': (test, label),
    'client2': (test, label),
    'client3': (test, label),
    'client4': (test, label)
    
}
#create clients
test_batched = create_clients(client_data1)
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
    
}
#create clients
clients_batched = create_clients(client_data2)


# In[10]:


import numpy as np
import random
import tensorflow as tf

class QMIXAgent:
    def __init__(self, state_dim, action_dim, n_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = 0.99
        self.lr = 0.001

        # Q-networks for each agent
        self.q_networks = [self.build_q_network() for _ in range(n_agents)]

        # Centralized critic network
        self.critic_network = self.build_critic_network()

        # Mixing network
        self.mixing_network = self.build_mixing_network()

        # Optimizers
        self.q_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.lr) for _ in range(n_agents)]
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.mixing_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid')
        ])
        return model

    def build_critic_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model

    def build_mixing_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='sigmoid')
        ])
        return model

    def select_actions(self, states):
        states_np = [np.array(state) for state in states]
        concatenated_states = tf.concat(states_np, axis=1)  # Concatenate states along axis 1
        q_values = [q_net(concatenated_states) for q_net in self.q_networks]
        q_values = tf.stack(q_values, axis=1)
        action = tf.reduce_mean(q_values, axis=1)
        action = tf.nn.sigmoid(action)
        return action[0]  # Return the action value
    def close_summary_writer(self):
        self.file_writer.close()
        
    def train(self, states, actions, rewards, next_states, dones):
     with tf.GradientTape(persistent=True) as tape:
        # Convert experiences to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Concatenate states along axis 1
        concatenated_states = tf.concat(states, axis=1)
        concatenated_states = tf.reshape(concatenated_states, (-1, self.state_dim * self.n_agents))

        q_values = [q_net(concatenated_states) for q_net in self.q_networks]
        q_values = tf.stack(q_values, axis=1)

        centralized_values = self.critic_network(concatenated_states)

        mixed_q_values = self.mixing_network(centralized_values)

        q_targets = rewards + self.gamma * tf.reduce_max(mixed_q_values, axis=1) * (1 - dones)
        actions = tf.cast(actions, dtype=tf.int32)

        individual_q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_dim, dtype=q_values.dtype), axis=-1)

        td_errors = tf.square(q_targets - individual_q_values)
        mean_loss = tf.reduce_mean(td_errors)

    # Compute gradients for Q-networks
     q_gradients = [tape.gradient(mean_loss, q_net.trainable_variables) for q_net in self.q_networks]

    # Apply gradients for Q-networks
     for i in range(self.n_agents):
        if any(g is None for g in q_gradients[i]):
            continue  # Skip None gradients
        self.q_optimizers[i].apply_gradients(zip(q_gradients[i], self.q_networks[i].trainable_variables))

    # Compute gradients for the critic network
     critic_gradients = tape.gradient(mean_loss, self.critic_network.trainable_variables)

    # Apply gradients for the critic network
     if critic_gradients is not None:
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))

    # Compute gradients for the mixing network
     mixing_gradients = tape.gradient(mean_loss, self.mixing_network.trainable_variables)

    # Apply gradients for the mixing network
     if mixing_gradients is not None:
        self.mixing_optimizer.apply_gradients(zip(mixing_gradients, self.mixing_network.trainable_variables))

    # Delete the tape to free resources
     del tape

        



        


# In[11]:


num_clients=4
state_size = 4 * num_clients
action_size = 1  # A single continuous action representing μ
# Define the action space range
action_low = 0.0
action_high = 1.0
qm_agent= QMIXAgent(4, action_size, 4)


# In[12]:


import numpy as np
from collections import Counter 

def count_classes_one_hot(labels):
    # Convert one-hot encoded labels to class indices
    class_indices = np.argmax(labels, axis=1)
    
    # Use Counter to count occurrences of each class
    class_counts = Counter(class_indices)
    
    print(class_counts)



count_classes_one_hot(label1)
count_classes_one_hot(label2)
count_classes_one_hot(label3)
count_classes_one_hot(label4)


# In[ ]:


import numpy as np

def calculate_Ek(Nc, Nk):
    Ek = 0.0
    for c in range(7):
        
            term = (Nc[c] / Nk) * np.log(Nc[0] / Nk)
            Ek += term
    return Ek

# Example usage:
def calculate_pk(Nk, N):
    
    return Nk/N



Nc1 =[896,  584,  568,  337,  248, 100,  75]
Nc2 =[913,  575,  569,  343,  247,  101,  76]
Nc3 =[ 919,  585,  579,  338,  251,  101,  76]
Nc4 =[910,  569,  558,  340,  248,  101,  77]
Nk1=sum(Nc1)
Nk2=sum(Nc2)
Nk3=sum(Nc3)
Nk4=sum(Nc4)
# print(Nk2)
N=Nk1+Nk2+Nk3+Nk4


# In[13]:


E1=calculate_Ek(Nc1, Nk1)
E2=calculate_Ek(Nc2, Nk2)
E3=calculate_Ek(Nc3, Nk3)
E4=calculate_Ek(Nc4, Nk4)
P1=calculate_pk(Nk1, N)
P2=calculate_pk(Nk2, N)
P3=calculate_pk(Nk3, N)
P4=calculate_pk(Nk4, N)


# In[14]:


import numpy as np

# Example initial sigma value and sigma decay rate value
initial_sigma = 2.0  
sigma_decay_rate = 0.1  

# Get the number of parameters in the global model
# num_parameters = np.sum([np.prod(w.shape) for w in global_model.get_weights()])

# Define the shape of the SOM grid to match the dimensionality of the global model parameters
# som_shape = (5, 5)  # Define the number of rows and columns for the SOM grid

# Initialize the SOM weights with random values
som_weights_shape = (2,3, 101)
som_weights = np.random.rand(*som_weights_shape)

# Verify the shape of the initialized SOM weights
print("Shape of SOM weights:", som_weights.shape)


# Initialize the SOM weights with random values
som_weights = np.random.rand(*som_shape, num_parameters)
def calculate_similarity_metrics(global_model_weights, client_model_weights_list):
    '''Calculate cosine similarity metrics between global model and client models.'''
    similarity_metrics = []
    global_weights_flat = flatten_weights(global_model_weights)
    
    for client_model_weights in client_model_weights_list:
        client_weights_flat = flatten_weights(client_model_weights)
        
        similarity = np.dot(client_weights_flat, global_weights_flat) /(np.linalg.norm(client_weights_flat) * np.linalg.norm(global_weights_flat))
        normalized_similarity = (similarity + 1) / 2  # Normalize to range [0, 1]
        similarity_metrics.append(normalized_similarity)
#         similarity_metrics.append(similarity)
    print("done_calculate_similarity_metrics")
    return np.array(similarity_metrics)

def pad_or_reshape(arr, target_shape):
    """
    Pad or reshape the input array or list of arrays to match the target shape.
    If the input array shape matches the target shape, it returns the array unchanged.
    If the input array shape is smaller than the target shape, it pads the array with zeros.
    If the input array shape is larger than the target shape, it reshapes the array to match the target shape.
    """
    if isinstance(arr, list):  # If arr is a list
        # Pad or reshape each array in the list
        return [pad_or_reshape(a, target_shape) for a in arr]
    elif isinstance(arr, np.ndarray):  # If arr is a numpy array
        if arr.shape == target_shape:
            return arr
        elif np.prod(arr.shape) < np.prod(target_shape):
            # Pad the array with zeros to match the target shape
            padded_arr = np.zeros(target_shape)
            padded_arr[:arr.shape[0]] = arr.reshape(-1)  # Reshape arr before assignment
            return padded_arr
        else:
            # Reshape the array to match the target shape
            return arr.reshape(target_shape)
    else:
        raise ValueError("Input must be a numpy array or a list of numpy arrays")




def update_weights_som(local_weight_list, global_model, som_weights, normalized_similarity_metrics, initial_sigma, sigma_decay_rate):
    for local_weights in local_weight_list:
        # Calculate distances between local weights and SOM weights
        print(som_weights.shape)
        distances = np.linalg.norm(som_weights - local_weights, axis=-1)

        # Find the best matching unit (BMU)
        bmu_indices = np.unravel_index(np.argmin(distances), distances.shape)

        # Update SOM weights based on BMU and neighborhood function
        for i in range(som_weights.shape[0]):
            for j in range(som_weights.shape[1]):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_indices))
                influence = np.exp(-distance_to_bmu / (2 * initial_sigma ** 2))
                som_weights[i, j] += influence * initial_sigma * (local_weights - som_weights[i, j])

    # Calculate weights based on SOM weights and similarity metrics
    weights = np.zeros(len(local_weight_list))
    for i, local_weights in enumerate(local_weight_list):
        # Calculate distances between local weights and SOM weights
        distances = np.linalg.norm(som_weights - local_weights, axis=-1)

        # Compute weights using SOM distances and normalized similarity metrics
        weights[i] = np.sum(np.exp(-distances / (2 * initial_sigma ** 2)) * normalized_similarity_metrics)

    # Normalize weights
    weights /= np.sum(weights)

    return weights
    







import tensorflow as tf

import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf
def flatten_weights(weights_list):
    """Flatten a list of weight arrays."""
    flattened_weights = np.concatenate([tf.reshape(w, [-1]).numpy() for w in weights_list])
    print("done_flatten_weights")
    return flattened_weights    



def weighted_aggregation(local_weight_list, weights):
    '''Weighted aggregation of local weights.'''
    aggregated_weights = []
    for layer_weights in zip(*local_weight_list):
        weighted_layer = np.zeros_like(layer_weights[0])
        for w, weight in zip(layer_weights, weights):
            weighted_layer += w * weight
        aggregated_weights.append(weighted_layer)
    print("done_weighted_aggregation")    
    return aggregated_weights    


# In[ ]:


def fairness_loss(F, F_k, M):
    F_w = tf.reduce_mean(F_k)  # F(w)
    term = tf.reduce_sum(tf.square(F_k - F_w))
    return term / M


# In[5]:


client_names = list(clients_batched.keys())


# In[15]:



states = np.array([
    [E1, P1, 0.0, 0.0],
    [E2, P2, 0.0, 0.0],
    [E3, P3, 0.0, 0.0],
    [E4, P4, 0.0, 0.0]
])


# In[16]:


acc1 = []
loss1=[]
target_acc=95.0
rewards=[]
best_acc = 0
best_weights = None
for comm_round in range(1):
    batch_size = 10

    for step in range(60):
        local_weight_list = []
        client_updates = []
        global_weights = global_model.get_weights()
        client_models_dict = {}
#         state = states

        

        for client, (train_data, label_data) in client_data2.items():
            i=0 
            history_list=[]
#             print(i)
            local_model = SimpleMLP.build()
            local_model.set_weights(global_weights)
#             action = qm_agent.select_actions([np.array(states) for _ in range(4)])[0]
            action = qm_agent.select_actions([states] * 4)[0]
#             mu = action.numpy().flatten()[0]
        
            mu = action.numpy().flatten()[0]
            print(mu)
            if client == 'client1':
                epochs = 1
            elif client == 'client2':
                epochs = 1
            elif client == 'client3':
                epochs = 1
            else:
                epochs = 1

            history = local_model.fit(
                np.array(clients_batched[client][0]),
                np.array(clients_batched[client][1]),
                validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
                epochs=epochs,
                batch_size=32,
                verbose=2
            )
            client_models_dict[client] = local_model
            history_list.append(history)
#             print(states)
            states[i][2] = history.history['accuracy'][0]
            states[i][3] = history.history['loss'][0]
            i+=1
#             print(states)
            local_weights = local_model.get_weights()
            for j in range(len(local_weights)):
                local_weights[j] += mu * (global_weights[j] - local_weights[j])
            local_weight_list.append(local_weights)
            client_models_dict[client] = local_model
#         Calculate fairness loss
        F_k = [history.history['loss'][0] for history in history_list]
        F = np.mean(F_k)
        fairness_loss_value = fairness_loss(F, F_k, len(client_names))

        # Update weights using fairness loss
        for local_weights in local_weight_list:
            for j in range(len(local_weights)):
                local_weights[j] +=  fairness_loss_value * (global_weights[j] - local_weights[j])

#         Aggregate weights
        average_weights = avg_weights(local_weight_list)
        a,b,c,d=calculate_normalization_parameters(global_model, traini, labeli)
        similarity_metrics = calculate_similarity_metrics(global_weights, [model.get_weights() for model in client_models_dict.values()])



        # similarity_metrics = calculate_similarity_metrics(global_weights, [model.get_weights() for model in client_models_dict.values()])
        weights = update_weights_som(local_weight_list, global_model, som_weights, similarity_metrics, initial_sigma, sigma_decay_rate)
        aggregated_weights = weighted_aggregation(local_weight_list, weights)
        average_weights = avg_weights(local_weight_list)
        global_model.set_weights(average_weights)
        global_acc, global_loss = test_model(test, label, global_model, step)
#         reward = np.exp(global_acc - target_acc) - 1
        reward=global_acc
        acc1.append(global_acc)
        loss1.append(global_loss)
        rewards.append(reward)
        new_state = states
        if global_acc > best_acc:
         best_acc = global_acc
         best_weights = global_model.get_weights()
       

        
        if step >= 10:
          qm_agent.train(np.array(states), np.array(action), np.array(reward), np.array(new_state), np.array(False))



        states = new_state



global_model.set_weights(best_weights)

