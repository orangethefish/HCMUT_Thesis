import numpy as np
import tensorflow as tf
# from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from numpy import array, dstack
import tf_keras as keras

trainx_file = "./datatrain_40/total/trainx.txt"
trainy_file = "./datatrain_40/total/trainy.txt"
testx_file = "./datatrain_40/total/testx.txt"
testy_file = "./datatrain_40/total/testy.txt"
config_file = "./datatrain_40/total/config.txt"
enlarged_dataset_path = "./datatrain_40/total/augmentation/"


class TimeGAN(keras.Model):
    def __init__(self, seq_len, n_features, hidden_dim):
        super(TimeGAN, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Generator
        self.embedder = self._build_network(name="embedder")
        self.recovery = self._build_network(name="recovery")
        self.generator = self._build_network(name="generator")

        # Discriminator
        self.discriminator = self._build_discriminator()

        # Supervisor
        self.supervisor = self._build_network(name="supervisor")

    def _build_network(self, name):
        return keras.Sequential([
            keras.layers.GRU(units=self.hidden_dim, return_sequences=True),
            keras.layers.GRU(units=self.hidden_dim, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(units=self.n_features))
        ], name=name)

    def _build_discriminator(self):
        return keras.Sequential([
            keras.layers.GRU(units=self.hidden_dim, return_sequences=True),
            keras.layers.GRU(units=self.hidden_dim, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='sigmoid'))
        ], name="discriminator")

    def embed(self, x):
        return self.embedder(x)

    def supervise(self, h):
        return self.supervisor(h)

    def generate(self, z):
        return self.generator(z)

    def reconstruct(self, h):
        return self.recovery(h)

    def discriminate(self, x):
        return self.discriminator(x)

    @tf.function
    def train_step(self, real_data):
        # Embedding
        batch_size = tf.shape(real_data)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.seq_len, self.hidden_dim))
        hidden = self.embed(real_data)
        
        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_e, tf.GradientTape() as tape_d:
            # Generator
            fake_data = self.generate(random_noise)
            
            # Supervisor
            generated_hidden = self.embed(fake_data)
            supervised_fake = self.supervise(hidden)

            # Discriminator
            real_output = self.discriminate(real_data)
            fake_output = self.discriminate(fake_data)

            # Losses
            # Reconstruction loss
            e_loss_t0 = tf.reduce_mean((real_data - self.reconstruct(hidden))**2)
            e_loss_0 = 10 * tf.sqrt(e_loss_t0)
            
            # Supervised loss
            g_loss_s = tf.reduce_mean((hidden[:, 1:, :] - supervised_fake[:, :-1, :])**2)

            # Unsupervised loss
            g_loss_u = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
            
            # Discriminator loss
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
            d_loss = d_loss_real + d_loss_fake

            # Generator loss
            g_loss = g_loss_u + 100 * g_loss_s

            # Embedding network loss
            e_loss = e_loss_0 + 0.1 * g_loss_s

        # Compute gradients
        e_gradients = tape_e.gradient(e_loss, self.embedder.trainable_variables + self.recovery.trainable_variables)
        g_gradients = tape_g.gradient(g_loss, self.generator.trainable_variables + self.supervisor.trainable_variables)
        d_gradients = tape_d.gradient(d_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(e_gradients, self.embedder.trainable_variables + self.recovery.trainable_variables))
        self.optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables + self.supervisor.trainable_variables))
        self.optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return {"e_loss": e_loss, "g_loss": g_loss, "d_loss": d_loss}

    def generate_samples(self, n_samples):
        random_noise = tf.random.normal(shape=(n_samples, self.seq_len, self.hidden_dim))
        return self.generate(random_noise)

def prepare_data(data, seq_len, train_split=0.8):
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - seq_len + 1):
        sequences.append(scaled_data[i:i+seq_len])
    sequences = np.array(sequences)
    
    # Split into train and test sets
    train_data, test_data = train_test_split(sequences, train_size=train_split, shuffle=False)
    
    return train_data, test_data, scaler

# 2. Initialize and train the model
def train_timegan(train_data, epochs=100, batch_size=32):
    seq_len, time_step, n_features = train_data.shape[1:]
    hidden_dim = 24  # You can adjust this

    model = TimeGAN(seq_len, n_features, hidden_dim)
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001))

    train_data_reshaped = train_data.reshape(-1, seq_len, n_features)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.fit(train_data_reshaped, epochs=1, batch_size=batch_size)

    return model

# 3. Generate synthetic samples
def generate_synthetic_samples(model, n_samples):
    return model.generate_samples(n_samples)

# 4. Combine original and synthetic data
def augment_dataset(original_data, synthetic_data):
    return np.concatenate([original_data, synthetic_data], axis=0)

# 5. Validate the augmented dataset
def validate_augmented_data(original_data, augmented_data):
    # Compare basic statistics
    print("Original data shape:", original_data.shape)
    print("Augmented data shape:", augmented_data.shape)
    
    print("\nOriginal data statistics:")
    print(np.mean(original_data, axis=(0,1)))
    print(np.std(original_data, axis=(0,1)))
    
    print("\nAugmented data statistics:")
    print(np.mean(augmented_data, axis=(0,1)))
    print(np.std(augmented_data, axis=(0,1)))


def readdata(trainx_file, trainy_file, testx_file, testy_file):
    trainx_data =[]
    trainy_data =[]
    testx_data =[]
    testy_data =[]
    with open(trainx_file, 'r') as trainx, open(trainy_file, 'r') as trainy, open(testx_file, 'r') as testx, open(testy_file, 'r') as testy  :
        trainy_data = extract_y(trainy)
        testy_data = extract_y(testy)
        
        trainx_data = extract_data(trainx)
        testx_data = extract_data(testx)

    trainx_data = np.vstack(trainx_data)
    testx_data = np.vstack(testx_data)
    trainy_data = to_categorical(trainy_data)
    testy_data = to_categorical(testy_data)
    return trainx_data, trainy_data, testx_data, testy_data

def extract_data(file):
    data = []
    while True:
            try:
                x1 = []
                x2 = []
                x3 = []
                x4 = []
                x5 = []
                x6 = []
                x7 = []
                line = next(file).strip().split()
                if len(line) >= 1680:
                    for i in range(240):
                        x1.append(float(line[i]))
                        x2.append(float(line[i + 240]))
                        x3.append(float(line[i + 480]))
                        x4.append(float(line[i + 720]))
                        x5.append(float(line[i + 960]))
                        x6.append(float(line[i + 1200]))
                        x7.append(float(line[i + 1440]))
                else:
                    print("Error train data.")
                x1 = array(x1)
                x2 = array(x2)
                x3 = array(x3)
                x4 = array(x4)
                x5 = array(x5)
                x6 = array(x6)
                x7 = array(x7)
                line_dataset = dstack([x1, x2, x3, x4, x5, x6, x7]) 
                line_dataset = line_dataset.reshape(1,240,7)
                data.append(line_dataset)
            except StopIteration:
                break
    return data

def extract_y(file):
    y = []
    for line in file:
        y.append(line)
    y = array(y)
    data = np.vstack(y)
    data.reshape(1,len(y))
    return data



trainX, trainy, testX, testy = readdata(trainx_file, trainy_file, testx_file, testy_file)
trainy = np.argmax(trainy, axis=1) 

seq_len = 80
class_labels = np.unique(trainy)
class_data = {}
for label in class_labels:
    class_data[label] = trainX[trainy == label] 

gan_models = {}  # Store trained GAN models for each class

for label in class_labels:
    data_for_class, test_data, _ = prepare_data(class_data[label], seq_len)    
    gan_models[label] = train_timegan(data_for_class)

num_samples = 100 
synthetic_data_by_class = {}

for label in class_labels:
    synthetic_data_by_class[label] = generate_synthetic_samples(gan_models[label], num_samples)

augmented_data = augment_dataset(trainX, np.vstack(list(synthetic_data_by_class.values())))
validate_augmented_data(trainX, augmented_data)

