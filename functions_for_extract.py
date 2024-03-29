def get_sound_index(sound):
    return sound_list.index(sound) if sound in sound_list else None   

def extract_features(file_name,label):

    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Extract additional features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_processed = np.mean(mel.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    contrast_processed = np.mean(contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    # Concatenate all features into a single feature vector
    feature_vector = np.concatenate(([],np.array([label]),mfccs_processed, chroma_processed, mel_processed, contrast_processed, tonnetz_processed))

    # Create a pandas dataframe with columns named correctly
    features = pd.DataFrame([feature_vector], columns=['label']+['MFCCs_{}'.format(i) for i in range(mfccs_processed.shape[0])] + 
                                                    ['Chroma_{}'.format(i) for i in range(chroma_processed.shape[0])] + 
                                                    ['Mel_{}'.format(i) for i in range(mel_processed.shape[0])] + 
                                                    ['Contrast_{}'.format(i) for i in range(contrast_processed.shape[0])] + 
                                                    ['Tonnetz_{}'.format(i) for i in range(tonnetz_processed.shape[0])])
    return features
    

    
def extract_features1(file_name):
  
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Extract additional features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_processed = np.mean(mel.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    contrast_processed = np.mean(contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    # Concatenate all features into a single feature vector
    feature_vector = np.concatenate(([],mfccs_processed, chroma_processed, mel_processed, contrast_processed, tonnetz_processed))

    # Create a pandas dataframe with columns named correctly
    features = pd.DataFrame([feature_vector], columns=['MFCCs_{}'.format(i) for i in range(mfccs_processed.shape[0])] + 
                                                    ['Chroma_{}'.format(i) for i in range(chroma_processed.shape[0])] + 
                                                    ['Mel_{}'.format(i) for i in range(mel_processed.shape[0])] + 
                                                    ['Contrast_{}'.format(i) for i in range(contrast_processed.shape[0])] + 
                                                    ['Tonnetz_{}'.format(i) for i in range(tonnetz_processed.shape[0])])
    return features
    
# Feature Engineering

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_std = np.std(mfccs.T, axis=0)
    return mfccs_std

def extract_chroma_stft(file_path):
    y, sr = librosa.load(file_path)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_std = np.std(chroma_stft.T, axis=0)
    return chroma_stft_std

def extract_melspectrogram(file_path):
    y, sr = librosa.load(file_path)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    melspectrogram_std = np.std(melspectrogram.T, axis=0)
    return melspectrogram_std

def extract_spectral_contrast(file_path):
    y, sr = librosa.load(file_path)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_std = np.std(spectral_contrast.T, axis=0)
    return spectral_contrast_std

def extract_tonnetz(file_path):
    y, sr = librosa.load(file_path)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_std = np.std(tonnetz.T, axis=0)
    return tonnetz_std

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(10, 10))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    



# Preprocess the data by standardizing and scaling
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Perform PCA for dimensionality reduction (optional)
def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Apply KMeans clustering algorithm to identify the clusters
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans

# Calculate the distance of each sample from the closest cluster center
def calculate_distance(data, kmeans):
    distances = kmeans.transform(data)
    closest_cluster = np.argmin(distances, axis=1)
    closest_distance = np.min(distances, axis=1)
    return closest_distance

# Plot the histogram of closest distances
def plot_histogram(closest_distance):
    plt.hist(closest_distance, bins=50)
    plt.show()

# Main function
def main(df, n_clusters=3, n_components=2):
    preprocessed_data = preprocess_data(df.values)
    reduced_data = perform_pca(preprocessed_data, n_components=n_components)
    kmeans = apply_kmeans(reduced_data, n_clusters=n_clusters)
    closest_distance = calculate_distance(reduced_data, kmeans)
#     plot_histogram(closest_distance)
    df['probability_outlier'] = closest_distance
    return df


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess_for_predict(file_path): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.reshape(spectrogram, (1,spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2]))
    return spectrogram