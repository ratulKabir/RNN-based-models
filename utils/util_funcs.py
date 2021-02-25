import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import ddsp
from gin_files import vq_vae
import gin
tfd = tfp.distributions

# load vq-vae model

gin.parse_config_file('saved_models/model_50k/operative_config-0.gin')
checkpoint_dir = "saved_models/model_50k/"

# VQ-VAE model
model_vq = vq_vae.QuantizingAutoencoder()
model_vq.restore(checkpoint_dir)

#codebook
latent_files_path = 'vq-50k-latent_space'
codebook = np.load('saved_latent_spaces/{}/codebook.npy'.format(latent_files_path))

# -------------------Feature converstion functions------------------

def f0_hzs_to_f0_scaled(f0_hzs):
    '''
    Here f0_hzs is converted into integer since they will be used as integers while training autoregressive model. 
    As a result of converting them into integer the scaled valeus are numerically very close to real f0_scaled values.
    However there are some numerical differences.
    '''
    return ((np.log2((tf.cast(f0_hzs, tf.dtypes.int64)/440) + 1e-7) * 12) + 69) / 127

def f0_scaleds_to_f0_hzs(f0_scaleds):
    return 2**(((f0_scaleds*127)-69)/12) * 440

def ld_scaleds_to_loudness_dbs(ld_scaleds):
    return tf.cast((ld_scaleds - 1.0) * 120, dtype=tf.dtypes.int64)

def loudness_dbs_to_ld_scaleds(loudness_dbs):
    return (loudness_dbs / 120) + 1.0

# -----------------------Helper functions for generating music---------------------

def sample_test(model_rnn, codes, chunk_len, seqlen, batch_size):
    '''
    Params: model_rnn: The thrained rnn model model (LSTM in this case)
            codes: code sequence from training data
            chunk_len: length of the chunk from code that is used as the seed of the generated sequence
            seqlen: expected length of the generated sequence
            batch_size: batch size
    function: This function uses part of a sequence from training data and generates the rest of the sequence using the rnn model
    output: gen_f0: scaled fundamental frequncy
            gen_ld: scaled loudness
            gen_z: corresponding codebook vector
    '''
    model_rnn.reset_states()
    out = codes[:, 0:int(chunk_len)]
    collected = [out] 
    seqlen *= 1000
    for t in range(int(seqlen)-chunk_len):
        out = model_rnn(out, training=False)
        gen_f0 = tfd.Categorical(logits=out[0][:,-1:]).sample()
        gen_f0 = f0_hzs_to_f0_scaled(gen_f0)
        gen_f0 = tf.cast(tf.reshape(gen_f0, (batch_size, 1, 1)), tf.dtypes.float32)
        gen_ld = tfd.Categorical(logits=out[1][:,-1:]).sample() * (-1)
        gen_ld = loudness_dbs_to_ld_scaleds(gen_ld)
        gen_ld = tf.cast(tf.reshape(gen_ld, (batch_size, 1, 1)), tf.dtypes.float32)
        gen_z = tfd.Categorical(logits=out[2][:,-1:]).sample()
        gen_z = tf.gather(codebook, gen_z)
        out = tf.concat((gen_f0, gen_ld, gen_z), axis=-1)
        collected.append(out)
        out = tf.concat(collected, axis=1)
        out = out[:, -1000:, :]
    collected = tf.concat(collected, axis=1)
    gen_f0, gen_ld, gen_z = tf.split(collected, [1,1,codebook.shape[-1]], axis=-1)
    return gen_f0, gen_ld, gen_z

def generate_rnn(model_rnn, seqlen, batch_size):
    '''
    Params: model_rnn: The thrained rnn model model (LSTM in this case)
            seqlen: expected length of the generated sequence
            batch_size: batch size
    function: This function generates the rest of the sequence using the rnn model from random starting starting point
    output: gen_f0: scaled fundamental frequncy
            gen_ld: scaled loudness
            gen_z: corresponding codebook vector
    '''
    model_rnn.reset_states()
    gen_f0 = tf.convert_to_tensor(np.random.randint(1998, size=(batch_size, 1)))
    gen_f0 = f0_hzs_to_f0_scaled(gen_f0)
    gen_f0 = tf.cast(tf.reshape(gen_f0/127, (batch_size, 1, 1)), tf.dtypes.float32)
    gen_ld = tf.convert_to_tensor(np.random.randint(121, size=(batch_size, 1))) * (-1)
    gen_ld = loudness_dbs_to_ld_scaleds(gen_ld)
    gen_ld = tf.cast(tf.reshape(gen_ld, (batch_size, 1, 1)), tf.dtypes.float32)
    starts = tf.convert_to_tensor(np.random.randint(64, size=(batch_size, 1)))
    gen_z = tf.gather(codebook, starts)
    out = tf.concat((gen_f0, gen_ld, gen_z), axis=-1)
    collected = [out] 
    for t in range(seqlen-1):
        out = model_rnn(out, training=False)
        gen_f0 = tfd.Categorical(logits=out[0][:,-1:]).sample()
        gen_f0 = f0_hzs_to_f0_scaled(gen_f0)
        gen_f0 = tf.cast(tf.reshape(gen_f0, (batch_size, 1, 1)), tf.dtypes.float32)
        gen_ld = tfd.Categorical(logits=out[1][:,-1:]).sample() * (-1)
        gen_ld = loudness_dbs_to_ld_scaleds(gen_ld)
        gen_ld = tf.cast(tf.reshape(gen_ld, (batch_size, 1, 1)), tf.dtypes.float32)
        gen_z = tfd.Categorical(logits=out[2][:,-1:]).sample()
        gen_z = tf.gather(codebook, gen_z)
        out = tf.concat((gen_f0, gen_ld, gen_z), axis=-1)
        collected.append(out)
        out = tf.concat(collected, axis=1)
    collected = tf.concat(collected, axis=1)
    gen_f0, gen_ld, gen_z = tf.split(collected, [1,1,codebook.shape[-1]], axis=-1)
    return gen_f0, gen_ld, gen_z

def generate_audio(model_rnn, seqlen, batch_size):
    '''
    Params: model_rnn: The thrained rnn model model (LSTM in this case)
            seqlen: expected length of the generated sequence
            codes: code sequence from training data
            batch_size: batch size
    Function: gathers the required outputs from the rnn model, converts them to necessary form
              and feeds them to the DDSP decoder to generate Audio
    Output: outputs generated audio of seqlen*4 seconds
    '''
    fs, ld, z = generate_rnn(model_rnn, seqlen, batch_size=32)
    fh = f0_scaleds_to_f0_hzs(fs)

    back_to_decoder = {}
    back_to_decoder['f0_hz'] = fh
    back_to_decoder['f0_scaled'] = fs
    back_to_decoder['ld_scaled'] = ld
    back_to_decoder['z'] = z
    
    audio_gen = model_vq.decode(back_to_decoder, training=False)
    return audio_gen

def generate_sample_test(model_rnn, code, chunk_len=1, seqlen=1, batch_size=32):
    '''
    Params: model_rnn: The thrained rnn model model (LSTM in this case)
            codes: code sequence from training data
            chunk_len: length of the chunk from code that is used as the seed of the generated sequence
            seqlen: expected length of the generated sequence
            batch_size: batch size
    Function: gathers the required outputs from the rnn model, converts them to necessary form
              and feeds them to the DDSP decoder to generate Audio
    Output: outputs generated audio of seqlen*4 seconds
    '''
    fs, ld, z = sample_test(model_rnn, code, chunk_len, seqlen, batch_size)
    fh = f0_scaleds_to_f0_hzs(fs)
    
    assert fs.shape[1] == seqlen*1000
    
    audio_generated = []
    end_seq = 1000
    for n_seq in range (seqlen): 
        start_seq = n_seq * 1000
        end_seq = start_seq + end_seq
        back_to_decoder = {}
        back_to_decoder['f0_hz'] = fh[:, start_seq:end_seq, :]
        back_to_decoder['f0_scaled'] = fs[:, start_seq:end_seq, :]
        back_to_decoder['ld_scaled'] = ld[:, start_seq:end_seq, :]
        back_to_decoder['z'] = z[:, start_seq:end_seq, :]     # shape = (batch_size, seq_len, ...)
        audio_gen = model_vq.decode(back_to_decoder, training=False) # shape = (batch_size, 64000)
        audio_generated.append(audio_gen)
    return tf.concat(audio_generated, axis=-1)