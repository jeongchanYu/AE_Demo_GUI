import os
import io_function as iof
import signal_processing as sp

input_file_dir = 'C:/Users/유정찬/Desktop/test/clean_test'
noise_file_dir = 'C:/Users/유정찬/Desktop/test/noise'
output_file_dir = 'C:/Users/유정찬/Desktop/test/noisy_test'
snr_or_ssnr = 'ssnr'
target_dB = 10
frame_size = 1600


# check input, noise directory
input_is_dir = os.path.isdir(input_file_dir)
noise_is_dir = os.path.isdir(noise_file_dir)
if input_is_dir:
    input_file_list = iof.read_dir_list(input_file_dir, extention='wav')
else:
    input_file_list = [input_file_dir]
if noise_is_dir:
    noise_file_list = iof.read_dir_list(noise_file_dir, extention='wav')
else:
    noise_file_list = [noise_file_dir]

if len(input_file_list)==0:
    raise Exception("ERROR: Input file is not exist")
if len(noise_file_list)==0:
    raise Exception("ERROR: Noise file is not exist")


# make noise set
old_sample_rate = 0
temp_sample_rate = 0
noise_signal_train = []
print('Processing noise file.')
i = 0
for wav_file in noise_file_list:
    i += 1
    print('Processing({}) {}/{} ...'.format(wav_file, i, len(noise_file_list)))
    temp_noise_signal, temp_sample_rate = iof.read_wav(wav_file)
    if old_sample_rate != 0 and old_sample_rate != temp_sample_rate:
        raise Exception("ERROR: Different sample rate is exist.")
    else:
        old_sample_rate = temp_sample_rate

    temp_noise_signal = sp.change_power(temp_noise_signal, 1)
    noise_signal_train.extend(temp_noise_signal)


# mix noise
old_sample_rate = 0
temp_sample_rate = 0
noise_start_point = 0
print('Generate noisy file.')
i = 0
for wav_file in input_file_list:
    i += 1
    print('Processing({}) {}/{} ...'.format(wav_file, i, len(input_file_list)))
    temp_input_signal, temp_sample_rate = iof.read_wav(wav_file)
    if old_sample_rate != 0 and old_sample_rate != temp_sample_rate:
        raise Exception("ERROR: Different sample rate is exist.")
    else:
        old_sample_rate = temp_sample_rate

    temp_noise_signal = []
    left_size = len(temp_input_signal)
    while left_size > 0:
        available_length = len(noise_signal_train)-noise_start_point
        available_length = min(available_length, left_size)
        temp_noise_signal.extend(noise_signal_train[noise_start_point:noise_start_point+available_length])
        left_size -= available_length
        noise_start_point += available_length
        if noise_start_point == len(noise_signal_train):
            noise_start_point = 0

    mixed_signal = sp.mix_noise(temp_input_signal, temp_noise_signal, target_dB, snr_or_ssnr, frame_size)

    temp_output_file_dir = wav_file.replace(input_file_dir, output_file_dir)
    iof.create_folder(os.path.dirname(temp_output_file_dir))
    iof.write_wav(mixed_signal, temp_output_file_dir, temp_sample_rate)
