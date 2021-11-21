import os
import io_function as iof
from tkinter import filedialog
from tkinter import messagebox
from tkinter import*
import tkinter.ttk
import platform

import tensorflow as tf
import custom_function as cf
import make_dataset as md
import json
import numpy as np
import time
import datetime
import math
import wav
import model

# tf version check
tf_version = cf.get_tf_version()

# prevent GPU overflow
cf.tf_gpu_active_alloc()

# read mpl_config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)
frame_size = config["frame_size"]
shift_size = config["shift_size"]
window_type = config["window_type"]
sampling_rate = config["sampling_rate"]
batch_size = config["batch_size"]
default_float = config["default_float"]
normal_checkpoint_name = config["normal_checkpoint_name"]
mbl_checkpoint_name = config["mbl_checkpoint_name"]
mpl_checkpoint_name = config["mpl_checkpoint_name"]

# multi gpu init
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # MPL model gen
    encoder = model.Encoder()
    decoder = model.Decoder(frame_size)    # load model

@tf.function
def test_step(dist_inputs):
    result_list = []
    def step_fn(inputs):
        index, x = inputs
        x = tf.reshape(x, [x.shape[0], x.shape[1], 1])

        latent = encoder(x)
        y_pred = decoder(latent)

        if y_pred.shape[0] != 0:
            batch_split_list = tf.split(y_pred, num_or_size_splits=y_pred.shape[0], axis=0)
            for i in range(len(batch_split_list)):
                result_list.append([index[i], tf.squeeze(batch_split_list[i])])
    if tf_version[1] > 2:
        strategy.run(step_fn, args=(dist_inputs,))
    else:
        strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    return result_list


# Generate window
window = Tk()
window.title("Speech Autoencoder Demonstration Program")
window.geometry("450x350")
window.resizable(True, False)
if platform.system() == 'Windows':
    window.iconbitmap("{}/gui_icon.ico".format(iof.load_path()))


def select_source_file():
    source_entry.delete(0, END)
    source_entry.insert(0, filedialog.askopenfilename(filetypes=(('wav file', '*.wav'),)))

def select_source_dir():
    source_entry.delete(0, END)
    source_entry.insert(0, filedialog.askdirectory())

# Source path select
source_window = PanedWindow()
source_window.pack(padx=10, pady=25, fill='x')
Label(source_window, text='Source Path', width=9).pack(side=LEFT)
source_entry = Entry(source_window)
source_entry.pack(fill='x', expand=True, side=LEFT, padx=5)
source_dir_button = Button(source_window, text='Load Directory', command=select_source_dir, width=13)
source_dir_button.pack(side=RIGHT)
source_file_button = Button(source_window, text='Load File', command=select_source_file, width=8)
source_file_button.pack(side=RIGHT)


def select_output_dir():
    output_entry.delete(0, END)
    output_entry.insert(0, filedialog.askdirectory())

# Output path select
output_window = PanedWindow()
output_window.pack(padx=10, fill='x')
Label(output_window, text='Output Path', width=9).pack(side=LEFT)
output_entry = Entry(output_window)
output_entry.pack(fill='x', expand=True, side=LEFT, padx=5)
output_dir_button = Button(output_window, text='Load Directory', command=select_output_dir, width=13)
output_dir_button.pack(side=RIGHT)
output_open_button = Button(output_window, text='Open', command=lambda: os.startfile(output_entry.get()), width=8)
output_open_button.pack(side=RIGHT)


# Model select part
model_select_window = PanedWindow()
model_select_window.pack(pady=40)
model_select_frame = Frame(model_select_window)
model_select_frame.pack()

model_select_radio_value = StringVar()
normal_autoencoder_radio = Radiobutton(model_select_frame, text='Normal Autoencoder', variable=model_select_radio_value, value='Normal', indicatoron=0)
normal_autoencoder_radio.pack(side=LEFT, padx=10)
mbl_autoencoder_radio = Radiobutton(model_select_frame, text='MBL Autoencoder', variable=model_select_radio_value, value='MBL', indicatoron=0)
mbl_autoencoder_radio.pack(side=LEFT, padx=10)
mpl_autoencoder_radio = Radiobutton(model_select_frame, text='MPL Autoencoder', variable=model_select_radio_value, value='MPL', indicatoron=0)
mpl_autoencoder_radio.pack(side=LEFT, padx=10)

normal_autoencoder_radio.invoke()


# Progress bar
progressbar = tkinter.ttk.Progressbar(window, mode='determinate')
progressbar.pack(fill='x', padx=20)


def main():
    input_file_dir = source_entry.get()
    output_file_dir = output_entry.get()
    model_select = model_select_radio_value.get()

    if input_file_dir == "":
        raise Exception("ERROR: Source path is empty")
    if output_file_dir == "":
        raise Exception("ERROR: Output path is empty")

    # check input
    input_is_dir = os.path.isdir(input_file_dir)
    if input_is_dir:
        input_file_list = iof.read_dir_list(input_file_dir, extention='wav')
    else:
        input_file_list = [input_file_dir]

    if len(input_file_list) == 0:
        raise Exception("ERROR: Input file is not exist")


    # test run
    with strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        if model_select == 'Normal':
            full_path = cf.load_directory() + '/checkpoint/' + normal_checkpoint_name
        elif model_select == 'MBL':
            full_path = cf.load_directory() + '/checkpoint/' + mbl_checkpoint_name
        elif model_select == 'MPL':
            full_path = cf.load_directory() + '/checkpoint/' + mpl_checkpoint_name

        encoder.load_weights(full_path + '/encoder_data.ckpt')
        decoder.load_weights(full_path + '/decoder_data.ckpt')

        # progressbar value set
        total_length = 0
        for l in range(len(input_file_list)):
            test_source_cut_list, _, test_number_of_total_frame, front_padding, rear_padding, padded_length, sample_rate_check = md.make_dataset_for_test(input_file_list[l], input_file_list[l], frame_size, shift_size, window_type, sampling_rate)
            total_length += math.ceil(test_number_of_total_frame / batch_size)
        progressbar['maximum'] = total_length

        for l in range(len(input_file_list)):
            # make dataset
            test_source_cut_list, _, test_number_of_total_frame, front_padding, rear_padding, padded_length, sample_rate_check = md.make_dataset_for_test(input_file_list[l], input_file_list[l], frame_size, shift_size, window_type, sampling_rate)
            test_dataset = tf.data.Dataset.from_tensor_slices((list(range(test_number_of_total_frame)), test_source_cut_list)).batch(batch_size).with_options(options)
            dist_dataset_test = strategy.experimental_distribute_dataset(dataset=test_dataset)

            if input_is_dir:
                file_name = input_file_list[l].replace(input_file_dir, '').lstrip('\\/')
            else:
                file_name = os.path.basename(input_file_dir)

            output_sort = []
            output_list = np.zeros(padded_length)
            i = 0
            start = time.time()
            for inputs in dist_dataset_test:
                if not processing:
                    raise Exception("E: Processing stopped")

                print("\rTest({}) : Iter {}/{}".format(file_name, i + 1, math.ceil(test_number_of_total_frame / batch_size)), end='')

                result_package = test_step(inputs)

                for index, value in result_package:
                    output_sort.append([int(index.numpy()), value])
                i += 1
                progressbar['value'] += 1
                window.update()


            output_sort.sort()
            for index, value in output_sort:
                value = np.fft.fft(value)
                value[0] = np.complex(0.0)
                value = np.real(np.fft.ifft(value))
                output_list[shift_size * index:shift_size * index + len(value)] += value

            # save wav file
            full_path = output_file_dir + "/" + model_select + "/" + file_name
            cf.createFolder(os.path.dirname(full_path))
            wav.write_wav(output_list[front_padding:len(output_list) - rear_padding], full_path, sample_rate_check)

        messagebox.showinfo("Processing completed", "{} files generated.".format(len(input_file_list)))


def processing_button_lock(lock_state):
    if lock_state:
        source_entry['state'] = tkinter.DISABLED
        source_file_button['state'] = tkinter.DISABLED
        source_dir_button['state'] = tkinter.DISABLED
        output_entry['state'] = tkinter.DISABLED
        output_dir_button['state'] = tkinter.DISABLED
        normal_autoencoder_radio['state'] = tkinter.DISABLED
        mbl_autoencoder_radio['state'] = tkinter.DISABLED
        mpl_autoencoder_radio['state'] = tkinter.DISABLED
    else:
        source_entry['state'] = tkinter.NORMAL
        source_file_button['state'] = tkinter.NORMAL
        source_dir_button['state'] = tkinter.NORMAL
        output_entry['state'] = tkinter.NORMAL
        output_dir_button['state'] = tkinter.NORMAL
        normal_autoencoder_radio['state'] = tkinter.NORMAL
        mbl_autoencoder_radio['state'] = tkinter.NORMAL
        mpl_autoencoder_radio['state'] = tkinter.NORMAL
    window.update()

processing= False
def process_click():
    global processing
    try:
        if not processing:
            processing = True
            process_button['text'] = 'Stop'
            window.update()
            processing_button_lock(True)
            main()
    except Exception as e:
        messagebox.showerror("ERROR", e)

    processing_button_lock(False)
    progressbar['value'] = 0
    processing = False
    process_button['text'] = 'Process'
    window.update()

process_button = Button(window, text='Process', command=process_click, width=15)
process_button.pack(pady=30)
Label(window, text='Copyright 2021. CSP Lab, Kwangwoon University ').pack(side=RIGHT)

window.mainloop()