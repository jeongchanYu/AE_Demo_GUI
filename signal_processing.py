from math import sqrt, log10, ceil

"""
This code is made by https://github.com/dbwls553/GenerateNoisyAudio/blob/master/signal_processing.py
"""


def calc_snr(signal, noise):
    """
    Calculate signal to noise ratio.
    If file is not noisy then SNR is about 100dB.
    :param signal: (list)
    :param noise: (list)
    :return: (value) SNR(dB)
    """
    if len(signal) != len(noise):
        raise Exception("ERROR: Signal Noise size mismatch")
    signal_power = calc_power(signal)
    noise_power = calc_power(noise)
    if noise_power == 0:
        noise_power = pow(0.1, 10)
    return 10 * log10(signal_power / noise_power)


def calc_ssnr(signal, noise, frame_size):
    """
    Calculate segmental signal noise ratio.
    If file is not noisy then SNR is about 100dB.
    :param signal: (list)
    :param noise: (list)
    :param frame_size: (int) ssnr frame size
    :return: (value) SSNR(dB)
    """
    if len(signal) != len(noise):
        raise Exception("ERROR: Signal noise size mismatch")
    number_of_frame_size = ceil(len(signal) / frame_size)
    sum = 0
    nonzero_frame_number = 0
    segmental_signal_power = [0] * number_of_frame_size
    segmental_noise_power = [0] * number_of_frame_size
    for i in range(number_of_frame_size):
        if i == number_of_frame_size - 1:
            segmental_signal_power[i] = calc_power(signal[frame_size * i:])
            segmental_noise_power[i] = calc_power(noise[frame_size * i:])
        else:
            segmental_signal_power[i] = calc_power(signal[frame_size * i:frame_size * (i + 1)])
            segmental_noise_power[i] = calc_power(noise[frame_size * i:frame_size * (i + 1)])
        if segmental_noise_power[i] == 0:
            segmental_noise_power[i] = pow(0.1, 10)
        if segmental_signal_power[i] != 0:
            nonzero_frame_number += 1
            sum += 10 * log10(segmental_signal_power[i] / segmental_noise_power[i])
    ssnr = sum / nonzero_frame_number

    return ssnr


def calc_power(input):
    """
    Calculate power of input.
    :param input: (list)
    :return: (value)
    """
    sum = 0
    for n in input:
        sum += pow(n, 2)
    return sum / len(input)


def change_power(input, power):
    """
    Change input power.
    :param input: (list)
    :param power: (value)
    :return: (list)
    """
    input_power = calc_power(input)
    ratio = sqrt(power / input_power)

    output = input.copy()
    for i in range(len(input)):
        output[i] = input[i] * ratio
    return output


def mix_noise(signal, noise, dB, snr_or_ssnr='snr', frame_size=None):
    """
    Mix noise to signal with specified dB. dB can be SNR or SSNR.
    :param signal: (list)
    :param noise: (list)
    :param dB: (value)
    :param snr_or_ssnr: 'snr' or 'ssnr'
    :param frame_size:  (int) only use in ssnr
    :return: (list) Noisy signal
    """
    if len(signal) != len(noise):
        raise Exception("ERROR: Signal Noise size mismatch")

    if snr_or_ssnr == 'snr':
        snr = calc_snr(signal, noise)
        ratio = pow(10, (snr - dB) / 20)
        noisy_signal = noise.copy()
        for i in range(len(noise)):
            noisy_signal[i] = noise[i] * ratio + signal[i]
        return noisy_signal

    elif snr_or_ssnr == 'ssnr':
        if frame_size == None:
            raise Exception("ERROR: ssnr must have frame_size")
        ssnr = calc_ssnr(signal, noise, frame_size)
        ratio = pow(10, (ssnr - dB) / 20)
        noisy_signal = noise.copy()
        for i in range(len(noise)):
            noisy_signal[i] = noise[i] * ratio + signal[i]
        return noisy_signal

    else:
        raise Exception("ERROR: You must select only in range of snr or ssnr.")
