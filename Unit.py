import numpy as np

def dB2watt(dB):
    watt = 10 ** (dB / 10)
    return watt

def watt2dB(watt):
    dB = 10 * np.log10(watt)
    return dB

def watt2dBm(watt):
    dBm = 10 * np.log10(watt/0.001)
    return dBm

def dBm2watt(dBm):
    watt = 10 ** ((dBm-30) / 10)
    return watt
