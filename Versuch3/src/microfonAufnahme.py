# sample file from moodle

import pyaudio
import struct
import math

INITIAL_TAP_THRESHOLD = 0.010
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)
CHANNELS = 2
RATE = 44100

INPUT_BLOCK_TIME = 0.05
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
# if we get this many noisy blocks in a row, increase thethreshold
OVERSENSITIVE = 15.0/INPUT_BLOCK_TIME
# if we get this many quite blocks in a row, decrease thethreshold
UNDERSENSITIVE = 120.0/INPUT_BLOCK_TIME
# if th enoise was longer than this many blocks, it’s not a ’tap’
MAX_TAP_BLOCKS = 0.15/INPUT_BLOCK_TIME

def get_rms( block ):
# RMS amplitude is defined as the square root of the# mean over time of the square of the amplitude.# So we need to convert this string of bytes into# a string of 16-bit samples...# we will get one short out for each# two xhars in the string.
    count = len(block)/2
    format ="%dh"%(count)
    shorts = struct.unpack(format, block)
    # iterate over the block
    sum_squares = 0.0
