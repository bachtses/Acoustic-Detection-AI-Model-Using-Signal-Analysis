import pyaudio
import wave
import os
import sys
import numpy as np

# Audio settings
RAW_CHANNELS = 6       # ReSpeaker always outputs 6 channels  
TARGET_CHANNELS = 1    # Save merged output
rate = 16000
chunk = 1024
record_seconds = 2
file_count = 0
GAIN = 3   # +9.5 dB 

recording_class = 'drone'

output_folder = os.path.join('recordings', recording_class)
os.makedirs(output_folder, exist_ok=True)


def list_input_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            devices.append((i, info.get('name')))
    p.terminate()
    return devices


def find_input_device_index(name_keyword):
    p = pyaudio.PyAudio()
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0 and name_keyword.lower() in info.get('name').lower():
            device_index = i
            break
    p.terminate()
    return device_index


# Print all available input devices
print("Available input devices:")
for i, name in list_input_devices():
    print(f"  ID {i}: {name}")

device_index = find_input_device_index("respeaker")
if device_index is None:
    print("Desired input device not found. Exiting.")
    sys.exit(1)

# Open multichannel input stream
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=RAW_CHANNELS,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk)

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        raw_frames = []

        # read 2 seconds of audio
        for _ in range(int(rate / chunk * record_seconds)):
            data = stream.read(chunk, exception_on_overflow=False)
            raw_frames.append(data)

        # convert to numpy
        data_bytes = b''.join(raw_frames)
        audio = np.frombuffer(data_bytes, dtype=np.int16)

        # raw mics: channels 1–4
        mic1 = audio[1::RAW_CHANNELS]
        mic2 = audio[2::RAW_CHANNELS]
        mic3 = audio[3::RAW_CHANNELS]
        mic4 = audio[4::RAW_CHANNELS]

        # merge
        merged = (mic1.astype(np.int32) +
                  mic2.astype(np.int32) +
                  mic3.astype(np.int32) +
                  mic4.astype(np.int32)) // 4

        # apply gain
        merged_float = merged.astype(np.float32) * GAIN
        merged_float = np.clip(merged_float, -32768, 32767)
        merged = merged_float.astype(np.int16)

        # save WAV
        output_filename = os.path.join(output_folder, f"{recording_class}_{file_count}.wav")
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(merged.tobytes())
        wf.close()

        print(f"\033[92mSaved:\033[00m {output_filename}")
        file_count += 1

except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
