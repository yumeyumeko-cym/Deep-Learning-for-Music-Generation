import music21 as m21
import numpy as np
import os
from music21 import *
import json
import torch
from torch.utils.data import Dataset, DataLoader
from MusicDataset import MusicDataset


env = environment.Environment()
env['musicxmlPath'] = r'D:/MuseScore 4/bin/MuseScore4.exe'
env['musescoreDirectPNGPath'] = r'D:/MuseScore 4/bin/MuseScore4.exe'


DATASET_PATH = "./test"
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

SAVE_DIR = "./saved_preprocessed"
SEQUENCE_LENGTH = 64
TRAINING_DATASET_FILE_PATH = "./training_dataset.txt"
MAPPING_FILE_PATH = "./mapping.json"




def load_songs_in_kern(dataset_path):
    """
    Loads dataset using m21.

    :param dataset_path (str): Path to dataset
    :return (List) <class 'music21.stream.base.Score'>
    """
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # Consider only MIDI files
            if file.lower().endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def valid_durations(song, acceptable_durations):
    """
    Check if all notes and rests have acceptable durations.

    :param song (m21 stream)
    :param acceptable_durations (list): List of acceptable duration in quarter length
    :return (bool):
    """
    
    return all(note.duration.quarterLength in acceptable_durations for note in song.flatten().notesAndRests)

"""
Music Transpose
In the context of deep learning for music generation, the ability to transpose music is valuable for several reasons, reflecting both the enhancement of the dataset used for training deep learning models and the adaptation of generated music to specific musical contexts or preferences. Here are key reasons why a transpose function is important:

1. Data Augmentation
Diversity in Training Data: Transposing existing pieces into different keys increases the diversity of the training dataset without needing to collect new compositions. This diversity helps in training more robust and versatile models capable of understanding and generating music in various keys.
Improved Generalization: By presenting the model with the same piece in multiple keys, it can learn to recognize and generate the underlying patterns of music theory that are invariant to key changes. This improves the model's ability to generalize from its training data to new compositions.
2. Key Normalization
Consistent Key for Model Training: Training models on data that is normalized to a specific key (often C major/A minor for simplicity) can simplify the learning process. It reduces the complexity of the input space, potentially making it easier for the model to learn the structure and progression of musical pieces.
Easier Pattern Recognition: Models might find it easier to recognize patterns and structures in music when the variability due to different keys is removed. This can lead to more efficient learning and a stronger focus on other musical aspects like rhythm, dynamics, and melody.
3. Adaptation to Instrumentation or Vocal Ranges
Matching Musical Range: Once music is generated, it might need to be adapted to fit the specific range of an instrument or the vocal range of a singer. Transposing the generated piece to a key that suits the intended performers can make the music more practical and enjoyable to play or sing.
"""

def transpose_to_Cmaj_Amin(song):
    """
    Transposes a song to C major or A minor.

    :param song (m21 stream)
    :return transposed song (m21 stream)
    """
    # Analyze the key of the song
    original_key = song.analyze('key')
    
    # Determine the target key based on the mode of the original key
    target_key = key.Key('C') if original_key.mode == 'major' else key.Key('A', 'minor')
    
    # Calculate the interval between the original key's tonic and the target key's tonic
    transposition_interval = interval.Interval(original_key.tonic, target_key.tonic)
    
    # Transpose the song by the calculated interval
    transposed_song = song.transpose(transposition_interval)
    
    return transposed_song

def encode(song, time_step=0.25):
    """
    p=60, d=1.0 -> 60, _, _, _

    interger for notes, 'r' for representing a rest, and '_' for representing notes/rests that are carried over into a new time step.

    :param song (m21 stream)
    :param time_step (float): Duration of each time step in quarter length
    :return encoded song (str)
    """            

    encoded_sequence = []

    for elem in song.flatten().notesAndRests:
        if isinstance(elem, note.Note):
            symbol = elem.pitch.midi

        elif isinstance(elem, note.Rest):
            symbol = "r"
        

        steps = int(elem.duration.quarterLength / time_step)
        
        for step_index in range(steps):
            if step_index == 0:
                # Encode the symbol directly for the first step
                encoded_sequence.append(symbol)
            else:
                # Use '_' to indicate continuation for subsequent steps
                encoded_sequence.append('_')

    # Convert the list of encoded elements into a single string for output
    encoded_string = " ".join(str(item) for item in encoded_sequence)


    return encoded_string




def preprocess(dataset_path):

    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    for song in songs:
        if not valid_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose song to Cmaj/Amin
        song = transpose_to_Cmaj_Amin(song)

        # encode song
        song_encoded = encode(song)

        # Ensure the 'preprocessed' directory exists
        preprocessed_dir = os.path.join(SAVE_DIR, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)

        # Handle potential issues with special characters in file names
        safe_title = song.metadata.title.replace(os.sep, "_").replace("\x84", "ae")
        save_path = os.path.join(preprocessed_dir, safe_title + ".txt")  # Adding .txt extension

        # Save songs to text file
        with open(save_path, "w", encoding='utf-8') as fp:  # Ensure encoding is set for special characters
            fp.write(song_encoded)

    return preprocessed_dir



def collating(dataset_path, file_dataset_path, sequence_length):
    """
    Generates a file collating all the encoded songs and adding new piece delimiters.
    
    :param dataset_path: Path to folder containing the encoded songs.
    :param file_dataset_path: Path to file for saving songs in a single file.
    :param sequence_length (int): number of delimiters used to separate songs
    :return <class 'str'> encoded songs with delimiters
    """
    new_song_delimiter = "/ " * sequence_length
    songs_list = []
    
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as f:
                song = f.read()
                songs_list.append(song + " " + new_song_delimiter)
    
    # Join all songs and delimiters, then trim the last delimiter
    songs = ''.join(songs_list).rstrip('/ ')
    
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    
    return songs
    
def load(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents

def map_json(songs, file_mapping_path):
    """
    Generate a mapping from each unique symbol in the encoded songs to an integer.
    
    :param songs: String containing all encoded songs with delimiters.
    :param file_mapping_path: Path to file for saving the mapping as JSON.
    """
    # Split the songs string into a list of symbols
    #songs = load(file_dataset_path)
    #print(songs)
    #print('\n')
    symbols = songs.split(" ")
    #print(symbols)
    # Remove empty symbols caused by consecutive spaces
    symbols = [symbol for symbol in symbols if symbol != ""]
    
    # Use a set to find unique symbols
    unique_symbols = set(symbols)
    # print(unique_symbols)
    # for symbol in enumerate(unique_symbols):
    #     print(symbol)
    # Create a mapping from symbols to integers
    symbol_to_int = {symbol: i for i, symbol in enumerate(unique_symbols)}
    
    # Save the mapping to a JSON file
    with open(file_mapping_path, "w") as f:
        json.dump(symbol_to_int, f, indent=4)

    return symbol_to_int

def mapped_songs(file_mapping_path, songs):
    """
    Generate mapped songs for training

    :param file_mapping_path: Path to file containing the mapping as JSON.
    :param songs: String containing all encoded songs with delimiters.
    :return: List of mapped songs
    """
    mapped_songs = []
    # Load the mapping from JSON
    with open(file_mapping_path, "r") as f:
        mapping = json.load(f)

    # Split the songs string into a list of symbols
    songs = songs.split(" ")
    for symbol in songs:
        # Map each symbol to an integer
        mapped_songs.append(mapping[symbol])

    return mapped_songs


if __name__ == '__main__':
    songs = load_songs_in_kern(DATASET_PATH)
    
    song_test = songs[0]
    sont_test_transposed = transpose_to_Cmaj_Amin(song_test)


    #song_test.show()
    #sont_test_transposed.show()

    preprocessed_dir = preprocess(DATASET_PATH)
    collating(preprocessed_dir, TRAINING_DATASET_FILE_PATH, SEQUENCE_LENGTH)
    songs = load(TRAINING_DATASET_FILE_PATH)
    map_json(songs, MAPPING_FILE_PATH)
    mapped_songs = mapped_songs(MAPPING_FILE_PATH, songs)
    #print(mapped_songs[:20])
    dataset = MusicDataset(mapped_songs, SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # # Iterate through the DataLoader to get batches
    for i, (input, target) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Input Shape: {input.shape}")  # Should be [batch_size, sequence_length, vocab_size]
        print(f"Target Shape: {target.shape}")  # Should be [batch_size]
        print(f"Sample target Output: {target[0:30]}")  # Should be an integer
        # Optionally, inspect the actual data (in a more interpretable format if necessary)
        if i == 1:  # Just as an example, break after the second batch
            break