import music21 as m21
import numpy as np
import os
from music21 import *

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

    :param song (m21 stream):
    :param acceptable_durations (list): List of acceptable duration in quarter length
    :return (bool):
    """
    
    return all(note.duration.quarterLength in acceptable_durations for note in song.flat.notesAndRests)

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

    :param song: A music21 stream object representing the piece to transpose.
    :return: A music21 stream object of the transposed song.
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






def preprocess(dataset_path):

    songs = load_songs_in_kern(dataset_path)
    
    for song in songs:
        if not valid_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose song to Cmaj/Amin
        song = transpose_to_Cmaj_Amin(song)
    
   

if __name__ == '__main__':
    songs = load_songs_in_kern(DATASET_PATH)
    
    song_test = songs[4]
    sont_test_transposed = transpose_to_Cmaj_Amin(song_test)


    song_test.show()
    sont_test_transposed.show()
