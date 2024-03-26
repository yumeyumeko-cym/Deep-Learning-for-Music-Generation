import numpy as np
import torch
import music21 as m21
from model import *
from preprocessing import *
import torch.nn.functional as F

# Sample with temperature
def sample_with_temperature(logits, temperature=1.0):
    probabilities = torch.softmax(logits / temperature, dim=0)
    sampled_indices = torch.multinomial(probabilities, 1)
    return sampled_indices.squeeze().cpu().numpy()

# load mappings
with open(MAPPING_FILE_PATH, 'r') as fp:
    mappings = json.load(fp)
# Inverse mappings
inv_mappings = {v: k for k, v in mappings.items()}

# map int to note
def int_to_note(integers):
    notes = [inv_mappings[i] for i in integers]
    return notes

# generate music sequence
def generate_music_sequence(model, seed, sequence_length, temperature=1.0):
    seed_integers = [mappings[note] if note in mappings else 0 for note in seed.split()]  # Convert seed to integers
    generated_sequence = seed_integers.copy()

    model.eval().to(device)
    with torch.no_grad():
        for _ in range(sequence_length):
            print(generated_sequence)
            input_sequence = generated_sequence[-SEQUENCE_LENGTH:]
            input_sequence += [0] * (SEQUENCE_LENGTH - len(input_sequence))  # Pad sequence
            input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(device)

            # One-hot encode the input tensor
            input_tensor_one_hot = F.one_hot(input_tensor, num_classes=len(mappings)).float().to(device)

            logits = model(input_tensor_one_hot)
            # print(logits)
            next_note = sample_with_temperature(logits, temperature)
            next_note = next_note.item()
            generated_sequence.append(next_note)


    generated_notes = [inv_mappings.get(i, '_') for i in generated_sequence]  # Convert back to notes
    return generated_notes

# create a function to save a melody to a MIDI file
def save_melody_to_midi(notes, file_name="generated_melody.mid"):
    stream = m21.stream.Stream()
    for note_val in notes:
        if note_val == 'r':
            m21_note = m21.note.Rest(quarterLength=0.25)
        elif note_val in ['_', '/']:
            continue  # Skip continuation and section symbols for MIDI conversion
        else:
            m21_note = m21.note.Note(int(note_val), quarterLength=0.25) if note_val.isdigit() else None
        if m21_note:
            stream.append(m21_note)
    stream.write('midi', fp=file_name)

# load the model
lstm_model = myLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUTPUT_UNIT)
lstm_model.load_state_dict(torch.load('trained_lstm_model.pth', map_location=device))

seed = "60 _ 60 _ 62 _ 64 _ | 60 _ 60 _ 60 _ 62 _ | 60 _ 62 _ 64 _ 65 _ | 67 _ _ 65 _60 _ 60 _ 62 _ 64 _ | 60 _ 60 _ 60 _ 62 _ | 67 _ 62 _ 64 _ 65 _ | 67 _ _ 65 _62 _ 62 _ 64 _ 62 _ | 60 _ 67 _ 65 _ | 62 _ 62 _ 64 _ 62 _ | 60 _ 59 _ 59 _ 57 _"
sequence_length = 100
notes = generate_music_sequence(lstm_model, seed, sequence_length, temperature=0.8)
save_melody_to_midi(notes, "generated_melody2.mid")
