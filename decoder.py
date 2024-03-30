import numpy as np
import torch
import music21 as m21
from model_lstm import *
from model_bilstm import *
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
            #print(generated_sequence)
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
    current_duration = 1.0
    for note_val in notes:
        if note_val == '_':
            # Add a quarter for a each _ character
            current_duration = current_duration / 2
        else:
            if note_val == 'r':
                m21_note = m21.note.Rest(quarterLength=current_duration)
            elif note_val in ['/']:
                continue
            else:
                m21_note = m21.note.Note(int(note_val), quarterLength=current_duration) if note_val.isdigit() else None
            
            if m21_note:
                stream.append(m21_note)
            current_duration = 1.0
    stream.write('midi', fp=file_name)

# load the model
lstm_model = myLSTM(input_size, hidden_size, num_layers, output_unit)
lstm_model.load_state_dict(torch.load('trained_lstm_model.pth', map_location=device))

seed = "60 _ 60 _ 69 _ _ 65 63 _ 64 _ 64 _ _"
sequence_length = 200
notes = generate_music_sequence(lstm_model, seed, sequence_length, temperature=1.2)
save_melody_to_midi(notes, "generated_melody4.mid")
