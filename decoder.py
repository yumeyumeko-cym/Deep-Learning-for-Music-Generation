import numpy as np
import torch
import music21 as m21
import torch.nn.functional as F
import json
from model_bilstm import myBiLSTM, hidden_size, num_layers
from preprocessing import MAPPING_FILE_PATH, SEQUENCE_LENGTH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sample with temperature
def sample_with_temperature(logits, sequence_length, max_sequence_length, temperature=1.0):
    if temperature is None:
        # dynamic temperature adjustment
        temperature = max(0.5, 1 - (sequence_length / max_sequence_length))
    # Softmax over vocab dimension to get a categorical distribution.
    probabilities = torch.softmax(logits / temperature, dim=-1)
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
def generate_music_sequence(model, seed, max_sequence_length, temperature=1.0):
    seed_integers = [mappings[note] if note in mappings else 0 for note in seed.split()]  # Convert seed to integers
    generated_sequence = seed_integers.copy()

    model.eval().to(device)
    with torch.no_grad():
        # Change the loop to iterate until the generated sequence reaches max_sequence_length
        while len(generated_sequence) < max_sequence_length:
            current_sequence_length = len(generated_sequence)  # Now accurately reflects the number of notes generated so far
            #print(generated_sequence)
            input_sequence = generated_sequence[-SEQUENCE_LENGTH:]
            input_sequence += [0] * (SEQUENCE_LENGTH - len(input_sequence))  # Pad sequence
            input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(device)

            # One-hot encode to match training input shape.
            input_tensor_one_hot = F.one_hot(input_tensor, num_classes=len(mappings)).float().to(device)

            logits = model(input_tensor_one_hot)
            next_note = sample_with_temperature(logits, current_sequence_length, max_sequence_length, temperature)
            next_note = next_note.item()
            generated_sequence.append(next_note)

    generated_notes = [inv_mappings.get(i, '_') for i in generated_sequence]  # Convert back to notes
    return generated_notes

# create a function to save a melody to a MIDI file
def save_melody_to_midi(notes, file_name="generated_melody.mid", time_step=0.25):
    stream = m21.stream.Stream()
    current_symbol = None
    current_duration = 0.0

    def flush_event():
        nonlocal current_symbol, current_duration
        if current_symbol is None:
            return
        if current_symbol == 'r':
            m21_note = m21.note.Rest(quarterLength=current_duration)
        elif current_symbol.isdigit():
            m21_note = m21.note.Note(int(current_symbol), quarterLength=current_duration)
        else:
            m21_note = None
        if m21_note:
            stream.append(m21_note)
        current_symbol = None
        current_duration = 0.0

    for note_val in notes:
        if note_val in ['/', '|']:
            flush_event()
            continue
        if note_val == '_':
            if current_symbol is not None:
                # Extend the previous note/rest by one time step.
                current_duration += time_step
            continue
        flush_event()
        current_symbol = note_val
        current_duration = time_step
    flush_event()
    stream.write('midi', fp=file_name)

vocab_size = len(mappings)
lstm_model = myBiLSTM(vocab_size, hidden_size, num_layers, vocab_size)

#lstm_model.load_state_dict(torch.load('trained_lstm_model.pth', map_location=device))
lstm_model.load_state_dict(torch.load('trained_bi_lstm_model.pth', map_location=device))

seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
seed2 = "64 _ 69 _ _ _ 71 _ 72 _ _ 71 69 _ 76 _ _ _ _ _ 71 _ _ _ 74 _ 72 _ _ "
sequence_length = 50
notes = generate_music_sequence(lstm_model, seed2, sequence_length, temperature=0.8)
save_melody_to_midi(notes, "generated_melody6.mid")
