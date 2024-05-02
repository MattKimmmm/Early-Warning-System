import matplotlib.pyplot as plt

# Define your data as a string
data = """
Timestep 1: Accuracy: 0.6119, Precision: 0.6816, Recall: 0.4164, F1: 0.5015, AUC: 0.6212
Timestep 2: Accuracy: 0.6256, Precision: 0.6350, Recall: 0.5837, F1: 0.5931, AUC: 0.6643
Timestep 3: Accuracy: 0.6387, Precision: 0.6429, Recall: 0.6491, F1: 0.6359, AUC: 0.6686
Timestep 4: Accuracy: 0.6384, Precision: 0.6414, Recall: 0.6521, F1: 0.6369, AUC: 0.6846
Timestep 5: Accuracy: 0.6679, Precision: 0.6658, Recall: 0.6962, F1: 0.6702, AUC: 0.6896
Timestep 6: Accuracy: 0.6548, Precision: 0.6534, Recall: 0.6968, F1: 0.6636, AUC: 0.6844
Timestep 7: Accuracy: 0.6482, Precision: 0.6426, Recall: 0.6958, F1: 0.6591, AUC: 0.6918
Timestep 8: Accuracy: 0.6524, Precision: 0.6401, Recall: 0.7261, F1: 0.6718, AUC: 0.6984
Timestep 9: Accuracy: 0.6524, Precision: 0.6391, Recall: 0.7350, F1: 0.6759, AUC: 0.7067
Timestep 10: Accuracy: 0.6586, Precision: 0.6427, Recall: 0.7427, F1: 0.6817, AUC: 0.7010
Timestep 11: Accuracy: 0.6524, Precision: 0.6319, Recall: 0.7355, F1: 0.6726, AUC: 0.7078
Timestep 12: Accuracy: 0.6485, Precision: 0.6273, Recall: 0.7313, F1: 0.6688, AUC: 0.7109
Timestep 13: Accuracy: 0.6423, Precision: 0.6202, Recall: 0.7313, F1: 0.6645, AUC: 0.7017
Timestep 14: Accuracy: 0.6423, Precision: 0.6202, Recall: 0.7313, F1: 0.6645, AUC: 0.7060
Timestep 15: Accuracy: 0.6423, Precision: 0.6202, Recall: 0.7313, F1: 0.6645, AUC: 0.7046
Timestep 16: Accuracy: 0.6423, Precision: 0.6202, Recall: 0.7313, F1: 0.6645, AUC: 0.7091
Timestep 17: Accuracy: 0.6464, Precision: 0.6238, Recall: 0.7313, F1: 0.6669, AUC: 0.7097
Timestep 18: Accuracy: 0.6485, Precision: 0.6250, Recall: 0.7361, F1: 0.6693, AUC: 0.7140
Timestep 19: Accuracy: 0.6485, Precision: 0.6250, Recall: 0.7361, F1: 0.6693, AUC: 0.7153
Timestep 20: Accuracy: 0.6464, Precision: 0.6224, Recall: 0.7361, F1: 0.6681, AUC: 0.7172
Timestep 21: Accuracy: 0.6464, Precision: 0.6224, Recall: 0.7361, F1: 0.6681, AUC: 0.7189
Timestep 22: Accuracy: 0.6464, Precision: 0.6224, Recall: 0.7361, F1: 0.6681, AUC: 0.7211
Timestep 23: Accuracy: 0.6464, Precision: 0.6224, Recall: 0.7361, F1: 0.6681, AUC: 0.7180
Timestep 24: Accuracy: 0.6464, Precision: 0.6224, Recall: 0.7361, F1: 0.6681, AUC: 0.7197
Timestep 25: Accuracy: 0.6464, Precision: 0.6224, Recall: 0.7361, F1: 0.6681, AUC: 0.7142
Timestep 26: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7361, F1: 0.6699, AUC: 0.7154
Timestep 27: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7361, F1: 0.6699, AUC: 0.7158
Timestep 28: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7158
Timestep 29: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7179
Timestep 30: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7163
Timestep 31: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7196
Timestep 32: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7196
Timestep 33: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7229
Timestep 34: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7218
Timestep 35: Accuracy: 0.6443, Precision: 0.6221, Recall: 0.7277, F1: 0.6643, AUC: 0.7215
Timestep 36: Accuracy: 0.6464, Precision: 0.6239, Recall: 0.7277, F1: 0.6656, AUC: 0.7234
Timestep 37: Accuracy: 0.6464, Precision: 0.6239, Recall: 0.7277, F1: 0.6656, AUC: 0.7244
Timestep 38: Accuracy: 0.6464, Precision: 0.6239, Recall: 0.7277, F1: 0.6656, AUC: 0.7229
Timestep 39: Accuracy: 0.6464, Precision: 0.6239, Recall: 0.7277, F1: 0.6656, AUC: 0.7229
Timestep 40: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7255
Timestep 41: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7255
Timestep 42: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7239
Timestep 43: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7267
Timestep 44: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7267
Timestep 45: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7256
Timestep 46: Accuracy: 0.6485, Precision: 0.6254, Recall: 0.7319, F1: 0.6681, AUC: 0.7261
Timestep 47: Accuracy: 0.6443, Precision: 0.6225, Recall: 0.7277, F1: 0.6644, AUC: 0.7243
Timestep 48: Accuracy: 0.6443, Precision: 0.6225, Recall: 0.7277, F1: 0.6644, AUC: 0.7264
Timestep 49: Accuracy: 0.6443, Precision: 0.6225, Recall: 0.7277, F1: 0.6644, AUC: 0.7264
Timestep 50: Accuracy: 0.6423, Precision: 0.6204, Recall: 0.7277, F1: 0.6633, AUC: 0.7275
Timestep 51: Accuracy: 0.6423, Precision: 0.6204, Recall: 0.7277, F1: 0.6633, AUC: 0.7244
Timestep 52: Accuracy: 0.6423, Precision: 0.6204, Recall: 0.7277, F1: 0.6633, AUC: 0.7249
Timestep 53: Accuracy: 0.6423, Precision: 0.6204, Recall: 0.7277, F1: 0.6633, AUC: 0.7274
Timestep 54: Accuracy: 0.6423, Precision: 0.6204, Recall: 0.7277, F1: 0.6633, AUC: 0.7263
Timestep 55: Accuracy: 0.6402, Precision: 0.6164, Recall: 0.7277, F1: 0.6611, AUC: 0.7247
Timestep 56: Accuracy: 0.6402, Precision: 0.6164, Recall: 0.7277, F1: 0.6611, AUC: 0.7237
Timestep 57: Accuracy: 0.6402, Precision: 0.6164, Recall: 0.7277, F1: 0.6611, AUC: 0.7230
Timestep 58: Accuracy: 0.6402, Precision: 0.6164, Recall: 0.7277, F1: 0.6611, AUC: 0.7232
Timestep 59: Accuracy: 0.6402, Precision: 0.6164, Recall: 0.7277, F1: 0.6611, AUC: 0.7226
Timestep 60: Accuracy: 0.6423, Precision: 0.6190, Recall: 0.7277, F1: 0.6628, AUC: 0.7221
Timestep 61: Accuracy: 0.6443, Precision: 0.6205, Recall: 0.7319, F1: 0.6655, AUC: 0.7226
Timestep 62: Accuracy: 0.6443, Precision: 0.6205, Recall: 0.7319, F1: 0.6655, AUC: 0.7219
Timestep 63: Accuracy: 0.6443, Precision: 0.6205, Recall: 0.7319, F1: 0.6655, AUC: 0.7225
Timestep 64: Accuracy: 0.6464, Precision: 0.6219, Recall: 0.7367, F1: 0.6682, AUC: 0.7235
Timestep 65: Accuracy: 0.6443, Precision: 0.6206, Recall: 0.7367, F1: 0.6673, AUC: 0.7235
Timestep 66: Accuracy: 0.6443, Precision: 0.6206, Recall: 0.7367, F1: 0.6673, AUC: 0.7225
Timestep 67: Accuracy: 0.6423, Precision: 0.6201, Recall: 0.7339, F1: 0.6656, AUC: 0.7251
Timestep 68: Accuracy: 0.6423, Precision: 0.6201, Recall: 0.7339, F1: 0.6656, AUC: 0.7235
Timestep 69: Accuracy: 0.6423, Precision: 0.6201, Recall: 0.7339, F1: 0.6656, AUC: 0.7241
Timestep 70: Accuracy: 0.6402, Precision: 0.6183, Recall: 0.7339, F1: 0.6645, AUC: 0.7252
Timestep 71: Accuracy: 0.6423, Precision: 0.6207, Recall: 0.7394, F1: 0.6681, AUC: 0.7241
Timestep 72: Accuracy: 0.6423, Precision: 0.6207, Recall: 0.7394, F1: 0.6681, AUC: 0.7241
"""

# Initialize lists to store timesteps and accuracy values
timesteps = []
accuracies = []

# Parse the string to extract numbers
for line in data.split('\n'):
    if "Accuracy" in line:
        parts = line.split(':')
        # Extract the timestep correctly by splitting the string "Timestep x" and taking the number
        timestep = int(parts[0].split()[1])
        # Extract the accuracy by splitting on ',' after 'Accuracy' and stripping spaces
        accuracy = float(parts[6].split(',')[0].strip())
        timesteps.append(timestep)
        accuracies.append(accuracy)

# Create a plot
print("hi")
plt.figure(figsize=(10, 5))
plt.plot(timesteps, accuracies, linestyle='-', color='b')
plt.title('AUC over Time')
plt.xlabel('Timestep')
plt.ylabel('AUC')
plt.grid(True)
plt.savefig('AUC_over_time.png') 
