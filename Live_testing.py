import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import time

training = True

CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
PIXEL_SIZE = 10


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs):
        input = np.array(inputs, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_hidden, input)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Network
input_nodes = 784
hidden_nodes = 523
output_nodes = 10
learning_rate = 0.1
epochs = 10
fill_value = 250

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
scorecard = []

question_train = input("do u want to load weights y/n ?       ")

if question_train == "y":
    weights_input_hidden = np.loadtxt('weights_input_hidden.txt')
    weights_hidden_output = np.loadtxt('weights_hidden_output.txt')

    n.weights_input_hidden = weights_input_hidden
    n.weights_hidden_output = weights_hidden_output
    training = False

#Train network
if training:
    print("training started")
    trainigs_data_file = open("mnist_train_american.csv", "r")
    trainigs_data_list = trainigs_data_file.readlines()
    trainigs_data_file.close()

    training_start = time.time()
    for epoch in range(epochs):
        for record in trainigs_data_list:
            all_values = record.split(",")
            inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        print("training passed")
        pass


test_data_file = open("mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()


for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label, "right label")

    inputs = (np.asarray(all_values[1:]) / 255 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label, "neural network answered")

    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

def save_weights():
    np.savetxt('weights_input_hidden.txt', n.weights_input_hidden)
    np.savetxt('weights_hidden_output.txt', n.weights_hidden_output)


question_save = input("do u want to save the weights y/n ?       ")
if question_save == "y":
    save_weights()

def softmax(x):
    exp_values = np.exp(x - np.max(x))  # vermeide Überlauf durch Subtraktion des Maximums
    return exp_values / np.sum(exp_values)
def draw(event):
    x = event.x // PIXEL_SIZE
    y = event.y // PIXEL_SIZE
    canvas.create_rectangle(x * PIXEL_SIZE, y * PIXEL_SIZE, (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE, fill="black")
    pixel_values[y][x] = fill_value

def clear_canvas():
    canvas.delete("all")
    for row in range(len(pixel_values)):
        for col in range(len(pixel_values[row])):
            pixel_values[row][col] = 0

def process_image(pixel_values):
    scale_inputs = (np.array(pixel_values) / 255.0 * 0.99) + 0.01
    return scale_inputs


def show_image():
    scale_inputs = process_image(pixel_values)
    outputs = n.query(scale_inputs.flatten())
    label = np.argmax(outputs)
    print(scale_inputs)
    print(label, "Neural network prediction")

    outputs = n.query(scale_inputs.flatten())
    probabilities = softmax(outputs).flatten()
    for i, prob in enumerate(probabilities):
        print(f'Zahl {i}: {prob}')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Zeige das Bild in ax1 an
    ax1.imshow(np.array(pixel_values), cmap="gray_r")
    ax1.axis('off')

    # Plotte ein Balkendiagramm der Wahrscheinlichkeiten in ax2
    classes = list(range(len(probabilities)))
    ax2.bar(classes, probabilities, align='center', alpha=0.5)

    # Füge die Zahlen unter die Balken hinzu
    for i, prob in enumerate(probabilities):
        ax2.annotate(f'{i}: {prob:.2f}', xy=(i, prob), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom')

    ax2.set_xlabel('Zahlen')
    ax2.set_ylabel('Wahrscheinlichkeit')
    ax2.set_title('Wahrscheinlichkeiten für jede Zahl')
    ax2.set_xticks(classes)

    # Zeige die gesamte Figur an
    plt.tight_layout()
    plt.show()


def output_num(num):
    return num


def open_second_window():
    second_window = tk.Toplevel(root)
    second_window.title("Second Window")

    entry_label = tk.Label(second_window, text="Geben Sie eine Zahl ein:")
    entry_label.pack()

    entry = tk.Entry(second_window)
    entry.pack()

    save_button = tk.Button(second_window, text="Speichern", command=lambda: output_num(entry.get()))
    save_button.pack()

    return entry


def train_single():
    entry = open_second_window()  # Get the entry widget
    num = entry.get()  # Get the value from the entry widget

    # Check if num is empty
    if num:
        inputs = (np.asfarray(pixel_values) / 255 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(num)] = 0.99
        n.train(inputs, targets)
    else:
        # Handle case when num is empty
        print("Error: No value entered")


# Erstelle ein 2D-Array für die Pixelwerte
pixel_values = [[0] * (CANVAS_WIDTH // PIXEL_SIZE) for _ in range(CANVAS_HEIGHT // PIXEL_SIZE)]

root = tk.Tk()
root.title("Mnist-Like Zeichenfenster")

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
canvas.pack()

canvas.bind("<B1-Motion>", draw)

clear_button = tk.Button(root, text="Löschen", command=clear_canvas)
clear_button.pack(side="left")

show_button = tk.Button(root, text="Bild anzeigen", command=show_image)
show_button.pack(side="left")

save_button = tk.Button(root, text="Gewichte speichern", command=save_weights)
save_button.pack(side="right")

train_button = tk.Button(root, text="Trainieren", command=train_single)
train_button.pack(side="right")

root.mainloop()