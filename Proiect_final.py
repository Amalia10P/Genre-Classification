import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog, font
import customtkinter
import pygame

import pandas as pd
import os
import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from datetime import datetime
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

### citim fisierul csv cu atributele
audio_dataset_path='/dataset/Data/'
metadata=pd.read_csv('./dataset/Data/features_30_sec_edit.csv')

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

# Luam toate fisierele din folder
file_names = os.listdir('./dataset/Data/genres_original')

extracted_features = []
for index_num,row in tqdm(metadata.iterrows()):
    if index_num < len(file_names):
        file_name = os.path.join('./dataset/Data/genres_original/',file_names[index_num])
        final_class_labels = row["label"]
        data=features_extractor(file_name)
        extracted_features.append([data,final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','label'])
# print(extracted_features_df.head(10))

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['label'].tolist())

# Categorizam numeric label-urile din dataset, pentru a eficientiza modelul.
labelencoder=LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# Impartim setul de date in date de test si date de antrenare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    current_sound = None
    model_a = Sequential()
    root = None
    search_input = None
    tree_model = None
    def __init__(self):
        super().__init__()

        # configure window
        self.title("")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Genre Genie", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event1, text = "Choose an audio")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event2, text = "Exit the program")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="Zoom:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        # --------------------------------------------
        # self.entry = customtkinter.CTkEntry(self, placeholder_text="Poate cautam aici ?")
        self.entry = customtkinter.CTkLabel(self, text="", width=250, height=80, anchor="w") # text="Accuracy:\n",
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="w")

        # self.main_button_1 = customtkinter.CTkButton(master=self, text = "Buton fara folos", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        # self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Commands")
        self.tabview.tab("Commands").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs

        # self.optionmenu_1 = customtkinter.CTkOptionMenu(self.tabview.tab("Commands"), dynamic_resizing=False,
        #                                                 values=["Value 1", "Value 2", "Value Long Long Long"])
        # self.optionmenu_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        # self.combobox_1 = customtkinter.CTkComboBox(self.tabview.tab("Commands"),
        #                                             values=["Value 1", "Value 2", "Value Long....."])
        # self.combobox_1.grid(row=1, column=0, padx=20, pady=(10, 10))
        # self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Commands"), text="Search for an audio file",
        #                                                    command=self.open_input_dialog_event)
        # self.string_input_button.grid(row=1, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Commands"), text="Antrenare A1", fg_color="#8B869E",
                                                           command=self.antrenare_a)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Commands"), text="Testare A1", fg_color="#8B869E",
                                                           command=self.testare_a)
        self.string_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Commands"), text="Antrenare A2",
                                                           command=self.antrenare_d)
        self.string_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Commands"), text="Testare A2",
                                                           command=self.testare_d)
        self.string_input_button.grid(row=5, column=0, padx=20, pady=(10, 10))

        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        self.seg_button_1 = customtkinter.CTkSegmentedButton(self.slider_progressbar_frame)
        self.seg_button_1.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_1.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        self.progressbar_2 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_2.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.slider_1 = customtkinter.CTkSlider(self.slider_progressbar_frame, from_=0, to=1, command=self.adjust_volume)
        self.slider_1.bind("<Motion>", self.adjust_volume)
        # self.slider_1 = tk.Scale(self.slider_progressbar_frame, from_=0, to=100, command=self.adjust_volume)
        self.slider_1.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        # --- set default values ---
        self.appearance_mode_optionemenu.set("Light")
        self.scaling_optionemenu.set("100%")
        # self.optionmenu_1.set("Mai Vedem")
        # self.combobox_1.set("Si aici mai vedem")
        self.slider_1.configure(command=self.progressbar_2.set)
        self.progressbar_1.configure(mode="indeterminnate")
        # self.progressbar_1.start()
        self.textbox.insert("0.0", "Write your own thoughts\n\n" + "In my opinion this song lacks a powerful guitar, which is why the program did not recognise it to be a rock song.\n\n" * 3)
        self.seg_button_1.configure(values=["Adjust the volume of the audio"]) #, "Value 2", "Value 3"])
        self.seg_button_1.set("Adjust the volume of the audio")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Search for an audio file:", title="Audio Search")
        self.search_input = dialog.get_input()
        print("Input:", self.search_input)

        if self.search_input:
            search_directory = "./dataset/Data/genres_original/" # os.getcwd() # "/path/to/your/directory"  # Replace with the actual directory path
            print("--------------ASTA E: ", search_directory)
            # search_directory = "C:\\Users\\borde\OneDrive\\Desktop"
            # matching_files = []
            matching_files = [filename for filename in os.listdir(search_directory) if self.search_input == filename] 
            # for filename in os.listdir(search_directory):
            #     print("--------------ASTA E: ", filename)
            #     if search_input == filename:
            #         matching_files.append(filename)
            
            if matching_files:
                print("Found matching files:", matching_files)
                pygame.mixer.init()
                if self.current_sound:
                    self.current_sound.stop()
                selected_filename = matching_files[0] # For maybe selecting the correct file, if needed.
                pygame.mixer.music.load(os.path.join(search_directory, selected_filename)) # load the file in a sound obj
                pygame.mixer.music.play()
                self.progressbar_1.start()
                self.current_sound = pygame.mixer.music
                self.slider_1.set(50)
            else:
                print("No matching files found.")
        else:
            print("No input provided.")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event1(self):
        print("SB_B1 click")
        init_dir = os.getcwd()
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.filename = filedialog.askopenfilename(initialdir=init_dir, title="Select an audio file",
                                                   filetypes=(("Wave files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")))
        if self.root.filename:
            # pygame.mixer.init()
            # if self.current_sound:
            #     self.current_sound.stop()
            # self.current_sound = pygame.mixer.music.load(root.filename)
            # self.current_sound.play()
            # pygame.mixer.music.set_volume(0.5)
            # self.slider_1.set(50)

            pygame.mixer.init()
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.root.filename)
            self.progressbar_1.start()
            pygame.mixer.music.play()
            self.current_sound = pygame.mixer.music
            # self.testare_a(self.current_sound)
            pygame.mixer.music.set_volume(0.5)
            self.slider_1.set(50)

    
    def sidebar_button_event2(self):
        print("SB_B2 click")
        exit()

    def adjust_volume(self, event):
        if self.current_sound:
            volume_percentage = self.slider_1.get()
            # print("Adjusting volume to:", volume_percentage)
            volume_level = float(volume_percentage)
            self.current_sound.set_volume(volume_level)

    def update_volume(self, volume_level):
        pass
    
    def antrenare_a(self):
        # Antrenarea retelei neurale
        
        ### first layer
        self.model_a.add(Dense(150, input_shape = (40,)))
        self.model_a.add(Activation('relu'))
        self.model_a.add(Dropout(0.5))
        ### second layer
        self.model_a.add(Dense(200))
        self.model_a.add(Activation('relu'))
        self.model_a.add(Dropout(0.5))
        ### third layer
        self.model_a.add(Dense(150))
        self.model_a.add(Activation('relu'))
        self.model_a.add(Dropout(0.5))

        ### final layer
        num_labels = 9
        self.model_a.add(Dense(num_labels))
        self.model_a.add(Activation('softmax'))

        # Setam componentele necesare pentru antrenare
        self.model_a.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        num_epochs = 400
        num_batch_size = 32

        checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                                    verbose=1, save_best_only=True)
        # calculam timpul necesar antrenarii
        start = datetime.now()

        # antrenarea retelei neurale
        self.model_a.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
        
        duration = datetime.now() - start
        print("Training completed in time: ", duration)

    def testare_a(self):
        print(self.current_sound)
        test_accuracy=self.model_a.evaluate(X_test,y_test,verbose=0)
        print(test_accuracy[1])

        filename = os.path.join("./dataset/Data/genres_original/", self.root.filename) #f"./dataset/Data/genres_original/{self.current_sound}"
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predicted_label=self.model_a.predict(mfccs_scaled_features)
        classes_x=np.argmax(predicted_label,axis=1)
        prediction_class = labelencoder.inverse_transform(classes_x)
        print(prediction_class)

        # Calculate and display accuracy
        test_accuracy = self.model_a.evaluate(X_test, y_test, verbose=0)
        accuracy_value = test_accuracy[1] * 100  # Accuracy value in percentage
        self.entry.configure(text=f"Accuracy: {accuracy_value:.2f}%", font = customtkinter.CTkFont(family="Helvetica", size=30))
        self.entry.configure(text=f"Accuracy: {accuracy_value:.2f}%\nGenre: {prediction_class[0]}", font = customtkinter.CTkFont(family="Helvetica", size=30))

    def antrenare_d(self):
        print("Antrenare model varianta 2")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # pentru a  nu mai primi avertisment ca nu gaseste un folder
        # incercand sa foloseasca GPU mai mult pentru eficientizare
        # Implementare algoritm ID3, arbori de decizie

        # Inițializează modelul de arbori de decizie
        # posibil sa se faca overfitting daca am ales ca arborele sa fie prea mare
        self.tree_model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=18)
        # cu 5 avem 0,28
        # cu 6 avem 0,35
        # cu 7 avem 0,36
        # cu 10 avem 0,40
        # cu 15 avem 0,40
        # cu 16 avem 0,37
        # cu 17 avem 0,43
        # cu 18 avem 0.43
        # cu 20 avem 0,42

        # Antrenează modelul folosind datele de antrenare
        self.tree_model.fit(X_train, y_train)



    def testare_d(self):
        # Realizează predicții pe datele de test folosind modelul de arbore de decizie
        y_pred_tree = self.tree_model.predict(X_test)

        # Calculează acuratețea predicțiilor
        test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred_tree.argmax(axis=1))
        print(f"Acuratețea modelului cu max_depth=18 : {test_accuracy:.2f}")

        filename = os.path.join("./dataset/Data/genres_original/", self.root.filename) #f"./dataset/Data/genres_original/{self.current_sound}"
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predicted_label=self.tree_model.predict(mfccs_scaled_features)
        classes_x=np.argmax(predicted_label,axis=1)
        prediction_class = labelencoder.inverse_transform(classes_x)
        print(prediction_class)

        # Calculate and display accuracy
        accuracy_value = test_accuracy * 100  # Accuracy value in percentage
        self.entry.configure(text=f"Accuracy: {accuracy_value:.2f}%\nGenre: {prediction_class[0]}", font = customtkinter.CTkFont(family="Helvetica", size=30))

        # Calculează matricea de confuzie
        y_pred = self.tree_model.predict(X_test)
        confusion = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

        # Afișează matricea de confuzie
        print("Matricea de confuzie:\n", confusion)


if __name__ == "__main__":
    app = App()
    app.mainloop()