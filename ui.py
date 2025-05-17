import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from classification import plot_confusion_matrices

class BCIInterface:
    def __init__(self, master, results, label_names):
        self.master = master
        self.results = results
        self.label_names = label_names
        
        # Set up the UI
        master.title("Motor Imagery BCI Interface")
        master.geometry("800x600")
        master.configure(bg='white')
        
        # Create frames
        self.top_frame = tk.Frame(master, bg='white')
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.bottom_frame = tk.Frame(master, bg='white')
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create confusion matrix plot
        self.fig = plot_confusion_matrices(results, label_names)
        self.canvas = FigureCanvasTkAgg(self.fig, self.top_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create arrow display frame
        self.arrow_frame = tk.Frame(master, bg='white', height=300)
        self.arrow_frame.pack(side=tk.TOP, fill=tk.X, pady=20)
        
        # Make arrow frame not shrink
        self.arrow_frame.pack_propagate(False)
        
        # Set up the arrows with fixed positions in a cross pattern
        self.arrow_size = 36
        self.arrows = {}
        
        # Arrow labels with distinct Unicode characters
        # SWAPPED positions 2 and 3 (up/down arrows)
        arrow_chars = {
            0: "←",  # Left arrow (Left hand)
            1: "→",  # Right arrow (Right hand)
            2: "↑",  # Up arrow (Tongue) - SWAPPED
            3: "↓"   # Down arrow (Foot) - SWAPPED
        }
        
        # SWAPPED labels for foot and tongue
        arrow_labels = {
            0: "LEFT HAND",
            1: "RIGHT HAND",
            2: "TONGUE",  # SWAPPED
            3: "FOOT"     # SWAPPED
        }
        
        # Create a container for each arrow with label
        for idx, (char, label) in enumerate(zip(arrow_chars.values(), arrow_labels.values())):
            # Create a frame for each arrow
            arrow_container = tk.Frame(self.arrow_frame, bg='white')
            
            # Position arrows in a cross pattern
            if idx == 0:  # Left
                arrow_container.place(relx=0.2, rely=0.5, anchor="center")
            elif idx == 1:  # Right
                arrow_container.place(relx=0.8, rely=0.5, anchor="center")
            elif idx == 2:  # Up
                arrow_container.place(relx=0.5, rely=0.2, anchor="center")
            else:  # Down
                arrow_container.place(relx=0.5, rely=0.8, anchor="center")
            
            # Create arrow symbol
            arrow_label = tk.Label(arrow_container, text=char, 
                                 font=("Arial", self.arrow_size), 
                                 bg='white', fg='black',
                                 width=2, height=1)
            arrow_label.pack()
            
            # Create text label
            text_label = tk.Label(arrow_container, text=label, 
                                font=("Arial", 10), 
                                bg='white', fg='black')
            text_label.pack()
            
            # Store reference to the arrow label
            self.arrows[idx] = arrow_label
        
        # Create classifier selection
        self.classifier_var = tk.StringVar(value="SVM")
        
        # Create a control panel frame
        control_panel = tk.Frame(self.bottom_frame, bg='white')
        control_panel.pack(fill=tk.X, pady=10)
        
        # Add classifier selection
        classifier_frame = tk.LabelFrame(control_panel, text="Classifier", bg='white')
        classifier_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.svm_radio = tk.Radiobutton(classifier_frame, text="SVM", variable=self.classifier_var, 
                                       value="SVM", bg='white', command=self.update_display)
        self.svm_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.lda_radio = tk.Radiobutton(classifier_frame, text="LDA", variable=self.classifier_var, 
                                       value="LDA", bg='white', command=self.update_display)
        self.lda_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Navigation panel
        nav_frame = tk.LabelFrame(control_panel, text="Navigation", bg='white')
        nav_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.prev_button = tk.Button(nav_frame, text="◀ Previous Trial", command=self.prev_trial,
                                   bg='#e0e0e0', width=15)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.next_button = tk.Button(nav_frame, text="Next Trial ▶", command=self.next_trial,
                                   bg='#e0e0e0', width=15)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Info panel
        self.info_frame = tk.LabelFrame(control_panel, text="Trial Information", bg='white')
        self.info_frame.pack(side=tk.LEFT, padx=20, pady=5, fill=tk.X, expand=True)
        
        self.info_label = tk.Label(self.info_frame, text="", bg='white', justify=tk.LEFT)
        self.info_label.pack(padx=10, pady=5)
        
        # Current trial index
        self.current_trial = 0
        self.update_display()
    
    def update_display(self):
        # Reset all arrows to default
        for arrow in self.arrows.values():
            arrow.config(fg="black", font=("Arial", self.arrow_size), bg='white')
        
        # Get the current prediction
        if self.classifier_var.get() == "SVM":
            pred = self.results['svm_preds'][self.current_trial]
            accuracy = self.results['svm_acc']
        else:  # LDA
            pred = self.results['lda_preds'][self.current_trial]
            accuracy = self.results['lda_acc']
        
        # Highlight the predicted arrow
        self.arrows[pred].config(fg="white", font=("Arial", self.arrow_size, "bold"), bg='red')
        
        # Show actual vs predicted
        true_label = self.results['y_test'][self.current_trial]
        
        # Update info text
        total_trials = len(self.results['y_test'])
        info_text = f"Trial: {self.current_trial + 1} of {total_trials}\n"
        info_text += f"True class: {self.label_names[true_label]}\n"
        info_text += f"Predicted: {self.label_names[pred]}\n"
        info_text += f"Classifier: {self.classifier_var.get()} (Accuracy: {accuracy:.2%})"
        
        self.info_label.config(text=info_text)
    
    def next_trial(self):
        if self.current_trial < len(self.results['y_test']) - 1:
            self.current_trial += 1
            self.update_display()
    
    def prev_trial(self):
        if self.current_trial > 0:
            self.current_trial -= 1
            self.update_display()