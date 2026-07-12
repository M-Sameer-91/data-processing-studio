import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import io

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------
BG_COLOR = "#EEF1F8"
HEADER_TOP = "#2563EB"
HEADER_BOTTOM = "#7C3AED"
CARD_BG = "#FFFFFF"
BTN_PRIMARY = "#2563EB"
BTN_PRIMARY_HOVER = "#1D4ED8"
BTN_SUCCESS = "#06D6A0"
BTN_SUCCESS_HOVER = "#04B589"
BTN_DANGER = "#EF4444"
BTN_DANGER_HOVER = "#DC2626"
BTN_WARNING = "#F59E0B"
BTN_WARNING_HOVER = "#D97706"
TEXT_DARK = "#1F2937"
TEXT_MUTED = "#6B7280"
LOG_BG = "#111827"
LOG_FG = "#93C5FD"


class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Data Processing & Visualization Studio")
        self.root.geometry("1200x850")
        self.root.configure(bg=BG_COLOR)

        self.df = None          # main dataframe
        self.file_path = None

        # Encoding helpers
        self.label_encoders = {}   # col -> fitted LabelEncoder

        # Feature scaling / split state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalers = {}          # col -> fitted scaler (for reference/log only)

        self.setup_styles()
        self.build_header()
        self.build_tabs()
        self.build_log_panel()

    # ---------------------------------------------------------------
    # Styles
    # ---------------------------------------------------------------
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[16, 9],
                         background="#DCE0F0", foreground=TEXT_DARK)
        style.map("TNotebook.Tab",
                  background=[("selected", HEADER_TOP)],
                  foreground=[("selected", "white")])
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("Treeview", font=("Consolas", 9), rowheight=22)

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    def build_header(self):
        header = tk.Canvas(self.root, height=70, highlightthickness=0)
        header.pack(fill="x")

        def draw_gradient(event=None):
            header.delete("g")
            width = header.winfo_width()
            steps = 50
            r1, g1, b1 = self.root.winfo_rgb(HEADER_TOP)
            r2, g2, b2 = self.root.winfo_rgb(HEADER_BOTTOM)
            for i in range(steps):
                x0 = int(width * i / steps)
                x1 = int(width * (i + 1) / steps)
                ratio = i / steps
                r = int(r1 + (r2 - r1) * ratio) >> 8
                g = int(g1 + (g2 - g1) * ratio) >> 8
                b = int(b1 + (b2 - b1) * ratio) >> 8
                color = f"#{r:02x}{g:02x}{b:02x}"
                header.create_rectangle(x0, 0, x1, 70, fill=color, outline=color, tags="g")
            header.create_text(width / 2, 25, text="📊 Data Processing & Visualization Studio",
                                font=("Segoe UI", 16, "bold"), fill="white", tags="g")
            header.create_text(width / 2, 48,
                                text="Upload → Clean → Encode → Scale → Statistics → Visualization",
                                font=("Segoe UI", 9), fill="#DBEAFE", tags="g")

        header.bind("<Configure>", draw_gradient)

    # ---------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------
    def build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(10, 5))

        self.tab_upload = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_process = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_encode = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_scale = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_stats = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_viz = tk.Frame(self.notebook, bg=BG_COLOR)

        self.notebook.add(self.tab_upload, text="📂 Upload & Preview")
        self.notebook.add(self.tab_process, text="🧹 Process Data")
        self.notebook.add(self.tab_encode, text="🏷️ Encoding")
        self.notebook.add(self.tab_scale, text="⚖️ Feature Scaling")
        self.notebook.add(self.tab_stats, text="📈 Statistics")
        self.notebook.add(self.tab_viz, text="📊 Visualization")

        self.build_upload_tab()
        self.build_process_tab()
        self.build_encoding_tab()
        self.build_scaling_tab()
        self.build_stats_tab()
        self.build_viz_tab()

    # =================================================================
    # TAB 1: UPLOAD & PREVIEW
    # =================================================================
    def build_upload_tab(self):
        top = tk.Frame(self.tab_upload, bg=BG_COLOR, pady=10)
        top.pack(fill="x", padx=15)

        tk.Label(top, text="File Path:", font=("Segoe UI", 10, "bold"),
                  bg=BG_COLOR, fg=TEXT_DARK).grid(row=0, column=0, sticky="w")
        self.path_entry = tk.Entry(top, font=("Segoe UI", 10), width=60, relief="flat",
                                    highlightthickness=1, highlightbackground=HEADER_TOP)
        self.path_entry.grid(row=0, column=1, padx=8, ipady=4)

        browse_btn = self.make_button(top, "📁 Browse", self.browse_file, BTN_PRIMARY, BTN_PRIMARY_HOVER)
        browse_btn.grid(row=0, column=2, padx=6)

        load_btn = self.make_button(top, "🚀 Load Dataset", self.load_file, BTN_SUCCESS, BTN_SUCCESS_HOVER)
        load_btn.grid(row=0, column=3, padx=6)

        tk.Label(top, text="Custom Headers (comma-separated, optional):", font=("Segoe UI", 9, "italic"),
                  bg=BG_COLOR, fg=TEXT_MUTED).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.headers_entry = tk.Entry(top, font=("Segoe UI", 9), width=90, relief="flat",
                                       highlightthickness=1, highlightbackground=HEADER_TOP)
        self.headers_entry.grid(row=2, column=0, columnspan=4, sticky="we", pady=4, ipady=3)

        # Preview area
        preview_frame = tk.LabelFrame(self.tab_upload, text="Data Preview (head)", bg=CARD_BG,
                                       font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        preview_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.preview_tree = self.make_treeview(preview_frame)

        # Info box
        info_frame = tk.LabelFrame(self.tab_upload, text="Dataset Info (shape, dtypes, non-null counts)",
                                    bg=CARD_BG, font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        info_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.info_text = tk.Text(info_frame, height=10, font=("Consolas", 9), bg="#F9FAFB")
        self.info_text.pack(fill="both", expand=True, padx=6, pady=6)

    def browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("All Supported", "*.csv *.data *.json *.xlsx *.xls"),
                       ("CSV files", "*.csv *.data"),
                       ("JSON files", "*.json"),
                       ("Excel files", "*.xlsx *.xls")]
        )
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

    def load_file(self):
        path = self.path_entry.get().strip()
        if not path:
            messagebox.showerror("Error", "Please browse and select a file first.")
            return
        try:
            headers_raw = self.headers_entry.get().strip()
            headers = [h.strip() for h in headers_raw.split(",")] if headers_raw else None

            if path.lower().endswith((".csv", ".data")):
                self.df = pd.read_csv(path, names=headers) if headers else pd.read_csv(path)
            elif path.lower().endswith(".json"):
                self.df = pd.read_json(path)
            elif path.lower().endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(path)
            else:
                messagebox.showerror("Error", "Unsupported file format.")
                return

            self.file_path = path
            # reset downstream state on fresh load
            self.label_encoders = {}
            self.X_train = self.X_test = self.y_train = self.y_test = None
            self.scalers = {}

            self.log(f"Loaded dataset: {path}  | Shape: {self.df.shape}")
            self.refresh_preview()
            self.refresh_info()
            self.refresh_column_dropdowns()
            messagebox.showinfo("Success", f"Dataset loaded successfully!\nShape: {self.df.shape}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def refresh_preview(self):
        self.populate_treeview(self.preview_tree, self.df.head(10))

    def refresh_info(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"Shape: {self.df.shape}\n\n{info_str}")

    # =================================================================
    # TAB 2: PROCESS DATA (Missing Values)
    # =================================================================
    def build_process_tab(self):
        top = tk.Frame(self.tab_process, bg=BG_COLOR, pady=10)
        top.pack(fill="x", padx=15)

        btn1 = self.make_button(top, "🔄 Replace '?' with NaN", self.replace_question_marks,
                                 BTN_PRIMARY, BTN_PRIMARY_HOVER)
        btn1.grid(row=0, column=0, padx=5)

        btn2 = self.make_button(top, "🔍 Show Missing Value Summary", self.show_missing_summary,
                                 BTN_PRIMARY, BTN_PRIMARY_HOVER)
        btn2.grid(row=0, column=1, padx=5)

        # Missing values table
        missing_frame = tk.LabelFrame(self.tab_process, text="Columns with Missing Values",
                                       bg=CARD_BG, font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        missing_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.missing_tree = self.make_treeview(missing_frame, columns=("Column", "Missing Count", "Dtype"))

        # Imputation controls
        impute_frame = tk.LabelFrame(self.tab_process, text="Handle Missing Data", bg=CARD_BG,
                                      font=("Segoe UI", 10, "bold"), fg=TEXT_DARK, padx=10, pady=10)
        impute_frame.pack(fill="x", padx=15, pady=8)

        tk.Label(impute_frame, text="Column:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5)
        self.impute_col_var = tk.StringVar()
        self.impute_col_menu = ttk.Combobox(impute_frame, textvariable=self.impute_col_var,
                                             state="readonly", width=25)
        self.impute_col_menu.grid(row=0, column=1, padx=5)

        tk.Label(impute_frame, text="Strategy:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5)
        self.strategy_var = tk.StringVar(value="mean")
        strategy_menu = ttk.Combobox(impute_frame, textvariable=self.strategy_var, state="readonly",
                                      values=["mean", "median", "most_frequent", "drop rows with NaN in this column"],
                                      width=32)
        strategy_menu.grid(row=0, column=3, padx=5)

        apply_btn = self.make_button(impute_frame, "✅ Apply", self.apply_imputation, BTN_SUCCESS, BTN_SUCCESS_HOVER)
        apply_btn.grid(row=0, column=4, padx=10)

        # Data type conversion
        dtype_frame = tk.LabelFrame(self.tab_process, text="Fix Data Format (Convert Column Type)",
                                     bg=CARD_BG, font=("Segoe UI", 10, "bold"), fg=TEXT_DARK, padx=10, pady=10)
        dtype_frame.pack(fill="x", padx=15, pady=8)

        tk.Label(dtype_frame, text="Column:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5)
        self.dtype_col_var = tk.StringVar()
        self.dtype_col_menu = ttk.Combobox(dtype_frame, textvariable=self.dtype_col_var, state="readonly", width=25)
        self.dtype_col_menu.grid(row=0, column=1, padx=5)

        tk.Label(dtype_frame, text="Convert to:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5)
        self.dtype_target_var = tk.StringVar(value="float")
        dtype_menu = ttk.Combobox(dtype_frame, textvariable=self.dtype_target_var, state="readonly",
                                   values=["float", "int", "str"], width=15)
        dtype_menu.grid(row=0, column=3, padx=5)

        convert_btn = self.make_button(dtype_frame, "🔧 Convert", self.convert_dtype, BTN_SUCCESS, BTN_SUCCESS_HOVER)
        convert_btn.grid(row=0, column=4, padx=10)

        # Save cleaned CSV
        save_frame = tk.Frame(self.tab_process, bg=BG_COLOR, pady=10)
        save_frame.pack(fill="x", padx=15)
        save_btn = self.make_button(save_frame, "💾 Save Cleaned Dataset as CSV", self.save_cleaned_csv,
                                     BTN_DANGER, BTN_DANGER_HOVER)
        save_btn.pack(side="left")

    def replace_question_marks(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        self.df.replace('?', np.nan, inplace=True)
        self.log("Replaced all '?' values with NaN.")
        self.refresh_preview()
        self.refresh_info()
        messagebox.showinfo("Done", "'?' values replaced with NaN successfully.")

    def show_missing_summary(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        for row in self.missing_tree.get_children():
            self.missing_tree.delete(row)

        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if not missing_cols:
            self.log("No missing values found in the dataset.")
            messagebox.showinfo("Info", "No missing values found!")
            return

        for col in missing_cols:
            count = self.df[col].isnull().sum()
            dtype = str(self.df[col].dtype)
            self.missing_tree.insert("", tk.END, values=(col, count, dtype))

        self.log(f"Missing values found in columns: {missing_cols}")

    def apply_imputation(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        col = self.impute_col_var.get()
        strategy = self.strategy_var.get()
        if not col:
            messagebox.showerror("Error", "Please select a column.")
            return
        try:
            if strategy == "drop rows with NaN in this column":
                before = len(self.df)
                self.df.dropna(subset=[col], axis=0, inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                after = len(self.df)
                self.log(f"Dropped {before - after} rows with missing '{col}'. New shape: {self.df.shape}")
            else:
                imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
                self.df[[col]] = imputer.fit_transform(self.df[[col]])
                self.log(f"Column '{col}' missing values replaced using strategy = '{strategy}'.")

            self.refresh_preview()
            self.refresh_info()
            self.refresh_column_dropdowns()
            messagebox.showinfo("Success", f"Imputation applied on '{col}'.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def convert_dtype(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        col = self.dtype_col_var.get()
        target = self.dtype_target_var.get()
        if not col:
            messagebox.showerror("Error", "Please select a column.")
            return
        try:
            self.df[[col]] = self.df[[col]].astype(target)
            self.log(f"Column '{col}' converted to {target}.")
            self.refresh_preview()
            self.refresh_info()
            messagebox.showinfo("Success", f"'{col}' converted to {target}.")
        except Exception as e:
            messagebox.showerror("Conversion Error", str(e))

    def save_cleaned_csv(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             initialfile="cleaned_data.csv")
        if path:
            self.df.to_csv(path, index=False)
            self.log(f"Cleaned dataset saved to: {path}")
            messagebox.showinfo("Saved", f"Cleaned dataset saved as:\n{path}")

    # =================================================================
    # TAB 3: ENCODING (Categorical Values)
    # =================================================================
    def build_encoding_tab(self):
        # ---- Label Encoding ----
        label_frame = tk.LabelFrame(self.tab_encode, text="🔠 Label Encoding (best for binary / ordinal columns)",
                                     bg=CARD_BG, font=("Segoe UI", 10, "bold"), fg=TEXT_DARK, padx=10, pady=10)
        label_frame.pack(fill="x", padx=15, pady=(15, 8))

        tk.Label(label_frame, text="Column:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5)
        self.label_enc_col_var = tk.StringVar()
        self.label_enc_col_menu = ttk.Combobox(label_frame, textvariable=self.label_enc_col_var,
                                                state="readonly", width=25)
        self.label_enc_col_menu.grid(row=0, column=1, padx=5)

        label_btn = self.make_button(label_frame, "🏷️ Apply LabelEncoder", self.apply_label_encoding,
                                      BTN_SUCCESS, BTN_SUCCESS_HOVER)
        label_btn.grid(row=0, column=2, padx=10)

        # ---- One-Hot Encoding ----
        ohe_frame = tk.LabelFrame(self.tab_encode, text="🧩 One-Hot Encoding (best for nominal columns)",
                                   bg=CARD_BG, font=("Segoe UI", 10, "bold"), fg=TEXT_DARK, padx=10, pady=10)
        ohe_frame.pack(fill="x", padx=15, pady=8)

        tk.Label(ohe_frame, text="Column:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5)
        self.ohe_col_var = tk.StringVar()
        self.ohe_col_menu = ttk.Combobox(ohe_frame, textvariable=self.ohe_col_var, state="readonly", width=25)
        self.ohe_col_menu.grid(row=0, column=1, padx=5)

        tk.Label(ohe_frame, text="Method:", bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5)
        self.ohe_method_var = tk.StringVar(value="pandas.get_dummies")
        ohe_method_menu = ttk.Combobox(ohe_frame, textvariable=self.ohe_method_var, state="readonly",
                                        values=["pandas.get_dummies", "sklearn.OneHotEncoder"], width=24)
        ohe_method_menu.grid(row=0, column=3, padx=5)

        ohe_btn = self.make_button(ohe_frame, "🧩 Apply One-Hot Encoding", self.apply_one_hot_encoding,
                                    BTN_SUCCESS, BTN_SUCCESS_HOVER)
        ohe_btn.grid(row=0, column=4, padx=10)

        tk.Label(ohe_frame, text="Note: the original column is dropped and replaced by new dummy/one-hot columns.",
                 bg=CARD_BG, font=("Segoe UI", 8, "italic"), fg=TEXT_MUTED).grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))

        # ---- Preview after encoding ----
        preview_frame = tk.LabelFrame(self.tab_encode, text="Preview After Encoding", bg=CARD_BG,
                                       font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        preview_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.encode_preview_tree = self.make_treeview(preview_frame)

    def apply_label_encoding(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        col = self.label_enc_col_var.get()
        if not col:
            messagebox.showerror("Error", "Please select a column.")
            return
        try:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            self.log(f"LabelEncoder applied on '{col}'. Mapping: {mapping}")

            self.refresh_preview()
            self.refresh_info()
            self.refresh_column_dropdowns()
            self.populate_treeview(self.encode_preview_tree, self.df.head(10))
            messagebox.showinfo("Success", f"Label encoding applied on '{col}'.\nMapping: {mapping}")
        except Exception as e:
            messagebox.showerror("Encoding Error", str(e))

    def apply_one_hot_encoding(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        col = self.ohe_col_var.get()
        method = self.ohe_method_var.get()
        if not col:
            messagebox.showerror("Error", "Please select a column.")
            return
        try:
            if method == "pandas.get_dummies":
                # Mirrors: dataset_dummies = pd.get_dummies(dataset[col])
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                self.log(f"pandas.get_dummies applied on '{col}'. New columns: {list(dummies.columns)}")
            else:
                # Mirrors: OneHotEncoder(sparse_output=False)
                encoder = OneHotEncoder(sparse_output=False)
                encoded = encoder.fit_transform(self.df[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]),
                                           index=self.df.index)
                self.df = pd.concat([self.df.drop(col, axis=1), encoded_df], axis=1)
                self.log(f"sklearn.OneHotEncoder applied on '{col}'. New columns: {list(encoded_df.columns)}")

            self.refresh_preview()
            self.refresh_info()
            self.refresh_column_dropdowns()
            self.populate_treeview(self.encode_preview_tree, self.df.head(10))
            messagebox.showinfo("Success", f"One-hot encoding ({method}) applied on '{col}'.")
        except Exception as e:
            messagebox.showerror("Encoding Error", str(e))

    # =================================================================
    # TAB 4: FEATURE SCALING
    # =================================================================
    def build_scaling_tab(self):
        # ---- Train/Test Split ----
        split_frame = tk.LabelFrame(self.tab_scale, text="✂️ Train/Test Split", bg=CARD_BG,
                                     font=("Segoe UI", 10, "bold"), fg=TEXT_DARK, padx=10, pady=10)
        split_frame.pack(fill="x", padx=15, pady=(15, 8))

        tk.Label(split_frame, text="Target Column (y):", bg=CARD_BG,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5)
        self.target_col_var = tk.StringVar()
        self.target_col_menu = ttk.Combobox(split_frame, textvariable=self.target_col_var,
                                             state="readonly", width=22)
        self.target_col_menu.grid(row=0, column=1, padx=5)

        tk.Label(split_frame, text="Test Size:", bg=CARD_BG,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5)
        self.test_size_var = tk.StringVar(value="0.2")
        tk.Entry(split_frame, textvariable=self.test_size_var, width=6).grid(row=0, column=3, padx=5)

        tk.Label(split_frame, text="Random State:", bg=CARD_BG,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=4, padx=5)
        self.random_state_var = tk.StringVar(value="4")
        tk.Entry(split_frame, textvariable=self.random_state_var, width=6).grid(row=0, column=5, padx=5)

        self.shuffle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(split_frame, text="Shuffle", variable=self.shuffle_var, bg=CARD_BG,
                        font=("Segoe UI", 9)).grid(row=0, column=6, padx=8)

        split_btn = self.make_button(split_frame, "✂️ Split Dataset", self.perform_train_test_split,
                                      BTN_WARNING, BTN_WARNING_HOVER)
        split_btn.grid(row=0, column=7, padx=10)

        self.split_info_label = tk.Label(split_frame, text="No split performed yet.", bg=CARD_BG,
                                          font=("Segoe UI", 9, "italic"), fg=TEXT_MUTED)
        self.split_info_label.grid(row=1, column=0, columnspan=8, sticky="w", pady=(6, 0))

        # ---- Scaling ----
        scale_frame = tk.LabelFrame(self.tab_scale, text="⚖️ Apply Feature Scaling (fit on X_train, transform both)",
                                     bg=CARD_BG, font=("Segoe UI", 10, "bold"), fg=TEXT_DARK, padx=10, pady=10)
        scale_frame.pack(fill="x", padx=15, pady=8)

        tk.Label(scale_frame, text="Columns to scale (ctrl/shift+click for multiple):",
                 bg=CARD_BG, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="nw", padx=5)
        self.scale_cols_listbox = tk.Listbox(scale_frame, selectmode=tk.MULTIPLE, height=5,
                                              exportselection=False, font=("Consolas", 9), width=28)
        self.scale_cols_listbox.grid(row=0, column=1, rowspan=3, padx=5, pady=4)

        tk.Label(scale_frame, text="Scaler:", bg=CARD_BG,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5, sticky="w")
        self.scaler_type_var = tk.StringVar(value="StandardScaler")
        scaler_menu = ttk.Combobox(scale_frame, textvariable=self.scaler_type_var, state="readonly",
                                    values=["StandardScaler", "MinMaxScaler"], width=18)
        scaler_menu.grid(row=0, column=3, padx=5, sticky="w")

        apply_scale_btn = self.make_button(scale_frame, "⚖️ Apply Scaling", self.apply_feature_scaling,
                                            BTN_SUCCESS, BTN_SUCCESS_HOVER)
        apply_scale_btn.grid(row=1, column=2, columnspan=2, padx=5, pady=6, sticky="w")

        # ---- Preview ----
        preview_top = tk.Frame(self.tab_scale, bg=BG_COLOR)
        preview_top.pack(fill="x", padx=15, pady=(8, 0))

        tk.Label(preview_top, text="Preview:", bg=BG_COLOR,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        self.scale_preview_choice = tk.StringVar(value="X_train")
        preview_menu = ttk.Combobox(preview_top, textvariable=self.scale_preview_choice, state="readonly",
                                     values=["X_train", "X_test", "y_train", "y_test"], width=15)
        preview_menu.pack(side="left", padx=5)

        show_btn = self.make_button(preview_top, "👁️ Show Head", self.show_scaling_preview,
                                     BTN_PRIMARY, BTN_PRIMARY_HOVER)
        show_btn.pack(side="left", padx=6)

        describe_btn = self.make_button(preview_top, "📈 Show describe()", self.show_scaling_describe,
                                         BTN_PRIMARY, BTN_PRIMARY_HOVER)
        describe_btn.pack(side="left", padx=6)

        preview_frame = tk.LabelFrame(self.tab_scale, text="Split / Scaled Data Preview", bg=CARD_BG,
                                       font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        preview_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.scale_preview_tree = self.make_treeview(preview_frame)

    def perform_train_test_split(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        target = self.target_col_var.get()
        if not target:
            messagebox.showerror("Error", "Please select a target column (y).")
            return
        try:
            test_size = float(self.test_size_var.get())
            random_state = int(self.random_state_var.get())
            shuffle = self.shuffle_var.get()

            X = self.df.drop(columns=[target])
            y = self.df[target]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
            )

            self.scalers = {}  # reset scalers, they no longer apply to a fresh split
            numeric_cols = self.X_train.select_dtypes(include=np.number).columns.tolist()
            self.scale_cols_listbox.delete(0, tk.END)
            for c in numeric_cols:
                self.scale_cols_listbox.insert(tk.END, c)

            info = (f"Split done → X_train: {self.X_train.shape}, X_test: {self.X_test.shape}, "
                    f"y_train: {self.y_train.shape}, y_test: {self.y_test.shape}")
            self.split_info_label.config(text=info)
            self.log(info)
            self.show_scaling_preview()
            messagebox.showinfo("Success", info)
        except Exception as e:
            messagebox.showerror("Split Error", str(e))

    def apply_feature_scaling(self):
        if self.X_train is None or self.X_test is None:
            messagebox.showerror("Error", "Please perform a train/test split first.")
            return
        selected_indices = self.scale_cols_listbox.curselection()
        cols = [self.scale_cols_listbox.get(i) for i in selected_indices]
        if not cols:
            messagebox.showerror("Error", "Please select at least one column to scale.")
            return
        try:
            scaler_name = self.scaler_type_var.get()
            scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()

            # Ensure numeric dtype (mirrors X_train.astype(float) step in the notebook)
            self.X_train[cols] = self.X_train[cols].astype(float)
            self.X_test[cols] = self.X_test[cols].astype(float)

            # Fit on train, transform both (avoids data leakage)
            self.X_train[cols] = scaler.fit_transform(self.X_train[cols])
            self.X_test[cols] = scaler.transform(self.X_test[cols])

            for c in cols:
                self.scalers[c] = scaler_name

            self.log(f"{scaler_name} applied on columns {cols} (fit on X_train, transform on X_train & X_test).")
            self.show_scaling_preview()
            messagebox.showinfo("Success", f"{scaler_name} applied on: {cols}")
        except Exception as e:
            messagebox.showerror("Scaling Error", str(e))

    def _get_scaling_preview_df(self):
        choice = self.scale_preview_choice.get()
        mapping = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }
        data = mapping.get(choice)
        if data is None:
            return None
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data

    def show_scaling_preview(self):
        data = self._get_scaling_preview_df()
        if data is None:
            messagebox.showerror("Error", "Please perform a train/test split first.")
            return
        self.populate_treeview(self.scale_preview_tree, data.head(10))
        self.log(f"Showing preview of {self.scale_preview_choice.get()}.")

    def show_scaling_describe(self):
        data = self._get_scaling_preview_df()
        if data is None:
            messagebox.showerror("Error", "Please perform a train/test split first.")
            return
        try:
            desc = data.describe(include='all').reset_index().rename(columns={"index": "stat"})
            self.populate_treeview(self.scale_preview_tree, desc)
            self.log(f"Showing describe() of {self.scale_preview_choice.get()}.")
        except Exception as e:
            messagebox.showerror("Describe Error", str(e))

    # =================================================================
    # TAB 5: STATISTICS
    # =================================================================
    def build_stats_tab(self):
        top = tk.Frame(self.tab_stats, bg=BG_COLOR, pady=10)
        top.pack(fill="x", padx=15)

        btn1 = self.make_button(top, "📈 Describe (Numeric Only)", lambda: self.show_describe(False),
                                 BTN_PRIMARY, BTN_PRIMARY_HOVER)
        btn1.grid(row=0, column=0, padx=5)

        btn2 = self.make_button(top, "📋 Describe (All Columns)", lambda: self.show_describe(True),
                                 BTN_PRIMARY, BTN_PRIMARY_HOVER)
        btn2.grid(row=0, column=1, padx=5)

        stats_frame = tk.LabelFrame(self.tab_stats, text="Statistical Summary", bg=CARD_BG,
                                     font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        stats_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.stats_tree = self.make_treeview(stats_frame)

    def show_describe(self, include_all):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        desc = self.df.describe(include='all') if include_all else self.df.describe()
        desc = desc.reset_index().rename(columns={"index": "stat"})
        self.populate_treeview(self.stats_tree, desc)
        self.log("Displayed describe() summary.")

    # =================================================================
    # TAB 6: VISUALIZATION
    # =================================================================
    def build_viz_tab(self):
        top = tk.Frame(self.tab_viz, bg=BG_COLOR, pady=10)
        top.pack(fill="x", padx=15)

        tk.Label(top, text="Select Column:", font=("Segoe UI", 10, "bold"),
                  bg=BG_COLOR, fg=TEXT_DARK).grid(row=0, column=0, padx=5)
        self.viz_col_var = tk.StringVar()
        self.viz_col_menu = ttk.Combobox(top, textvariable=self.viz_col_var, state="readonly", width=25)
        self.viz_col_menu.grid(row=0, column=1, padx=5)

        hist_btn = self.make_button(top, "📊 Show Histogram (Mean/Median)", self.show_histogram,
                                     BTN_PRIMARY, BTN_PRIMARY_HOVER)
        hist_btn.grid(row=0, column=2, padx=8)

        outlier_btn = self.make_button(top, "🚨 Detect Outliers (IQR)", self.detect_outliers,
                                        BTN_DANGER, BTN_DANGER_HOVER)
        outlier_btn.grid(row=0, column=3, padx=8)

        self.plot_frame = tk.LabelFrame(self.tab_viz, text="Histogram", bg=CARD_BG,
                                         font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        self.plot_frame.pack(fill="both", expand=True, padx=15, pady=8)

        outlier_frame = tk.LabelFrame(self.tab_viz, text="Outlier Report", bg=CARD_BG,
                                       font=("Segoe UI", 10, "bold"), fg=TEXT_DARK)
        outlier_frame.pack(fill="both", expand=True, padx=15, pady=8)
        self.outlier_text = tk.Text(outlier_frame, height=8, font=("Consolas", 9), bg="#F9FAFB")
        self.outlier_text.pack(fill="both", expand=True, padx=6, pady=6)

    def show_histogram(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        col = self.viz_col_var.get()
        if not col:
            messagebox.showerror("Error", "Please select a column.")
            return
        try:
            data = pd.to_numeric(self.df[col], errors='coerce').dropna()

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data, bins=20, color="#4361EE", edgecolor="black")
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="green", linestyle="dashed", linewidth=2, label=f"Median: {median_val:.2f}")
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            self.log(f"Histogram displayed for column '{col}'. Mean={mean_val:.2f}, Median={median_val:.2f}")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def detect_outliers(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        col = self.viz_col_var.get()
        if not col:
            messagebox.showerror("Error", "Please select a column.")
            return
        try:
            data = pd.to_numeric(self.df[col], errors='coerce')
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data < lower) | (data > upper)]

            self.outlier_text.delete("1.0", tk.END)
            self.outlier_text.insert(tk.END, f"Column: {col}\n")
            self.outlier_text.insert(tk.END, f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}\n")
            self.outlier_text.insert(tk.END, f"Lower Bound={lower:.2f}, Upper Bound={upper:.2f}\n")
            self.outlier_text.insert(tk.END, f"Number of Outliers: {len(outliers)}\n\n")
            self.outlier_text.insert(tk.END, str(outliers))

            self.log(f"Outlier detection on '{col}': {len(outliers)} outliers found.")
        except Exception as e:
            messagebox.showerror("Outlier Detection Error", str(e))

    # =================================================================
    # LOG PANEL
    # =================================================================
    def build_log_panel(self):
        log_card = tk.Frame(self.root, bg=LOG_BG)
        log_card.pack(fill="x", padx=12, pady=(0, 10))

        header = tk.Frame(log_card, bg=LOG_BG)
        header.pack(fill="x")
        tk.Label(header, text="🕘 Activity Log", font=("Segoe UI", 10, "bold"),
                  bg=LOG_BG, fg="white", padx=10, pady=4).pack(side="left")
        clear_btn = tk.Button(header, text="Clear", command=self.clear_log, bg="#374151", fg="white",
                               relief="flat", font=("Segoe UI", 8, "bold"), cursor="hand2")
        clear_btn.pack(side="right", padx=10, pady=4)

        self.log_listbox = tk.Listbox(log_card, height=5, bg=LOG_BG, fg=LOG_FG,
                                       font=("Consolas", 9), relief="flat", highlightthickness=0)
        self.log_listbox.pack(fill="x", padx=10, pady=(0, 10))

    def log(self, message):
        self.log_listbox.insert(0, message)

    def clear_log(self):
        self.log_listbox.delete(0, tk.END)

    # =================================================================
    # HELPER METHODS
    # =================================================================
    def make_button(self, parent, text, command, bg, hover):
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg="white",
                         font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                         padx=10, pady=6, activebackground=hover, activeforeground="white")
        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    def make_treeview(self, parent, columns=None):
        container = tk.Frame(parent, bg=CARD_BG)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        vsb = ttk.Scrollbar(container, orient="vertical")
        hsb = ttk.Scrollbar(container, orient="horizontal")

        tree = ttk.Treeview(container, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        if columns:
            tree["columns"] = columns
            tree["show"] = "headings"
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150, anchor="center")

        return tree

    def populate_treeview(self, tree, dataframe):
        tree.delete(*tree.get_children())
        tree["columns"] = list(dataframe.columns)
        tree["show"] = "headings"
        for col in dataframe.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        for _, row in dataframe.iterrows():
            values = [str(v) for v in row.values]
            tree.insert("", tk.END, values=values)

    def refresh_column_dropdowns(self):
        cols = list(self.df.columns)
        self.impute_col_menu["values"] = cols
        self.dtype_col_menu["values"] = cols
        self.viz_col_menu["values"] = cols
        self.label_enc_col_menu["values"] = cols
        self.ohe_col_menu["values"] = cols
        self.target_col_menu["values"] = cols


if __name__ == "__main__":
    root = tk.Tk()
    app = DataProcessingApp(root)
    root.mainloop()
