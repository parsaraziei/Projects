import tkinter as tk
import pandas as pd
import time as time
import matplotlib.pyplot as plt
import datetime as date
from tkcalendar import DateEntry
import numpy as np
from tkinter import ttk
import utils
import models
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog
from PIL import Image, ImageTk, ImageSequence
from tkinter import Toplevel
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
import copy
from matplotlib.ticker import FuncFormatter
from tkinter import messagebox
from scipy.spatial import QhullError
from pathlib import Path



#global variables
current_case = None
# Arrays to store data
x_vals = []
lgn_plot_data = []
his_plot_data = []
jmp_diff_data = []
HS_plot_data=[]
Strike_vals = []
RH_plot_data = []

loading_overlay = None  # Keep reference globally
seed = np.random.randint(1, 10**9)

simulation_start_time = None
last_runtime = None
# Create main window
root = tk.Tk()

style = ttk.Style()
right_frame = tk.Frame(root)
right_frame.pack(side="right",anchor="ne",padx=10,pady=0)

style.configure('righttab.TNotebook', tabposition='nw')  # 'wn' = west-north (right side, top-down)
tabs = ttk.Notebook(right_frame,style='righttab.TNotebook')
tabs.pack(expand=True, fill='both')


BLS_frame = tk.Frame(tabs)
tabs.add(BLS_frame,text="Black Scholes")

jump_frame=tk.Frame(tabs)
tabs.add(jump_frame,text="Merton's Jump Diffusion")

heston_frame = tk.Frame(tabs)
tabs.add(heston_frame,text="Heston Stochastic Volatility")

R_Heston_frame = tk.Frame(tabs)
tabs.add(R_Heston_frame,text="Rough Heston")

submit_frame = tk.Frame(right_frame,bd=3,relief="ridge")
submit_frame.pack(expand=True, fill="both")

vol_surface_frame = tk.Frame(tabs)
tabs.add(vol_surface_frame,text = "Volatility Surface")


# graphs setup:
# Top-left BLS_frame to hold the graph
top_left_frame = tk.Frame(root)
top_left_frame.pack(side="left", padx=10, pady=0, anchor="nw")  # Top-left corner


# Create the figure and subplots
fig = Figure(figsize=(7, 5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid(True)
ax2.grid(True)

fig.subplots_adjust(hspace = 0.2)
# Embed figure into the top_left_frame
canvas = FigureCanvasTkAgg(fig, master=top_left_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()


def animate_gif(label, frames, index=0):
    frame = frames[index]
    label.configure(image=frame)
    index = (index + 1) % len(frames)
    label.after(100, animate_gif, label, frames, index)

def setup_calender():
    global BLS_frame, start_date,end_date
    #code concerning the option start date and end date
    today_time = pd.Timestamp.now()
    current_date = date.date(today_time.year,today_time.month,today_time.day)

    min_date = date.date(today_time.year - 3, today_time.month, today_time.day)
    max_date = date.date(today_time.year + 4, today_time.month, today_time.day)

    start_date_label = tk.Label(BLS_frame, text="Option Start Date:")
    start_date = DateEntry(BLS_frame, mindate = min_date, maxdate = current_date,year=current_date.year, month=current_date.month, day=current_date.day)
    start_date.bind("<<DateEntrySelected>>", lambda e: fetch_price_thread())
    start_date_label.grid(row=0, column=0, sticky="e",pady=(10,0))
    start_date.grid(row=0, column=1, sticky="w",pady=(10,0))

    end_date_label = tk.Label(BLS_frame,text="Option End Date:")
    end_date = DateEntry(BLS_frame, mindate = min_date ,maxdate = max_date ,year=current_date.year+1, month=current_date.month, day=current_date.day)
    end_date_label.grid(row=0, column=2, sticky="w",pady=(10,0))
    end_date.grid(row=0, column=3, sticky="w",pady=(10,0))


#  code concering inputs
def validate_decimal_input(new_value):
    if new_value == "":
        return True
    try:
        val = float(new_value)
        return val >= 0
    except ValueError:
        return False

    
vcmd = (BLS_frame.register(validate_decimal_input), "%P")


def setup_IR():
    global BLS_frame, selected_IR, user_interest_rate
    interest_rate_label = tk.Label(BLS_frame,text="Interest rate (r):")
    selected_IR = tk.StringVar(value="USDC")


    def toggle_IR():
        if(selected_IR.get() == "USDC"):
            user_interest_rate.config(state="disabled")
            user_interest_rate_lbl.config(fg="gray")
            percent_interest_rate_lbl.config(fg="gray")
        elif(selected_IR.get() == "treasury"):
            user_interest_rate.config(state="disabled")
            user_interest_rate_lbl.config(fg="gray")
            percent_interest_rate_lbl.config(fg="gray")
        elif(selected_IR.get() == "mean"):
            user_interest_rate.config(state="disabled")
            user_interest_rate_lbl.config(fg="gray")
            percent_interest_rate_lbl.config(fg="gray")
        else:
            user_interest_rate.config(state="normal")
            user_interest_rate_lbl.config(fg="black")
            percent_interest_rate_lbl.config(fg="black")

    USDC_radio = tk.Radiobutton(BLS_frame,text="USDC lending rate",variable=selected_IR, value="USDC",command=toggle_IR)
    treasury_radio = tk.Radiobutton(BLS_frame,text="US treasury bonds",variable=selected_IR, value="treasury",command=toggle_IR)
    input_radio = tk.Radiobutton(BLS_frame,text="User input",variable=selected_IR, value="input",command=toggle_IR)
    mean_radio = tk.Radiobutton(BLS_frame,text="Historical mean returns",variable=selected_IR, value="mean",command=toggle_IR)

    interest_rate_label.grid(row=3, column=0, sticky="e",pady=(20,0))
    USDC_radio.grid(row=4,column=0,sticky="e")
    treasury_radio.grid(row=4,column=1,sticky="e")
    input_radio.grid(row=4,column=2,sticky="e")
    mean_radio.grid(row=4,column=3,sticky="e")
    
    user_interest_rate_lbl = tk.Label(BLS_frame,text="Input interest rate:",fg="gray")
    percent_interest_rate_lbl = tk.Label(BLS_frame,text="%",fg="gray")
    user_interest_rate = tk.Entry(BLS_frame,state="disabled",validate="key", validatecommand = vcmd)

    user_interest_rate_lbl.grid(row=5,column=0, sticky="e")
    user_interest_rate.grid(row=5,column=1,sticky="e")
    percent_interest_rate_lbl.grid(row=5,column=2,sticky="w")


def fetch_price_thread():
    def worker():
        date_ = pd.to_datetime(start_date.get()).strftime("%Y-%m-%d")
        print(date_)
        print(selected_crypto.get())
        try:
            price = utils.get_initial_value(selected_crypto.get(), date_)
            current_price_lbl.after(0, lambda: current_price_lbl.config(text=f"{price} $"))
        except Exception as e:
            current_price_lbl.after(0, lambda: current_price_lbl.config(text="Error fetching"))
    threading.Thread(target=worker, daemon=True).start()


# choose crypto:
def setup_crypto():
    global BLS_frame, selected_crypto, start_date, current_price_lbl

    selected_crypto = tk.StringVar(value="BTC")
    crypto_select_lbl = tk.Label(BLS_frame, text="Select Crypto:")

    crypto_dropdown = tk.OptionMenu(
        BLS_frame,
        selected_crypto,
        "BTC", "ETH", "BNB", "SOL", "LTC", "──────────",
        "SHIB", "DOGE", "PEPE", "──────────",
        "USDC", "RAI", "──────────",
        "LINK", "UNI", "AAVE", "XRP",
        command=lambda val: fetch_price_thread()
    )

    menu = crypto_dropdown["menu"]
    menu.entryconfigure(5, state="disabled")
    menu.entryconfigure(9, state="disabled")
    menu.entryconfigure(12, state="disabled")

    crypto_select_lbl.grid(row=7, column=0, sticky="e", pady=(20, 0), padx=10)
    crypto_dropdown.grid(row=7, column=1, sticky="w", pady=(20, 0))

    current_price_lbl = tk.Label(BLS_frame, text="", fg="blue")
    current_price_lbl.grid(row=7, column=2, sticky="w", pady=(20, 0))


# volatility:
def setup_vol():
    global BLS_frame, selected_vol, user_vol_field
    selected_vol = tk.StringVar(value="RV")
    def toggle_vol():
        if(selected_vol.get() == "RV"):
            user_vol_field.config(fg="gray",state="disabled")
            warning_volatility.grid_remove()
            user_vol_field_lbl.config(fg="gray")
        elif(selected_vol.get() == "IV"):
            user_vol_field.config(fg="gray",state="disabled")
            warning_volatility.grid()
            user_vol_field_lbl.config(fg="gray")
        else:
            user_vol_field.config(fg="black",state="normal")
            warning_volatility.grid_remove()
            user_vol_field_lbl.config(fg="black")
            
    vol_select_lbl = tk.Label(BLS_frame,text="Select the type of the volatility (\u03C3):")
    realized_vol_radio = tk.Radiobutton(BLS_frame,text="Realized Volatillity (RV)",variable=selected_vol, value="RV",command=toggle_vol)
    implied_vol_radio = tk.Radiobutton(BLS_frame,text="Implied Volatility (IV)",variable=selected_vol, value="IV",command=toggle_vol)
    user_vol_input_radio = tk.Radiobutton(BLS_frame,text="User input",variable=selected_vol, value="input",command=toggle_vol)
    user_vol_field_lbl = tk.Label(BLS_frame,fg="gray",text="Input volatility:")
    user_vol_field = tk.Entry(BLS_frame,fg="gray",state="disabled",validate="key", validatecommand = vcmd)
    vol_select_lbl.grid(row=8,column = 0,sticky = "e",pady=(20,0))
    realized_vol_radio.grid(row=9,column = 0,sticky = "e")
    implied_vol_radio.grid(row=9,column = 1,sticky = "e")
    user_vol_input_radio.grid(row=9 ,column = 2,sticky = "e")
    user_vol_field_lbl.grid(row=10,column = 0,sticky = "e")
    user_vol_field.grid(row=10,column = 1,sticky = "e",)
    warning_volatility = tk.Label(BLS_frame,text="IV may not available for some cryptos, RV will be used",fg="red")
    warning_volatility.grid(row=11,column=0,columnspan=2,sticky='e')
    warning_volatility.grid_remove()


#Intial value:
def setup_initial():
    global selected_initial_value, initial_value_field
    def toggle_S0(selected_initial_value):
        if selected_initial_value == "Historical/Real":
            initial_value_input_lbl.config(state="disabled",fg="gray")
            initial_value_field.config(state="disabled",fg="gray")
            initial_value_input_lbl2.config(state="disabled",fg="gray")
        else:
            initial_value_input_lbl.config(state="normal",fg="black")
            initial_value_field.config(state="normal",fg="black")
            initial_value_input_lbl2.config(state="normal",fg="black")

    selected_initial_value = tk.StringVar(value="Historical/Real")  # default value
    initial_value_select_lbl = tk.Label(BLS_frame,text="Select Initial Value (S\u2080):")
    initial_value_dropdown = tk.OptionMenu(BLS_frame,selected_initial_value,"Historical/Real","Arbitrary",command=toggle_S0)

    initial_value_select_lbl.grid(row=12,column=0,sticky="e",pady=(20,0))
    initial_value_dropdown.grid(row=12,column=1,sticky="e",pady=(20,0))

    initial_value_input_lbl = tk.Label(BLS_frame,text="Input (S\u2080):",fg="gray",state="disabled")
    initial_value_field = tk.Entry(BLS_frame,fg="gray",state="disabled",validate="key", validatecommand=vcmd)
    initial_value_input_lbl2 = tk.Label(BLS_frame,text="$",fg="gray",state="disabled")

    initial_value_input_lbl.grid(row=13,column=0,sticky="e")
    initial_value_field.grid(row=13,column=1,sticky="e")
    initial_value_input_lbl2.grid(row=13,column=2,sticky="w")


#strike price
def setup_strike():
    global strike_field
    strike_field_lbl = tk.Label(BLS_frame,text="Input Strike Price (K):")
    strike_field = tk.Entry(BLS_frame,validate="key", validatecommand=vcmd)
    strike_field_lbl2 = tk.Label(BLS_frame,text="$")

    strike_field_lbl.grid(row=14,column=0,sticky="e",pady=(20,0))
    strike_field.grid(row=14,column=1,sticky="e",pady=(20,0))
    strike_field_lbl2.grid(row=14,column=2,sticky="w",pady=(20,0))


#btn
def register():
    global error
    error_message = ""
    if(int((pd.to_datetime(end_date.get())-pd.to_datetime(start_date.get())).days)>0):
        if((selected_IR.get() == "input" and get_ouput(user_interest_rate.get())>=0 and get_ouput(user_interest_rate.get())<40) or selected_IR.get() != "input"):
            if((selected_vol.get() == "input" and get_ouput(user_vol_field.get())>0.001 and get_ouput(user_vol_field.get())<3) or selected_vol.get() != "input"):
                if((selected_initial_value.get()=="Arbitrary" and get_ouput(initial_value_field.get())>0) or selected_initial_value.get()!="Arbitrary"):
                    if(get_ouput(strike_field.get())>0):
                        res,error_message=verify_jump_params()
                        if(res):
                            res,error_message=verify_HS_params()
                            if(res):
                                res1,error_message = verify_RH_params()
                                if(res1):
                                    global simulation_start_time
                                    simulation_start_time = time.time()  
                                    disable_all_widgets()
                                    clear_plots()
                                    extract_inputs()
                                    
                                    print("correct inputs for " + str(selected_crypto.get()))
                    else: error_message = "strike_error"
                else: error_message = "inital_val_error"
            else: error_message = "vol_error"
        else: error_message = "IR_error"
    else: error_message = "date_error"
    display_error_message(error_message)
         
    
def get_ouput(x):
    if (x is None or x == ''): return 0.0
    else: return float(x)
    
def get_output_nonzero(x):
    if (x is None or x == ''): return None
    else: return float(x)

def clear_plots():
    global ax1,ax2,x_vals,lgn_plot_data,his_plot_data,jmp_diff_data,Strike_vals,HS_plot_data,RH_plot_data
    lgn_plot_data = []
    his_plot_data = []
    jmp_diff_data = []
    x_vals=[]
    RH_plot_data = []
    HS_plot_data = []
    Strike_vals=[]
    ax1.clear()
    ax2.clear()

def disable_all_widgets():
    global BLS_frame, disable_list,submit_frame,vol_surface_frame,disable_list,re_enable_list
    disable_list,re_enable_list = [],[]
    for frame in (BLS_frame,jump_frame,submit_frame,heston_frame,vol_surface_frame,R_Heston_frame):
        for widget in frame.winfo_children():
            try:
                if widget['state'] == 'disabled':
                    disable_list.append(widget)
                else:
                    widget.configure(state='disabled')
                    re_enable_list.append(widget)
            except tk.TclError:
                # Some widgets (like Label or Frame) don't support state config
                pass
   
def verify_jump_params():
    global jump_mean_input,jump_rate_input,jump_std_input,JD_bool,uploaded_file_path
    
    if(JD_bool.get()==False):
        return True,""
    if(JD_radio_button.get()=="input"):
        if(get_ouput(jump_mean_input.get())>0.001 and get_ouput(jump_mean_input.get())<1000):
            if(get_ouput(jump_std_input.get())>0.0001 and get_ouput(jump_std_input.get())<1000):
                if(get_ouput(jump_rate_input.get())>0.001 and get_ouput(jump_rate_input.get())<10000): 
                    return True,""
                else:
                    return False, "mean-jump-missing"
            else:
                return False,"std-jump-missing"
        else:
            return False,"jump-rate-missing"   
    elif(JD_radio_button.get()=="implied"):
        if(uploaded_file_path != None):
            return True,""
        else:
            print("missing file path")
            return False,"jump-file-missing"
    else:
        return True,""
        
def verify_HS_params():
    global HS_vov_input,HS_radio_button,HS_kappa_input,HS_theta_input,HS_rho_input,HS_bool,uploaded_file_path_HS
    
    if(HS_bool.get()==False):
        return True,""
    if(HS_radio_button.get()=="input"):
       
        if((get_output_nonzero(HS_kappa_input.get()) is not None) and ((get_output_nonzero(HS_kappa_input.get()))>-1000) and ((get_output_nonzero(HS_kappa_input.get()))<1000)):
            if((get_output_nonzero(HS_rho_input.get()) is not None) and ((get_output_nonzero(HS_rho_input.get()))>= -1) and ((get_output_nonzero(HS_rho_input.get())) <= 1)):
                if(get_ouput(HS_vov_input.get())>0.0001 and get_ouput(HS_vov_input.get())<10000): 
                    if(((get_output_nonzero(HS_theta_input.get()))is not None) and get_output_nonzero(HS_theta_input.get())>0.0001 and get_output_nonzero(HS_theta_input.get())<10000 ):
                        return True,""
                    else:
                        return False,"theta-missing"
                else:
                    return False, "vov-missing"
            else:
                return False,"rho-missing"
        else:
            return False,"kappa-missing"   
    elif(HS_radio_button.get()=="implied"):
        if(uploaded_file_path_HS != None):
            return True,""
        else:
            print("missing file path")
            return False,"HS-file-missing"
    else:
        return True,""  

import threading

def extract_inputs():
    show_loading()
    def worker():
        global current_case, K
        S0 = K = IR = ta = sig = 0
        sym = beg_date = fin_date = ''

        sym = selected_crypto.get()
        beg_date = start_date.get()
        fin_date = end_date.get()
        K = get_ouput(strike_field.get())
        ta = utils.option_duration(start_date.get(), end_date.get())

        S0 = get_ouput(initial_value_field.get()) if selected_initial_value.get() == "Arbitrary" else utils.get_initial_value(sym, beg_date)

        if selected_IR.get() == "treasury":
            IR = utils.get_risk_free_treasury_rate(beg_date, fin_date)
        elif selected_IR.get() == "USDC":
            IR = utils.get_usdc_lend_rate(beg_date)
        elif selected_IR.get() == "mean":
            IR = utils.get_historical_mean(sym, beg_date)
        else:
            IR = get_ouput(user_interest_rate.get()) / 100

        if selected_vol.get() == "IV":
            sig = utils.get_implied_vol(sym, beg_date)
            if sig is None:
                sig = utils.get_realized_vol(sym, beg_date)
        elif selected_vol.get() == "RV":
            sig = utils.get_realized_vol(sym, beg_date)
        else:
            sig = get_ouput(user_vol_field.get())

        if(rnd_draw_type.get()=="Antithetic Variates (AV)"):
            AV = True
        else:
            AV = False
            
        current_case = models.models(S0, K, IR, ta, sig, sym, beg_date, fin_date, seed, int(sim_val_label.cget("text")), AV)

        if JD_bool.get():
            if JD_radio_button.get() == "input":
                poisson_rate = get_ouput(jump_rate_input.get())
                std_jump = get_ouput(jump_std_input.get())
                mean_jump = get_ouput(jump_mean_input.get())
            elif JD_radio_button.get() == "historical":
                poisson_rate, std_jump, mean_jump = utils.get_jump_diff_comp(sym, beg_date)
            else:
                file = pd.read_csv(uploaded_file_path)
                params = utils.calibrate_jump_fixed_sigma(
                    S=S0,
                    K_list=file["Strike"].values,
                    T_list=file["TimeToExpiry"].values,
                    r=IR,
                    option_prices=file["OptionPrice"].values,
                    sigma=sig
                )
                poisson_rate, std_jump, mean_jump = params["lambda"], params["sigma_j"], params["mu_j"]

            current_case.setup_jd(poisson_rate, std_jump, mean_jump)

        if HS_bool.get():
            if HS_radio_button.get() == "historical":
                vov, theta, rho, kappa, hurst = utils.get_heston_params(sym, beg_date)
            elif HS_radio_button.get() == "input":
                vov = get_output_nonzero(HS_vov_input.get())
                theta = get_output_nonzero(HS_theta_input.get())
                rho = get_output_nonzero(HS_rho_input.get())
                kappa = get_output_nonzero(HS_kappa_input.get())
            else:
                HS_file = pd.read_csv(uploaded_file_path_HS, index_col=False)
                #print(HS_file)
                st_time = pd.Timestamp.now()
                #print(f"IR is {IR}, v0 is {sig**2}, S0 is {S0}, strikes are {HS_file["Strike"].values}, Prices are {HS_file["OptionPrice"].values}, the TTE is {HS_file["TimeToExpiry"].values}")
                params = utils.heston_calibration_imp(HS_file["OptionPrice"].values, HS_file["Strike"].values, HS_file["TimeToExpiry"].values, S0, IR,sig**2)
                kappa = params["kappa"]
                theta = params["theta"]
                vov   = params["sigma"]
                rho   = params["rho"]
                
            
            current_case.setup_HS(kappa, theta, vov, rho)
        
        if RH_bool.get():
            #RH_vov, RH_theta, RH_rho, RH_kappa, RH_hurst
            if RH_selected_type.get() == "historical":
                if (HS_bool.get() and HS_radio_button.get()=="historical"):
                    RH_vov,RH_theta,RH_rho,RH_kappa,RH_hurst = vov,theta,rho,kappa,hurst
                else:
                    RH_vov,RH_theta,RH_rho,RH_kappa,RH_hurst = utils.get_heston_params(sym,beg_date)
            else:
                RH_vov,RH_theta,RH_rho,RH_kappa,RH_hurst = get_output_nonzero(RH_hurst_input.get()),get_output_nonzero(RH_theta_input.get()),get_output_nonzero(RH_rho_input.get()),get_output_nonzero(RH_kappa_input.get()),get_output_nonzero(RH_hurst_input.get())
            print("----------------------------------------")
            print(f"RH_vov {RH_vov}, RH_theta {RH_theta}, RH_rho {RH_rho}, RH_kappa {RH_kappa}, RH_hurst {RH_hurst}")    
            current_case.setup_RH(RH_kappa,RH_theta,RH_vov,RH_rho,RH_hurst)
                
        print(f"IR is {IR} and sigma is {sig}")
        
        root.after(0, lambda: finish_extract_and_plot(ta, beg_date))

    threading.Thread(target=worker, daemon=True).start()


def finish_extract_and_plot(ta, beg_date):
    plot_graphs(ta, beg_date)
    hide_loading()


# exclusive errors for specific faults in the setup
def display_error_message(error_code):
    global error
    error.grid_remove()
    match error_code:
        case "date_error":
            error.config(text="The input dates are wrong")
        case "IR_error":
            error.config(text="Out of Range Interest Rate")
        case "vol_error":
            error.config(text="The volatility is too large or small")
        case "inital_val_error":
            error.config(text="Invalid inital value (S)")
        case "strike_error":
            error.config(text="Invalid Stike (k) input")
        case "mean-jump-missing":
            error.config(text="Jump Mean Parameter is invalid")
        case "std-jump-missing":
            error.config(text="Jump Standard deviation is invalid")
        case "jump-rate-missing":
            error.config(text="Jump Frequency is invalid")
        case "jump-file-missing":
            error.config(text="Data file containing Jump Diffusion option\n prices is required for extraction and calibration of parameters")
        case "theta-missing":
             error.config(text="Long-term variance (mean level) is invalid (θ)")
        case "vov-missing":
            error.config(text="Invalid value in the Volatility of volatility\n (vol-of-vol) input (σ)")
        case "rho-missing":
             error.config(text="Correlation value is invalid (ρ)")
        case "kappa-missing":
            error.config(text="Incorrect or missing Mean reversion \nspeed value (κ)")
        case "HS-file-missing":
            error.config(text="Data file containing Heston Model option prices\n is required for extraction and calibration of parameters")
        case "RH_vov_missing":
            error.config(text="Invalid value in the Volatility of \nvolatility (vol-of-vol) input (σ) in Rough Heston")
        case "RH_kappa_missing":
            error.config(text="Incorrect or missing Mean reversion\n speed value (κ) in Rough Heston")   
        case "RH_rho_missing":
            error.config(text="Correlation value is invalid (ρ) in\n Rough Heston")
        case "RH_theta_missing":
            error.config(text="Long-term variance (mean level) is\n invalid (θ) in Rough Heston")
        case "RH_hurst_missing":
            error.config(text="The Hurst exponent (H) is invalid\n in Rough Heston")
        case _:
            error.config(text="")
    error.grid()
                             

def enable_all_widgets():
    global re_enable_list,disable_list
    for widget in re_enable_list:
        widget.configure(state='normal')
    

def setup_submission_btn():
    global error, rnd_draw_type, sim_val_label

    def on_slider_change(val):
        slider_val = int(val) * 10000
        sim_val_label.config(text=f"{slider_val}")

    rnd_draw_type = tk.StringVar(value="Regular Random Draws")
    draw_type_lbl = tk.Label(submit_frame,text="Type Of Random Draws:")
    draw_type_dropdown = tk.OptionMenu(submit_frame, rnd_draw_type,"Regular Random Draws", "Antithetic Variates (AV)")
    draw_type_dropdown.config(width=25)
    draw_type_dropdown.grid(row=0, column=1, sticky="w")  
    draw_type_lbl.grid(row=0,column=0,padx=(5,0),sticky="e")

    num_sims_slider = tk.Scale(submit_frame, from_=1, to=10, orient="horizontal",length= 150,command= on_slider_change, showvalue=0)
    num_sims_slider.grid(row=1, column=1, sticky="w")
    num_sims_slider.set(10) 
    num_sims_lbl = tk.Label(submit_frame,text="Number of Simulations")
    num_sims_lbl.grid(row=1,column=0,sticky="n",padx=(0,10),pady=(10,0))
    sim_val_label = tk.Label(submit_frame,text="100000")
    sim_val_label.grid(row=1,column=1,sticky="e")
    
    error = tk.Label(submit_frame, text="", fg="red")
    error.grid(row=0, column=2, sticky="e", padx=5)
    error.grid_remove()

    btn = tk.Button(submit_frame, text="Start Simulating", command=register, width=20, height=3)
    btn.grid(row=0,rowspan=2, column=2, sticky="se",padx=(100,0),pady=(40,0))  


#JD section
def initialize_JD():
    global JD_bool,JD_radio_button,jump_mean_input,jump_rate_input,jump_std_input,uploaded_file_path
    def enable_JD_components():
        if JD_bool.get():
            for widget in jump_frame.winfo_children():
                if not isinstance(widget, tk.Checkbutton):
                    try:
                        widget.configure(state='normal',fg="black")
                    except tk.TclError:
                        pass
            activate_fields()
            
        else:
            for widget in jump_frame.winfo_children():
               if not isinstance(widget, tk.Checkbutton):
                    try:
                        widget.configure(state='disabled',fg="gray")
                    except tk.TclError:
                        pass
       
               
    def activate_fields():
        if(JD_radio_button.get()=="input"):
            jump_rate_input.config(fg="black",state="normal")
            jump_mean_input.config(fg="black",state="normal")
            jump_std_input.config(fg="black",state="normal")
            upload_button.config(fg="gray",state="disabled")
            
            jump_rate_label.config(fg="black")
            jump_mean_label.config(fg="black")
            jump_std_label.config(fg="black")
            file_upload_label.config(fg="gray")
            path_label.config(fg="gray")
            
        elif(JD_radio_button.get()=="historical"):
            jump_rate_input.config(fg="gray",state="disabled")
            jump_mean_input.config(fg="gray",state="disabled")
            jump_std_input.config(fg="gray",state="disabled")
            upload_button.config(fg="gray",state="disabled")
            
            jump_rate_label.config(fg="gray")
            jump_mean_label.config(fg="gray")
            jump_std_label.config(fg="gray")
            file_upload_label.config(fg="gray")
            path_label.config(fg="gray")
        
        else:
            jump_rate_input.config(fg="gray",state="disabled")
            jump_mean_input.config(fg="gray",state="disabled")
            jump_std_input.config(fg="gray",state="disabled")
            upload_button.config(fg="black",state="normal")
            
            jump_rate_label.config(fg="gray")
            jump_mean_label.config(fg="gray")
            jump_std_label.config(fg="gray")
            file_upload_label.config(fg="black")
            if uploaded_file_path:
                path_label.config(fg="blue")
            else:
                path_label.config(fg="red")
        
    JD_bool = tk.BooleanVar(value=False)
    JD_activation = tk.Checkbutton(jump_frame,text="Activate Jump Diffusion model",variable=JD_bool,command=enable_JD_components)
    JD_activation.grid(row=0, column=1,columnspan=2, pady=(5, 10), sticky="")
    JD_param_label = tk.Label(jump_frame,text="Parameters:",fg="gray",font=("Arial", 11))
    
    JD_radio_button = tk.StringVar(value="")
    radio_input_jump = tk.Radiobutton(jump_frame, text="User Input Parameters", variable=JD_radio_button, value="input",state="disabled",command=activate_fields)
    radio_historical_jump = tk.Radiobutton(jump_frame, text="Historical Parameters", variable=JD_radio_button, value="historical",state="disabled",command=activate_fields)
    radio_implied_jump = tk.Radiobutton(jump_frame, text="Implied Parameters", variable=JD_radio_button, value="implied",state="disabled",command=activate_fields)
    JD_radio_button.set(value="input")

    JD_param_label.grid(row=1,column=0,pady=3,sticky="e")
    radio_input_jump.grid(row=2,column=1,pady=10,sticky="w")
    radio_historical_jump.grid(row=2,column=2,pady=10,sticky="w")
    radio_implied_jump.grid(row=2,column=3,pady=10,sticky="w")
    
    jump_rate_label = tk.Label(jump_frame, text="Jump Frequency (𝜆):",fg="gray")
    jump_std_label = tk.Label(jump_frame,text="Jump Standard Deviation(𝛿):",fg="gray")
    jump_mean_label = tk.Label(jump_frame,text="Mean Jump Size (𝜇𝐽):",fg="gray")
    
    jump_rate_input = tk.Entry(jump_frame,state="disabled",validate="key",validatecommand = vcmd)
    jump_std_input = tk.Entry(jump_frame,state="disabled",validate="key",validatecommand = vcmd)
    jump_mean_input = tk.Entry(jump_frame,state="disabled",validate="key",validatecommand = vcmd)
    
    jump_mean_label.grid(row=3,column=1,sticky="w")
    jump_rate_input.grid(row=3,column=2,sticky="w")
    
    jump_std_label.grid(row=4,column=1,sticky="w")
    jump_std_input.grid(row=4,column=2,sticky="w")
    
    jump_rate_label.grid(row=5,column=1,sticky="w")
    jump_mean_input.grid(row=5,column=2,sticky="w")
    
    uploaded_file_path = None
    def upload_file():
        global uploaded_file_path
        file_path = filedialog.askopenfilename(title="Select a CSV File",filetypes=[("CSV files", "*.csv;*.CSV"), ("All files", "*.*")])
        uploaded_file_path = file_path
        if file_path:
            path_label.config(text=f"Selected File:\n{file_path}",fg="blue")
    
    file_upload_label = tk.Label(jump_frame,text="File Upload:",fg="gray",font=("Arial", 11))
    file_upload_label.grid(row=6,column=0,pady=10,sticky="e")
    
    icon_img = Image.open(Path.cwd() / "upload.png")
    icon_img = icon_img.resize((40, 40))
    icon_tk = ImageTk.PhotoImage(icon_img)

    upload_button = tk.Button(jump_frame,text="Import File",command=upload_file,state="disabled",image=icon_tk,compound="left",padx=30,pady=10,fg="gray")
    
    upload_button.image = icon_tk  # Keep a reference to avoid garbage collection
    upload_button.grid(row=7, column=1, columnspan=2, sticky="n")

    path_label = tk.Label(jump_frame,text="No file selected!", wraplength=350, justify="left",fg="gray")
    path_label.grid(row = 8,column=1,columnspan=2,pady=10,sticky="n")
    
   
   
def initialize_HS():
    global HS_kappa_input,HS_rho_input,HS_theta_input,HS_vov_input,uploaded_file_path_HS,HS_radio_button,HS_bool
    
    def enable_JD_components():
        if HS_bool.get():
            for widget in heston_frame.winfo_children():
                if not isinstance(widget, tk.Checkbutton):
                    try:
                        widget.configure(state='normal',fg="black")
                    except tk.TclError:
                        pass
            activate_fields()
            
        else:
            for widget in heston_frame.winfo_children():
               if not isinstance(widget, tk.Checkbutton):
                    try:
                        widget.configure(state='disabled',fg="gray")
                    except tk.TclError:
                        pass
    
    
    def activate_fields():
        if(HS_radio_button.get()=="input"):
            HS_kappa_input.config(fg="black",state="normal")
            HS_rho_input.config(fg="black",state="normal")
            HS_theta_input.config(fg="black",state="normal")
            HS_vov_input.config(fg="black",state="normal")
            upload_button_HS.config(fg="gray",state="disabled")
            
            HS_theta_label.config(fg="black")
            HS_rho_label.config(fg="black")
            HS_kappa_label.config(fg="black")
            HS_vov_label.config(fg="black")
            file_upload_label_HS.config(fg="gray")
            path_label_HS.config(fg="gray")
            
        elif(HS_radio_button.get()=="historical"):
            HS_kappa_input.config(fg="gray",state="disabled")
            HS_rho_input.config(fg="gray",state="disabled")
            HS_theta_input.config(fg="gray",state="disable")
            HS_vov_input.config(fg="gray",state="disabled")
            upload_button_HS.config(fg="gray",state="disabled")
            
            HS_theta_label.config(fg="gray")
            HS_rho_label.config(fg="gray")
            HS_kappa_label.config(fg="gray")
            HS_vov_label.config(fg="gray")
            file_upload_label_HS.config(fg="gray")
            path_label_HS.config(fg="gray")
        
        else:
            HS_kappa_input.config(fg="gray",state="disabled")
            HS_rho_input.config(fg="gray",state="disabled")
            HS_theta_input.config(fg="gray",state="disable")
            HS_vov_input.config(fg="gray",state="disabled")
            upload_button_HS.config(fg="black",state="normal")
            
            HS_theta_label.config(fg="gray")
            HS_rho_label.config(fg="gray")
            HS_kappa_label.config(fg="gray")
            HS_vov_label.config(fg="gray")
            file_upload_label_HS.config(fg="black")
            if uploaded_file_path_HS:
                path_label_HS.config(fg="blue")
            else:
                path_label_HS.config(fg="red")
    
    HS_bool = tk.BooleanVar(value=False)
    HS_activation = tk.Checkbutton(heston_frame,text="Activate Heston Stochastic Volatility model",variable=HS_bool,command=enable_JD_components)
    HS_activation.grid(row=0, column=1,columnspan=2, pady=(5, 10), sticky="")
    HS_param_label = tk.Label(heston_frame,text="Parameters:",fg="gray",font=("Arial", 11))
    
    HS_radio_button = tk.StringVar(value="")
    radio_input_HS = tk.Radiobutton(heston_frame, text="User Input Parameters", variable=HS_radio_button, value="input",state="disabled",command=activate_fields)
    radio_historical_HS = tk.Radiobutton(heston_frame, text="Historical Parameters", variable=HS_radio_button, value="historical",state="disabled",command=activate_fields)
    radio_implied_HS = tk.Radiobutton(heston_frame, text="Implied Parameters", variable=HS_radio_button, value="implied",state="disabled",command=activate_fields)
    HS_radio_button.set(value="input")

    HS_param_label.grid(row=1,column=0,pady=3,sticky="w")
    radio_input_HS.grid(row=2,column=1,pady=10,sticky="w")
    radio_historical_HS.grid(row=2,column=2,pady=10,sticky="w")
    radio_implied_HS.grid(row=2,column=3,pady=10,sticky="w")
    
    HS_theta_label = tk.Label(heston_frame, text="Long Term Variance (θ):",fg="gray")
    HS_vov_label = tk.Label(heston_frame,text="Vol-of-vol (𝜎):",fg="gray")
    HS_kappa_label = tk.Label(heston_frame,text="Mean reversion speed (𝜅):",fg="gray")
    HS_rho_label = tk.Label(heston_frame,text="Correlation (𝜌):",fg="gray")
    
    HS_theta_input = tk.Entry(heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    HS_vov_input = tk.Entry(heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    HS_kappa_input = tk.Entry(heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    HS_rho_input = tk.Entry(heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    
    HS_theta_label.grid(row=3,column=0,sticky="w")
    HS_theta_input.grid(row=3,column=1,sticky="w")
    
    HS_vov_label.grid(row=3,column=2,sticky="e")
    HS_vov_input.grid(row=3,column=3,sticky="w")
    
    HS_kappa_label.grid(row=4,column=0,sticky="w")
    HS_kappa_input.grid(row=4,column=1,sticky="w")
    
    HS_rho_label.grid(row=4,column=2,sticky="e")
    HS_rho_input.grid(row=4,column=3,sticky="w")
    
    uploaded_file_path_HS = None
    def upload_file():
        global uploaded_file_path_HS
        file_path = filedialog.askopenfilename(title="Select a CSV File",filetypes=[("CSV files", "*.csv;*.CSV"), ("All files", "*.*")])
        uploaded_file_path_HS = file_path
        if file_path:
            path_label_HS.config(text=f"Selected File:\n{file_path}",fg="blue")
    
    file_upload_label_HS = tk.Label(heston_frame,text="File Upload:",fg="gray",font=("Arial", 11))
    file_upload_label_HS.grid(row=6,column=0,pady=15,padx=5,sticky="w")
    
    
    icon_img_HS = Image.open(Path.cwd() / "upload.png")
    icon_img_HS = icon_img_HS.resize((40, 40))
    icon_tk_HS = ImageTk.PhotoImage(icon_img_HS)

    upload_button_HS = tk.Button(heston_frame,text="Import File",command=upload_file,state="disabled",image=icon_tk_HS,compound="left",padx=10,pady=5,fg="gray")
    
    upload_button_HS.image = icon_tk_HS  # Keep a reference to avoid garbage collection
    upload_button_HS.grid(row=7, column=1, columnspan=2, sticky="n")

    path_label_HS = tk.Label(heston_frame,text="No file selected!", wraplength=350, justify="left",fg="gray")
    path_label_HS.grid(row = 8,column=1,columnspan=2,pady=10,sticky="n")
    
    
    

def initialize_vol_surf():
    global surf_vol_type,vol_surf_selected_crypto,vol_surf_selected_IR,vol_surf_uploaded_file_path\
        ,vol_surf_intial_price_inp,vol_surf_user_interest_rate,vol_surf_error_message,selected_vol_axis
    
    surf_vol_type = tk.StringVar(value="internet")
    vol_surf_selected_IR = tk.StringVar(value="USDC")
    
    def activate_fields():
        if(surf_vol_type.get()=="internet"):
            vol_surf_crypto_select_lbl.config(fg="black")
            vol_surf_crypto_dropdown.config(fg="black",state="normal")
            vol_surf_intial_price_inp.config(fg="gray",state="disabled")
            vol_surf_initial_price_lbl.config(fg="gray")
            vol_surf_dollar_lbl.config(fg="gray")
            vol_surf_upload_button.config(fg="gray",state="disabled")
            vol_surf_file_path_label.config(fg="gray")
            vol_surf_path_label.config(fg="gray")
        else:
            vol_surf_crypto_select_lbl.config(fg="gray")
            vol_surf_crypto_dropdown.config(fg="gray",state="disabled")
            vol_surf_intial_price_inp.config(fg="black",state="normal")
            vol_surf_initial_price_lbl.config(fg="black")
            vol_surf_dollar_lbl.config(fg="gray")
            vol_surf_upload_button.config(fg="black",state="normal")
            if vol_surf_uploaded_file_path:
                vol_surf_path_label.config(fg="blue")
            else:
                vol_surf_path_label.config(fg="red")
            
    vol_surf_internet_radio = tk.Radiobutton(vol_surface_frame, text="Online Data", variable=surf_vol_type, value="internet",command=activate_fields)
    vol_surf_file_radio = tk.Radiobutton(vol_surface_frame, text="Import CSV", variable=surf_vol_type, value="input",command=activate_fields)
    vol_surf_crypto_select_lbl = tk.Label(vol_surface_frame,text="Crypto:")
    vol_surf_general_specifications_label = tk.Label(vol_surface_frame,text="General Setup:")
    
    vol_surf_general_specifications_label.grid(row=0,column=0,sticky="e",pady=6,padx=5)
    vol_surf_internet_radio.grid(row=1,column=1,sticky="w",pady=3)
    vol_surf_file_radio.grid(row=1,column=2,sticky="w",pady=3)
    
    vol_surf_selected_crypto = tk.StringVar(value="BTC")
    vol_surf_crypto_dropdown = tk.OptionMenu(vol_surface_frame, vol_surf_selected_crypto, "BTC", "ETH")
    
    vol_surf_crypto_select_lbl.grid(row=4,column=3,sticky="w",pady=10)
    vol_surf_crypto_dropdown.grid(row=4,column=3,sticky="e",pady=10)
    
    vol_surf_file_path_label = tk.Label(vol_surface_frame, text="CSV-related Parameters:",fg="gray")
    vol_surf_file_path_label.grid(row=6,column=0,sticky="e",pady=(10,0))
    
    vol_surf_interest_rate_label = tk.Label(vol_surface_frame,text="Interest rate (r):")
    
    def toggle_IR():
        if(vol_surf_selected_IR.get() == "USDC"):
            vol_surf_user_interest_rate.config(state="disabled")
            vol_surf_user_interest_rate_lbl.config(fg="gray")
            vol_surf_percent_interest_rate_lbl.config(fg="gray")
        elif(vol_surf_selected_IR.get() == "treasury"):
            vol_surf_user_interest_rate.config(state="disabled")
            vol_surf_user_interest_rate_lbl.config(fg="gray")
            vol_surf_percent_interest_rate_lbl.config(fg="gray")
        elif(vol_surf_selected_IR.get() == "mean"):
            vol_surf_user_interest_rate.config(state="disabled")
            vol_surf_user_interest_rate_lbl.config(fg="gray")
            vol_surf_percent_interest_rate_lbl.config(fg="gray")
        else:
            vol_surf_user_interest_rate.config(state="normal")
            vol_surf_user_interest_rate_lbl.config(fg="black")
            vol_surf_percent_interest_rate_lbl.config(fg="black")

    vol_surf_USDC_radio = tk.Radiobutton(vol_surface_frame,text="USDC lending rate",variable=vol_surf_selected_IR, value="USDC",command=toggle_IR)
    vol_surf_treasury_radio = tk.Radiobutton(vol_surface_frame,text="US treasury bonds",variable=vol_surf_selected_IR, value="treasury",command=toggle_IR)
    vol_surf_input_radio = tk.Radiobutton(vol_surface_frame,text="User input",variable=vol_surf_selected_IR, value="input",command=toggle_IR)
    vol_surf_mean_radio = tk.Radiobutton(vol_surface_frame,text="Historical mean returns",variable=vol_surf_selected_IR, value="mean",command=toggle_IR)

    vol_surf_interest_rate_label.grid(row=2, column=0, sticky="e",pady=20)
    vol_surf_USDC_radio.grid(row=3,column=0,sticky="e")
    vol_surf_treasury_radio.grid(row=3,column=1,sticky="e")
    vol_surf_input_radio.grid(row=3,column=2,sticky="w")
    vol_surf_mean_radio.grid(row=3,column=3,sticky="w")
    
    vol_surf_user_interest_rate_lbl = tk.Label(vol_surface_frame,text="Input interest rate:",fg="gray")
    vol_surf_percent_interest_rate_lbl = tk.Label(vol_surface_frame,text="%",fg="gray")
    vol_surf_user_interest_rate = tk.Entry(vol_surface_frame,state="disabled",validate="key", validatecommand = vcmd)

    vol_surf_user_interest_rate_lbl.grid(row=4,column=0, sticky="e",pady=(3,0))
    vol_surf_user_interest_rate.grid(row=4,column=1,sticky="e",pady=(3,0))
    vol_surf_percent_interest_rate_lbl.grid(row=4,column=2,sticky="w",pady=(3,0))
    
    vol_surf_initial_price_lbl = tk.Label(vol_surface_frame,text="Intial Price (S0):",fg="gray")
    vol_surf_intial_price_inp = tk.Entry(vol_surface_frame,state="disabled")
    vol_surf_dollar_lbl = tk.Label(vol_surface_frame,text="$",fg="gray")
    
    vol_surf_initial_price_lbl.grid(row=7,column=0,sticky="e",pady=(5,0))
    vol_surf_intial_price_inp.grid(row=7,column=1,sticky="w",pady=(5,0))
    vol_surf_dollar_lbl.grid(row=7,column=2,sticky="w")
    
    vol_surf_uploaded_file_path = None
    def upload_file():
        global vol_surf_uploaded_file_path
        file_path = filedialog.askopenfilename(title="Select a CSV File",filetypes=[("CSV files", "*.csv;*.CSV"), ("All files", "*.*")])
        vol_surf_uploaded_file_path = file_path
        if file_path:
            vol_surf_path_label.config(text=f"Selected File:\n{file_path}",fg="blue")
    
    
    icon_img = Image.open(Path.cwd() / "upload.png")
    icon_img = icon_img.resize((40, 40))
    icon_tk = ImageTk.PhotoImage(icon_img)

    vol_surf_upload_button = tk.Button(vol_surface_frame,text="Import File",command=upload_file,state="disabled",image=icon_tk,compound="left",padx=7,pady=7,fg="gray")
    
    vol_surf_upload_button.image = icon_tk  # Keep a reference to avoid garbage collection
    vol_surf_upload_button.grid(row=7, column=2, columnspan=2, sticky="e")

    vol_surf_path_label = tk.Label(vol_surface_frame,text="No file selected!", wraplength=350, justify="left",fg="gray")
    vol_surf_path_label.grid(row = 8,column=2,columnspan=2,pady=10,sticky="e")
    
    vol_surf_show_btn = tk.Button(vol_surface_frame, text="➤ Plot Surface Volatility",width=30,height=2,font=("Segoe UI", 12, "italic"),cursor="hand2",command=register_vol_surf)
    vol_surf_show_btn.grid(row=9, column=1, columnspan=4, sticky="w")
    
    vol_surf_error_message = tk.Label(vol_surface_frame,text="",fg="red")
    vol_surf_error_message.grid(row=10, column=1 , columnspan = 4,sticky="w")
    
    
    selected_vol_axis = tk.StringVar(value="Strike")
    vol_surf_axis_label = tk.Label(vol_surface_frame,text="X-Axis:")
    vol_surf_axis_dropdown = tk.OptionMenu(vol_surface_frame, selected_vol_axis, "Strike","Moneyness","Log-Moneyness") 
    vol_surf_axis_dropdown.config(width=15)
    vol_surf_axis_dropdown.grid(row=8, column=1,sticky="w")
    vol_surf_axis_label.grid(row=8,column=0,sticky="e")
    
    
def verify_RH_params():
    if(RH_bool.get() == False):
        return True,""
    if(RH_selected_type.get()=="historical"):
        return True,""
    else:
        if(get_output_nonzero(RH_theta_input.get()) is not None and get_output_nonzero(RH_theta_input.get())<200 and get_output_nonzero(RH_theta_input.get())>0):
            if(get_output_nonzero(RH_rho_input.get()) is not None and get_output_nonzero(RH_rho_input.get())<=1 and get_output_nonzero(RH_rho_input.get())>=-1):
                if(get_output_nonzero(RH_kappa_input.get()) is not None and get_output_nonzero(RH_kappa_input.get())<1000 and get_output_nonzero(RH_kappa_input.get())>-1000):
                    if(get_output_nonzero(RH_vov_input.get()) is not None and get_output_nonzero(RH_vov_input.get())>0 and get_output_nonzero(RH_vov_input.get())<1000):
                        if(get_output_nonzero(RH_hurst_input.get()) is not None and get_output_nonzero(RH_hurst_input.get())>=0 and get_output_nonzero(RH_hurst_input.get())<=1):
                            return True,""
                        else:
                            return False,"RH_hurst_missing"
                    else:
                       return False,"RH_vov_missing"
                else:
                    return False,"RH_kappa_missing"
            else:
                return False,"RH_rho_missing"
        else:
            return False,"RH_theta_missing"
        
def register_vol_surf():
    res, message = validate_vol_surface_inputs()
    display_vol_error(message)

    if not res:
        return

    disable_all_widgets()
    show_loading()

    def run_background():
        try:
            crypto_symbol = vol_surf_selected_crypto.get()
            r = fetch_r(vol_surf_selected_IR.get(), crypto_symbol)
            S = get_output_nonzero(vol_surf_intial_price_inp.get())

            if surf_vol_type.get() == "internet":
                df, S = utils.prepare_vol_surface_data(crypto_symbol, r)
            else:
                df, S = utils.prepare_vol_surface_data(crypto_symbol, r, vol_surf_uploaded_file_path, S)

            # Use Tkinter's thread-safe method to show the plot on main thread
            root.after(0, lambda: plot_surface_from_dataframe(df, S))
        finally:
            
            root.after(0, hide_loading)
            #root.after(0, enable_all_widgets)

    threading.Thread(target=run_background, daemon=True).start()


def plot_surface_from_dataframe(df, spot_price, currency="", title="Implied Volatility Surface"):
    global vol_plot_frame
    
    xaxis = selected_vol_axis.get().lower() 
    
    # Prepare data
    X1 = df[xaxis].to_numpy()
    Y1 = df['time_to_expiry (days)'].to_numpy()
    Z1 = df['implied_vol'].to_numpy()

    print(f"Data summary: {len(df)} points, {len(np.unique(X1))} unique {xaxis}, {len(np.unique(Y1))} unique expiries")

    # --- More permissive data validation ---
    unique_expiries = np.unique(Y1)
    unique_x = np.unique(X1)

    # Create grid for surface plot - use more conservative parameters
    x_grid_vals = np.linspace(df[xaxis].min(), df[xaxis].max(), 50)  # Reduced from 80
    expiry_grid_vals = np.sort(df['time_to_expiry (days)'].unique())
    x_grid, expiry_grid = np.meshgrid(x_grid_vals, expiry_grid_vals)
    
    # --- Robust interpolation with multiple fallback strategies ---
    z_grid = None
    interpolation_success = False
    
    # Try multiple interpolation strategies in order of robustness
    strategies = [
        {'method': 'linear', 'description': 'linear interpolation'},
        {'method': 'nearest', 'description': 'nearest neighbor'},
    ]
    
    for strategy in strategies:
        try:
            print(f"Attempting {strategy['description']}...")
            z_grid = griddata((X1, Y1), Z1, (x_grid, expiry_grid), method=strategy['method'])
            
            # Check if we got meaningful results
            valid_points = np.sum(~np.isnan(z_grid))
            total_points = z_grid.size
            valid_ratio = valid_points / total_points
            
            print(f"Interpolation result: {valid_points}/{total_points} valid points ({valid_ratio:.1%})")
            
            if valid_ratio > 0.3:  # Only require 30% valid points
                interpolation_success = True
                print(f"Success with {strategy['description']}")
                break
            else:
                print(f"Too many NaN values with {strategy['description']}, trying next method...")
                
        except QhullError as e:
            print(f"QhullError with {strategy['description']}: {e}")
            continue
        except Exception as e:
            print(f"Error with {strategy['description']}: {e}")
            continue
    
    # If all methods fail, try a very simple approach
    if not interpolation_success:
        print("All interpolation methods failed, attempting fallback...")
        try:
            # Use nearest with a coarser grid
            x_grid_vals_fallback = np.linspace(df[xaxis].min(), df[xaxis].max(), 20)
            expiry_grid_vals_fallback = np.sort(df['time_to_expiry (days)'].unique())
            x_grid_fallback, expiry_grid_fallback = np.meshgrid(x_grid_vals_fallback, expiry_grid_vals_fallback)
            
            z_grid = griddata((X1, Y1), Z1, (x_grid_fallback, expiry_grid_fallback), method='nearest')
            interpolation_success = True
            print("Success with fallback nearest interpolation")
        except Exception as e:
            print(f"Fallback also failed: {e}")
    
    # If still no success, show 2D plots instead
    if not interpolation_success or z_grid is None:
        print("Cannot create 3D surface, offering 2D volatility smiles instead")
        create_volatility_smiles_only(df, spot_price, currency, xaxis)
        enable_all_widgets()
        return

    # Handle remaining NaN values more gracefully
    z_median = np.nanmedian(Z1) if len(Z1) > 0 else 0.5
    z_grid = np.nan_to_num(z_grid, nan=z_median)
    
    # Apply gentle smoothing if we have enough data
    try:
        if np.max(z_grid) - np.min(z_grid) > 1e-6:  # Only smooth if we have variation
            z_grid = gaussian_filter(z_grid, sigma=0.8)  # Reduced smoothing
    except Exception as e:
        print(f"Smoothing failed: {e}")

    # --- Create visualization ---
    window = Toplevel()
    window.title(f"{title} - {currency}")
    window.resizable(True, True)
    window.protocol("WM_DELETE_WINDOW", lambda: [enable_all_widgets(), window.destroy()])
    
    vol_plot_frame = tk.Frame(window)
    vol_plot_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create plot frame
    plot_frame = tk.Frame(vol_plot_frame)
    plot_frame.grid(row=0, column=0, sticky="nsew")
    
    # Create 3D plot
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set dynamic color range
    positive_vols = z_grid[z_grid > 0]
    if len(positive_vols) > 0:
        vmin, vmax = np.percentile(positive_vols, 10), np.percentile(positive_vols, 90)
    else:
        vmin, vmax = np.min(z_grid), np.max(z_grid)
    
    # Ensure valid range
    if vmax <= vmin:
        vmin, vmax = 0, max(1.0, np.max(z_grid))
        
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cm.inferno(norm(z_grid))
    
    # Plot surface with error handling
    try:
        surf = ax.plot_surface(x_grid, expiry_grid, z_grid, 
                             facecolors=colors, edgecolor='none',
                             rstride=1, cstride=1, alpha=0.95, 
                             antialiased=True)
    except Exception as e:
        print(f"3D plot failed: {e}, falling back to 2D smiles")
        window.destroy()
        create_volatility_smiles_only(df, spot_price, currency, xaxis)
        enable_all_widgets()
        return
    
    # Add market data points
    ax.scatter(X1, Y1, Z1, c='red', s=15, label='Market Data', alpha=0.6)
    
    # Configure plot
    ax.view_init(elev=25, azim=-120)
    ax.set_title(f"{currency} Volatility Surface (Spot: {spot_price:.2f})", fontsize=10)
    ax.set_xlabel(xaxis, fontsize=8)
    ax.set_ylabel("Time to Expiry (days)", fontsize=8)
    ax.set_zlabel("Implied Volatility", fontsize=8)
    ax.legend(fontsize=6)
    
    # Add colorbar
    try:
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        mappable.set_array(z_grid)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Implied Volatility')
    except Exception as e:
        print(f"Colorbar creation failed: {e}")

    fig.tight_layout()
    
    # Embed in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=0, columnspan=3, sticky="nsew")
    
    # Add volatility smile controls (same as before)
    control_frame = tk.Frame(vol_plot_frame)
    control_frame.grid(row=1, column=0, sticky="ew")
    
    vol_smile_lbl = tk.Label(control_frame, text="Extract volatility smile for selected expiry:")
    vol_smile_lbl.grid(row=0, column=0, sticky="e")
    
    expiry_options = sorted(df['time_to_expiry (days)'].unique())
    expiry_var = tk.StringVar(value=expiry_options[0] if len(expiry_options) > 0 else "")
    expiry_menu = tk.OptionMenu(control_frame, expiry_var, *expiry_options)
    expiry_menu.grid(row=0, column=1, sticky="n")
    
    def plot_volatility_smile():
        target_expiry = float(expiry_var.get())
        
        if not hasattr(vol_plot_frame, 'smile_frame'):
            vol_plot_frame.smile_frame = tk.Frame(vol_plot_frame)
            vol_plot_frame.smile_frame.grid(row=2, column=0, sticky="nsew")
        else:
            for widget in vol_plot_frame.smile_frame.winfo_children():
                widget.destroy()
        
        fig_smile = Figure(figsize=(6, 3))
        ax_smile = fig_smile.add_subplot(111)
        
        expiry_mask = np.isclose(df['time_to_expiry (days)'], target_expiry, atol=0.5)
        expiry_data = df[expiry_mask].sort_values(xaxis)
        
        if len(expiry_data) >= 2:  # Reduced requirement from 3 to 2
            ax_smile.plot(expiry_data[xaxis], expiry_data['implied_vol'], 
                         'bo-', markersize=5, label='Market Data', linewidth=1)
            
            if len(expiry_data) >= 3:
                from scipy.interpolate import CubicSpline
                try:
                    cs = CubicSpline(expiry_data[xaxis], expiry_data['implied_vol'])
                    x_smooth = np.linspace(expiry_data[xaxis].min(), expiry_data[xaxis].max(), 100)
                    ax_smile.plot(x_smooth, cs(x_smooth), 'r-', label='Spline Fit', alpha=0.7)
                except Exception as e:
                    print(f"Spline failed: {e}")
            
            if xaxis == 'moneyness':
                ax_smile.axvline(x=1.0, color='g', linestyle='--', label='ATM')
            
            ax_smile.set_title(f"{currency} Volatility Smile\nExpiry: {target_expiry} days (Spot: {spot_price:.2f})")
            ax_smile.set_xlabel(xaxis)
            ax_smile.set_ylabel("Implied Volatility")
            ax_smile.grid(True, alpha=0.3)
            ax_smile.legend()
        
            if xaxis == 'moneyness':
                def moneyness_to_strike(x):
                    return x * spot_price
                def strike_to_moneyness(x):
                    return x / spot_price
                secax = ax_smile.secondary_xaxis('top', functions=(moneyness_to_strike, strike_to_moneyness))
                secax.set_xlabel("Strike Price")
            
            fig_smile.tight_layout()
            
            canvas_smile = FigureCanvasTkAgg(fig_smile, master=vol_plot_frame.smile_frame)
            canvas_smile.draw()
            canvas_smile.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(vol_plot_frame.smile_frame, 
                    text=f"Not enough data for selected expiry ({len(expiry_data)} points)").pack()
    
    smile_btn = tk.Button(control_frame, text="Plot Volatility Curve",
                         command=plot_volatility_smile, width=20)
    smile_btn.grid(row=0, column=2, sticky="e", padx=10)

def create_volatility_smiles_only(df, spot_price, currency, xaxis):
    """Fallback function to create only 2D volatility smiles when 3D fails"""
    window = Toplevel()
    window.title(f"{currency} Volatility Smiles")
    window.geometry("800x600")
    window.protocol("WM_DELETE_WINDOW", lambda: [enable_all_widgets(), window.destroy()])
    
    # Create notebook for different expiries
    notebook = ttk.Notebook(window)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    unique_expiries = sorted(df['time_to_expiry (days)'].unique())
    
    for expiry in unique_expiries:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"{expiry} days")
        
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        expiry_data = df[np.isclose(df['time_to_expiry (days)'], expiry, atol=0.5)].sort_values(xaxis)
        
        if len(expiry_data) > 0:
            ax.plot(expiry_data[xaxis], expiry_data['implied_vol'], 'bo-', 
                   markersize=6, linewidth=2, label='Implied Volatility')
            
            if xaxis == 'moneyness':
                ax.axvline(x=1.0, color='red', linestyle='--', label='ATM', alpha=0.7)
            
            ax.set_title(f"{currency} Volatility Smile - {expiry} days to expiry\n(Spot: ${spot_price:.2f})")
            ax.set_xlabel(xaxis)
            ax.set_ylabel("Implied Volatility")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if xaxis == 'moneyness':
                def moneyness_to_strike(x):
                    return x * spot_price
                def strike_to_moneyness(x):
                    return x / spot_price
                secax = ax.secondary_xaxis('top', functions=(moneyness_to_strike, strike_to_moneyness))
                secax.set_xlabel("Strike Price")
        
        else:
            ax.text(0.5, 0.5, f"No data for {expiry} days expiry", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"No Data - {expiry} days")
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
def fetch_r(interest_rate_type,crypto):   
    beg_date = pd.Timestamp.today().date()
    fin_date = beg_date + pd.Timedelta(days=90)
    IR = 0
    if(interest_rate_type == "treasury"):
        IR = utils.get_risk_free_treasury_rate(beg_date,fin_date)
    elif (interest_rate_type=="USDC"):
        IR = utils.get_usdc_lend_rate(beg_date)
    elif (interest_rate_type=="mean"):
        IR = utils.get_historical_mean(crypto,beg_date)
    else:
        IR = get_ouput(vol_surf_user_interest_rate.get())/100
    return IR
    
    
def display_vol_error(message):
    vol_surf_error_message.config(text=message)
    
          
    
def validate_vol_surface_inputs():
    if vol_surf_selected_IR.get() == "input":
        if get_output_nonzero(vol_surf_user_interest_rate.get()) is not None:
            pass
        else:
            return False, "The interest rate value is invalid"

    if surf_vol_type.get() == "input":
        if get_output_nonzero(vol_surf_intial_price_inp.get()) is None:
            return False, "Invalid value for the initial Crypto Price"

        if not vol_surf_uploaded_file_path:
            return False, "No option prices file to plot the volatility surface"
        
    return True, ""


def initialize_RH():
    global RH_bool,RH_selected_type,RH_hurst_input,RH_kappa_input,RH_theta_input,RH_vov_input,RH_rho_input
    
    def enable_RH():
        if(RH_bool.get()==False):
            for widget in R_Heston_frame.winfo_children():
                if not isinstance(widget, tk.Checkbutton):
                    try:
                        widget.configure(state='disabled',fg="gray")
                    except tk.TclError:
                        pass
        else:
            R_HS_params_label.config(fg="black",state="normal")
            R_HS_hist_radio.config(state="normal",fg="black")
            R_HS_input_radio.config(state="normal",fg="black")
            toggle_RH()
            
        
    def toggle_RH():
        if(RH_selected_type.get()=="historical"):
            RH_kappa_input.config(state="disable",fg="gray")
            RH_theta_input.config(state="disable",fg="gray")
            RH_rho_input.config(state="disable",fg="gray")
            RH_hurst_input.config(state="disable",fg="gray")
            RH_vov_input.config(state="disable",fg="gray")
            
            RH_kappa_label.config(state="disabled",fg="gray")
            RH_theta_label.config(state="disabled",fg="gray")
            RH_rho_label.config(state="disabled",fg="gray")
            RH_hurst_label.config(state="disabled",fg="gray")
            RH_vov_label.config(state="disabled",fg="gray")
            RH_copy_params_button.config(state="disabled",fg="gray")
        else:
            RH_kappa_input.config(state="normal",fg="black")
            RH_theta_input.config(state="normal",fg="black")
            RH_rho_input.config(state="normal",fg="black")
            RH_hurst_input.config(state="normal",fg="black")
            RH_vov_input.config(state="normal",fg="black")
            
            RH_kappa_label.config(state="normal",fg="black")
            RH_theta_label.config(state="normal",fg="black")
            RH_rho_label.config(state="normal",fg="black")
            RH_hurst_label.config(state="normal",fg="black")
            RH_vov_label.config(state="normal",fg="black")
            RH_copy_params_button.config(state="normal",fg="black")
    
    def copy_params():
        RH_kappa_input.delete(0, tk.END)
        RH_kappa_input.insert(0,(HS_kappa_input.get()))
        RH_rho_input.delete(0, tk.END)
        RH_rho_input.insert(0,(HS_rho_input.get()))
        RH_theta_input.delete(0, tk.END)
        RH_theta_input.insert(0,(HS_theta_input.get()))
        RH_vov_input.delete(0, tk.END)
        RH_vov_input.insert(0,(HS_vov_input.get()))
        
    
    RH_bool = tk.BooleanVar(value=False)
    RH_activation = tk.Checkbutton(R_Heston_frame,text="Activate Rough Heston",variable=RH_bool,command = enable_RH)
    RH_activation.grid(row=0,column=1,sticky="n",columnspan=2)
    
    RH_selected_type = tk.StringVar(value="historical")
    R_HS_input_radio = tk.Radiobutton(R_Heston_frame,variable=RH_selected_type,value="input",text="User Input",state="disabled",command=toggle_RH)
    R_HS_hist_radio = tk.Radiobutton(R_Heston_frame,variable=RH_selected_type,value="historical",text="Historical Parameters",state="disabled",command=toggle_RH)
    
    R_HS_params_label = tk.Label(R_Heston_frame,text="Parameters:",fg="gray")
    R_HS_params_label.grid(row=1,column=0,sticky="w")
    
    R_HS_input_radio.grid(row=2,column=0,columnspan=2,sticky="n")
    R_HS_hist_radio.grid(row=2,column=2,columnspan=2,sticky="n")
    
    RH_theta_label = tk.Label(R_Heston_frame, text="Long Term Variance (θ):",fg="gray")
    RH_vov_label = tk.Label(R_Heston_frame,text="Vol-of-vol (𝜎):",fg="gray")
    RH_kappa_label = tk.Label(R_Heston_frame,text="Mean reversion speed (𝜅):",fg="gray")
    RH_rho_label = tk.Label(R_Heston_frame,text="Correlation (𝜌):",fg="gray")
    RH_hurst_label = tk.Label(R_Heston_frame,text="Hurst Exponent (H):",fg="gray")
    
    RH_theta_input = tk.Entry(R_Heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    RH_vov_input = tk.Entry(R_Heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    RH_kappa_input = tk.Entry(R_Heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    RH_rho_input = tk.Entry(R_Heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    RH_hurst_input = tk.Entry(R_Heston_frame,state="disabled",validate="key",validatecommand = vcmd)
    
    RH_theta_label.grid(row=3,column=0,sticky="e",pady=7)
    RH_theta_input.grid(row=3,column=1,sticky="w",pady=7)
    
    RH_vov_label.grid(row=3,column=2,sticky="e",pady=7)
    RH_vov_input.grid(row=3,column=3,sticky="w",pady=7)
    
    RH_kappa_label.grid(row=4,column=0,sticky="e",pady=7)
    RH_kappa_input.grid(row=4,column=1,sticky="w",pady=7)
    
    RH_rho_label.grid(row=4,column=2,sticky="e",pady=7)
    RH_rho_input.grid(row=4,column=3,sticky="w",pady=7)
    
    RH_hurst_label.grid(column=1,row=5,sticky="e",pady=7)
    RH_hurst_input.grid(column=2,row=5,sticky="w",pady=7)
    
    RH_copy_params_button=tk.Button(R_Heston_frame,state="disabled",text="Copy Heston Parameters",command=copy_params,width=25)
    RH_copy_params_button.grid(row=6,column=1,columnspan=2,sticky="n",pady=10)
    
def create_toolbar():
    global toolbar
    toolbar = NavigationToolbar2Tk(canvas, top_left_frame)
    toolbar.update()
    toolbar.pack_forget()
    
def create_form():
    create_toolbar()
    initialize_BLS()
    initialize_JD()
    initialize_HS()
    initialize_vol_surf()
    initialize_RH()
    setup_submission_btn()
    
        
def initialize_BLS():
    setup_IR()
    setup_calender()
    setup_crypto()
    setup_vol()
    setup_strike()
    setup_initial()
       
def plot_graphs(ta,strt, step=0):
    global x_vals, lgn_plot_data,jmp_diff_data, his_plot_data, ax1, ax2, HS_plot_data,RH_plot_data, toolbar
    
    if step == 0:
        if toolbar is not None:
            toolbar.pack_forget()
        def custom_format_coord(x, y):
            date_str = mdates.num2date(x).strftime('%Y-%m-%d')
            return f"Date = {date_str}, Price = ${y:.2f}"
        ax1.format_coord = custom_format_coord
        ax2.format_coord = custom_format_coord
    
    if step >= ta:
        print("done")
        toolbar.pack()
        toolbar.update()
        #enable_all_widgets()
        show_stats()
        current_case.show()
        return

    LGN = current_case.step_lognormal(step)
    HIS = current_case.step_historical(step)
    JUD = 0
    HES = 0
    RHS = 0
    
    if(JD_bool.get()):
        JUD = current_case.step_jump_diff(step)
        jmp_diff_data.append(JUD)
    
    if(HS_bool.get()):
        HES = current_case.step_Heston(step)
        HS_plot_data.append(HES)
    
    if(RH_bool.get()):
        RHS = current_case.step_R_heston(step)
        RH_plot_data.append(RHS)
    
    inc = pd.to_datetime(strt) + pd.Timedelta(days=step)
   
    print(f"LGN is: {LGN} and HES is: {HES} and RHS is: {RHS} and JUD is {JUD} in {inc}")
    
    lgn_plot_data.append(LGN)
    x_vals.append(inc)
    Strike_vals.append(K)
    his_plot_data.append(HIS)
    
    ax1.clear()
    ax1.plot(x_vals, lgn_plot_data, label="Lognormal")
    ax1.plot(x_vals,Strike_vals,label="strike",color="red",linestyle='dotted')
    if (JD_bool.get()):
        ax1.plot(x_vals, jmp_diff_data, label="Jump Diffusion",color = "pink")
    if(HS_bool.get()):
         ax1.plot(x_vals, HS_plot_data, label="Heston Model",color = "green")
    if(RH_bool.get()):
         ax1.plot(x_vals, RH_plot_data, label="R-Heston Model",color = "purple")
           
    relevant_data = lgn_plot_data \
            + (jmp_diff_data if JD_bool.get() else []) \
            + (HS_plot_data if HS_bool.get() else []) \
            + (RH_plot_data if RH_bool.get() else [])

    relevant_data = [val for val in relevant_data if val is not None and not np.isnan(val)]

    ax1.ticklabel_format(style='plain', axis='y')
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(float(x))))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(float(x))))

    if relevant_data:
        ymin = np.min(relevant_data)
        ymax = np.max(relevant_data)
        if ymax == ymin:
            margin = 1
        else:
            margin = (ymax - ymin) * 0.1
        ax1.set_ylim(ymin - margin, ymax + margin)
 
    ax1.legend()
    ax1.tick_params(axis='x', rotation=90, labelsize=6)

    ax2.clear()
    ax2.plot(x_vals, his_plot_data, label="Historical", color="orange")
    #ax2.set_title("Historical Path")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=90, labelsize=6)
    
    ax1.grid(True)
    ax2.grid(True)
    
    canvas.draw()
    root.after(30, lambda: plot_graphs(ta, strt, step + 1))
     
def show_loading():
    global loading_overlay, frames  # Need to keep reference to frames

    loading_overlay = tk.Toplevel(root)
    loading_overlay.overrideredirect(True)
    loading_overlay.attributes("-alpha", 0.8)
    loading_overlay.configure(bg="#000000")
    loading_overlay.grab_set()

    root_x = root.winfo_rootx()
    root_y = root.winfo_rooty()
    root_w = root.winfo_width()
    root_h = root.winfo_height()
    loading_overlay.geometry(f"{root_w}x{root_h}+{root_x}+{root_y}")

    # Create a frame with black background
    frame = tk.Frame(loading_overlay, bg="#000000", bd=0)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    # Add text label
    text_label = tk.Label(frame, text="Fetching Parameters...\nSetting Up Monte Carlo", 
                         font=("Helvetica", 16), bg="#000000", fg="white")
    text_label.pack(pady=(10, 5))

    # Load and process GIF frames
    gif  = Image.open(Path(__file__).parent / "Loading3.gif")


    
    # Process each frame to composite with black background
    frames = []
    for frame_img in ImageSequence.Iterator(gif):
        # Convert to RGBA if not already
        if frame_img.mode != 'RGBA':
            frame_img = frame_img.convert('RGBA')
        
        # Create new image with black background
        new_frame = Image.new('RGBA', frame_img.size, "#000000")
        # Paste the frame with transparency
        new_frame.paste(frame_img, (0, 0), frame_img)
        frames.append(ImageTk.PhotoImage(new_frame))

    # Create label for animation
    gif_label = tk.Label(frame, bg="#000000", bd=0)
    gif_label.pack(pady=(0, 15))

    # Animation function
    def animate_gif(index=0):
        gif_label.config(image=frames[index])
        loading_overlay.after(30, animate_gif, (index + 1) % len(frames))

    animate_gif()
    track_loading_position()
    
def track_loading_position():
    if loading_overlay is None:
        return

    root_x = root.winfo_rootx()
    root_y = root.winfo_rooty()
    root_w = root.winfo_width()
    root_h = root.winfo_height()
    loading_overlay.geometry(f"{root_w}x{root_h}+{root_x}+{root_y}")
    root.after(30, track_loading_position)


def hide_loading():
    global loading_overlay
    if loading_overlay:
        loading_overlay.destroy()
        root.attributes("-disabled", False)
        loading_overlay = None


def show_stats():
    global result_frame
    result_window = Toplevel()
    result_window.title("Results")
    result_window.resizable(True, True)
    result_window.protocol("WM_DELETE_WINDOW", lambda: [enable_all_widgets(), result_window.destroy()])
    result_frame = tk.Frame(result_window)
    result_frame.pack(fill="both", expand=True)
    initialize_result_frame()
    if simulation_start_time is not None:
        last_runtime = time.time() - simulation_start_time
        print(f"End-to-end runtime: {last_runtime:.2f} seconds")
    
def initialize_result_frame():
    global result_frame, selected_greek, selected_method, greek_res

    result_frame.grid_columnconfigure(0, weight=1, minsize=200)
    result_frame.grid_columnconfigure(1, weight=1, minsize=200)
    result_frame.grid_columnconfigure(2, weight=1, minsize=200)
    result_frame.grid_columnconfigure(3, weight=1, minsize=200)

    main_header_style = {"font": ("Arial", 14, "bold"), "fg": "#2c3e50", "pady": 10}
    section_header_style = {"font": ("Arial", 12, "bold"), "fg": "#3498db", "pady": 5}
    label_style = {"font": ("Arial", 10), "anchor": "w", "fg": "#34495e"}
    value_style = {"font": ("Arial", 10), "anchor": "w", "fg": "#16a085"}
    price_style = {"font": ("Arial", 10, "bold"), "anchor": "w", "fg": "#e74c3c"}
    highlight_style = {"font": ("Arial", 10, "bold"), "fg": "#e74c3c"}

    tk.Label(result_frame, text="Option Pricing Results", **main_header_style).grid(row=0, column=0, columnspan=4, sticky="w", padx=10)
    tk.Label(result_frame, text="Contract Parameters", **section_header_style).grid(row=1, column=0, sticky="w", padx=10, pady=5, columnspan=2)
    tk.Label(result_frame, text=f"Underlying: {current_case.sym}", **highlight_style).grid(row=2, column=0, sticky="w", padx=10, pady=2, columnspan=2)

    contract_data = [
        ("Initial Price (S₀)", f"{current_case.S0} $"),
        ("Strike Price (K)", f"{current_case.K} $"),
        ("Time to Expiry", f"{current_case.ta} days"),
        ("Volatility (σ)", f"{current_case.sig:.4f}"),
        ("Interest Rate (r)", f"{current_case.IR:.4f}"),
        ("Start Date", current_case.beg_date),
        ("End Date", current_case.fin_date)
    ]

    for i, (label, value) in enumerate(contract_data, start=3):
        tk.Label(result_frame, text=label, **label_style).grid(row=i, column=0, sticky="w", padx=10, pady=1)
        tk.Label(result_frame, text=value, **value_style).grid(row=i, column=1, sticky="w", padx=10, pady=1)

    right_side_row = 1
    tk.Label(result_frame, text="Black-Scholes Model", **section_header_style).grid(row=right_side_row, column=2, sticky="w", padx=10, pady=5, columnspan=2)
    right_side_row += 1

    bls_call_price, bls_put_price = current_case.price_option("BLS")
    tk.Label(result_frame, text="Call Price:", **label_style).grid(row=right_side_row, column=2, sticky="w", padx=10)
    tk.Label(result_frame, text=f"{bls_call_price:.4f} $", **price_style).grid(row=right_side_row, column=3, sticky="w", padx=10)
    right_side_row += 1
    tk.Label(result_frame, text="Put Price:", **label_style).grid(row=right_side_row, column=2, sticky="w", padx=10)
    tk.Label(result_frame, text=f"{bls_put_price:.4f} $", **price_style).grid(row=right_side_row, column=3, sticky="w", padx=10)
    right_side_row += 1

    tk.Label(result_frame, text="Greeks", **section_header_style).grid(row=right_side_row, column=2, sticky="w", padx=10, pady=5, columnspan=2)
    right_side_row += 1

    bls_call_delta, bls_put_delta = current_case.PW_delta()
    bls_call_rho, bls_put_rho = current_case.PW_rho()
    bls_call_vega, bls_put_vega = current_case.PW_vega()
    bls_call_theta, bls_put_theta = current_case.PW_Theta()
    greeks = [
        ("Call Delta (Δ)", bls_call_delta),
        ("Put Delta (Δ)", bls_put_delta),
        ("Call Rho (ρ)", bls_call_rho),
        ("Put Rho (ρ)", bls_put_rho),
        ("Call Theta (θ)", bls_call_theta),
        ("Put Theta (θ)", bls_put_theta),
        ("Vega (𝓥)", bls_call_vega)
    ]
    for greek, val in greeks:
        tk.Label(result_frame, text=greek, **label_style).grid(row=right_side_row, column=2, sticky="w", padx=10, pady=1)
        tk.Label(result_frame, text=f"{val:.6f}", **value_style).grid(row=right_side_row, column=3, sticky="w", padx=10, pady=1)
        right_side_row += 1

    if HS_bool.get():
        tk.Label(result_frame, text="Heston Model", **section_header_style).grid(row=right_side_row, column=2, sticky="w", padx=10, pady=5, columnspan=2)
        right_side_row += 1

        HS_call, HS_put = current_case.price_option("HES")
        tk.Label(result_frame, text="Call Price:", **label_style).grid(row=right_side_row, column=2, sticky="w", padx=10)
        tk.Label(result_frame, text=f"{HS_call:.4f} $", **price_style).grid(row=right_side_row, column=3, sticky="w", padx=10)
        right_side_row += 1
        tk.Label(result_frame, text="Put Price:", **label_style).grid(row=right_side_row, column=2, sticky="w", padx=10)
        tk.Label(result_frame, text=f"{HS_put:.4f} $", **price_style).grid(row=right_side_row, column=3, sticky="w", padx=10)
        right_side_row += 1

        hs_params = [
            ("Vol-of-Vol", current_case.vov),
            ("Kappa", current_case.kappa),
            ("Theta", current_case.theta),
            ("Rho", current_case.rho)
        ]
        for param, val in hs_params:
            tk.Label(result_frame, text=param, **label_style).grid(row=right_side_row, column=2, sticky="w", padx=10, pady=1)
            tk.Label(result_frame, text=str(val), **value_style).grid(row=right_side_row, column=3, sticky="w", padx=10, pady=1)
            right_side_row += 1

    if JD_bool.get():
        row_offset = len(contract_data) + 3
        JD_call_price, JD_put_price = current_case.price_option("JUD")
        tk.Label(result_frame, text="Jump Diffusion Model", **section_header_style).grid(row=row_offset, column=0, sticky="w", padx=10, pady=5, columnspan=2)
        tk.Label(result_frame, text="Call Price:", **label_style).grid(row=row_offset+1, column=0, sticky="w", padx=10)
        tk.Label(result_frame, text=f"{JD_call_price:.4f} $", **price_style).grid(row=row_offset+1, column=1, sticky="w", padx=10)
        tk.Label(result_frame, text="Put Price:", **label_style).grid(row=row_offset+2, column=0, sticky="w", padx=10)
        tk.Label(result_frame, text=f"{JD_put_price:.4f} $", **price_style).grid(row=row_offset+2, column=1, sticky="w", padx=10)
        jd_params = [
            ("Jump Frequency (λ)", current_case.poisson_rate),
            ("Jump Std Dev (δ)", current_case.std_jump),
            ("Mean Jump (μJ)", current_case.mean_jump)
        ]
        for i, (param, val) in enumerate(jd_params, start=row_offset+3):
            tk.Label(result_frame, text=param, **label_style).grid(row=i, column=0, sticky="w", padx=10, pady=1)
            tk.Label(result_frame, text=str(val), **value_style).grid(row=i, column=1, sticky="w", padx=10, pady=1)

    if RH_bool.get():
        base_row = len(contract_data) + 3
        row_offset = base_row + 8 if JD_bool.get() else base_row
        RHS_call, RHS_put = current_case.price_option("RHS")
        tk.Label(result_frame, text="Rough Heston Model", **section_header_style).grid(row=row_offset, column=0, sticky="w", padx=10, pady=5, columnspan=2)
        tk.Label(result_frame, text="Call Price:", **label_style).grid(row=row_offset+1, column=0, sticky="w", padx=10)
        tk.Label(result_frame, text=f"{RHS_call:.4f} $", **price_style).grid(row=row_offset+1, column=1, sticky="w", padx=10)
        tk.Label(result_frame, text="Put Price:", **label_style).grid(row=row_offset+2, column=0, sticky="w", padx=10)
        tk.Label(result_frame, text=f"{RHS_put:.4f} $", **price_style).grid(row=row_offset+2, column=1, sticky="w", padx=10)
        rhs_params = [
            ("Vol-of-Vol", current_case.RH_vov),
            ("Kappa", current_case.RH_kappa),
            ("Theta", current_case.RH_theta),
            ("Rho", current_case.RH_rho),
            ("Hurst", current_case.RH_hurst)
        ]
        for i, (param, val) in enumerate(rhs_params, start=row_offset+3):
            tk.Label(result_frame, text=param, **label_style).grid(row=i, column=0, sticky="w", padx=10, pady=1)
            tk.Label(result_frame, text=str(val), **value_style).grid(row=i, column=1, sticky="w", padx=10, pady=1)

    if JD_bool.get() or RH_bool.get() or HS_bool.get():
        greek_row = right_side_row + 1
        tk.Label(result_frame, text="Calculate Specific Greeks:", font=("Arial", 10, "bold")).grid(row=greek_row, column=2, sticky="w", padx=10, pady=(15,5), columnspan=2)
        tk.Label(result_frame, text="Greek:", **label_style).grid(row=greek_row+1, column=2, sticky="w", padx=10)
        selected_greek = tk.StringVar(value="Delta (Δ)")
        greek_dropdown = tk.OptionMenu(result_frame, selected_greek, "Delta (Δ)", "Rho (ρ)", "Theta (θ)", "Vega (𝓥)")
        greek_dropdown.config(width=15)
        greek_dropdown.grid(row=greek_row+1, column=3, sticky="w", padx=10, pady=2)
        tk.Label(result_frame, text="Model:", **label_style).grid(row=greek_row+2, column=2, sticky="w", padx=10)
        selected_method = tk.StringVar()
        methods = []
        if JD_bool.get(): methods.append("Mertons JD")
        if HS_bool.get(): methods.append("Heston Vol") 
        if RH_bool.get(): methods.append("Rough Heston")
        if methods:
            selected_method.set(methods[0])
            calc_method_dropdown = tk.OptionMenu(result_frame, selected_method, *methods)
            calc_method_dropdown.config(width=15)
            calc_method_dropdown.grid(row=greek_row+2, column=3, sticky="w", padx=10, pady=2)
        calc_button = tk.Button(result_frame, text="Calculate", command=calculate_greek, width=25)
        calc_button.grid(row=greek_row+3, column=2,columnspan=2, pady=5, padx=10, sticky="n")
        
        greek_res=tk.Label(result_frame, text="", width=35,font=("Arial", 13))
        greek_res.grid(row=greek_row+4, column=2,columnspan=2, pady=5,  sticky="n")
        
                
def calculate_greek():
    global alt_copy1, alt_copy2,alt_copy_main, alt_copy1_lbl, alt_copy2_lbl, alt_main_lbl, differencing, greek_gap

    alt_copy1 = copy.deepcopy(current_case)
    alt_copy2 = None
    alt_copy_main = copy.deepcopy(current_case)
    
    match selected_greek.get():
        case "Delta (Δ)":
            h, differencing = compute_delta_h()
            print("changed delta")
            if(differencing == "central"):
                alt_copy2 = copy.deepcopy(current_case)
                alt_copy1.S0 += h
                alt_copy2.S0 -= h
                print(f"First S0:{alt_copy1.S0}")
                print(f"Second S0:{alt_copy2.S0}")
                print(f"The main S0:{alt_copy_main.S0}")
                alt_copy1_lbl = "S0 + ΔS"
                alt_main_lbl = "S0"
                alt_copy2_lbl = "S0 - ΔS"
                plot_simulation_paths(alt_copy1,alt_copy_main,alt_copy2)
            else:
                alt_copy1.S0 += h
                alt_copy1_lbl = "S0 + ΔS"
                alt_main_lbl = "S0"
                plot_simulation_paths(alt_copy1,alt_copy_main)
                
        case "Rho (ρ)":
            h,differencing = compute_rho_h()
            print("changed rho")
            if(differencing=="central"):
                alt_copy2 = copy.deepcopy(current_case)
                alt_copy1.IR += h
                alt_copy2.IR -= h
                alt_copy1_lbl = "IR + Δ(IR)"
                alt_main_lbl = "IR"
                alt_copy2_lbl = "IR - Δ(IR)"
                plot_simulation_paths(alt_copy1,alt_copy_main,alt_copy2)
            else:
                alt_copy1.IR += h
                alt_copy1_lbl = "IR + Δ(IR)"
                alt_main_lbl = "IR"
                plot_simulation_paths(alt_copy1,alt_copy_main)
                
        case "Theta (θ)":
            h, differencing = compute_theta_h()
            print("changed theta")
            if(differencing=="central"):
                alt_copy2 = copy.deepcopy(current_case)
                
                alt_copy1.ta += h 
                alt_copy1.x_ta = alt_copy1.ta
                
                alt_copy2.ta -= h
                alt_copy2.x_ta = alt_copy1.ta
                
                alt_copy_main.x_ta = alt_copy1.ta
                
                alt_copy1_lbl = "τ + Δτ"
                alt_main_lbl = "τ"
                alt_copy2_lbl = "τ - Δτ"
                plot_simulation_paths(alt_copy1,alt_copy_main,alt_copy2)
            else:
                alt_copy1.ta += h
                alt_copy1.x_ta = alt_copy1.ta
                alt_copy_main.x_ta = alt_copy1.ta
                alt_copy1_lbl = "τ + Δτ"
                alt_main_lbl = "τ"
                plot_simulation_paths(alt_copy1,alt_copy_main)
                
        case "Vega (𝓥)":
            h,differencing = compute_vega_h()
            print("changed vol")
            if(differencing=="central"):
                alt_copy2 = copy.deepcopy(current_case)
                alt_copy1.sig += h
                alt_copy2.sig -= h
                alt_copy1_lbl = "σ + Δσ"
                alt_main_lbl = "σ"
                alt_copy2_lbl = "σ - Δσ"
                plot_simulation_paths(alt_copy1,alt_copy_main,alt_copy2)
            else:
                alt_copy1.sig += h
                alt_copy1_lbl = "σ + Δσ"
                alt_main_lbl = "σ"
                plot_simulation_paths(alt_copy1,alt_copy_main)
                         
    greek_gap = h


def plot_simulation_paths(sim1, sim_main, sim2=None, update_interval=100):
    """Plot simulation paths with date-based x-axis and improved toolbar formatting."""
    
    if sim_main.ta != sim1.ta:
        sim_main.draw_randoms()
        sim1.draw_randoms()
        print(sim_main.ta, np.shape(sim_main.draws))
        print(sim1.ta, np.shape(sim1.draws))
        if sim2:
            sim2.draw_randoms()
            print(sim2.ta, np.shape(sim2.draws))
    
    # Create plot window
    plot_window = tk.Toplevel()
    plot_window.title(f"Simulation Paths: {selected_greek.get()}")
    
    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 4))
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create toolbar but don't pack it yet
    toolbar = NavigationToolbar2Tk(canvas, plot_window)
    toolbar.pack_forget()  # Hide initially
    
    # Customize toolbar coordinates display
    def format_coord(x, y):
        try:
            # Convert matplotlib date number to datetime
            date = mdates.num2date(x).strftime('%Y-%m-%d')
            return f"Date: {date}, Price: ${y:,.2f}"
        except:
            return ""
    ax.format_coord = format_coord

    # Configure simulations
    simulations = [
        {'sim': sim1, 'path': [], 'label': alt_copy1_lbl, 'color': 'blue'},
        {'sim': sim_main, 'path': [], 'label': alt_main_lbl, 'color': 'green'}
    ]
    if sim2:
        simulations.append({'sim': sim2, 'path': [], 'label': alt_copy2_lbl, 'color': 'red'})

    # Format y-axis as dollars
    ax.yaxis.set_major_formatter('${x:,.2f}')

    # Get start date
    start_date = pd.to_datetime(current_case.beg_date)
    
    # Add timestamp in bottom right
    timestamp = ax.annotate('', 
                          xy=(0.98, 0.02), 
                          xycoords='axes fraction',
                          ha='right',
                          va='bottom',
                          fontsize=8,
                          color='gray')

    method = selected_method.get()

    def safe_step_func(sim, step):
        try:
            if method == "Mertons JD":
                if step >= len(sim.draws):
                    return sim.path[-1][1] if sim.path else sim.S0
                return sim.step_jump_diff(step)
            elif method == "Heston Vol":
                return sim.step_Heston(step)
            elif method == "Rough Heston":
                return sim.step_R_heston(step)
        except Exception as e:
            print(f"Error in step {step}: {str(e)}")
            return sim.path[-1][1] if sim.path else sim.S0

    def update_plot(step):
        nonlocal simulations, toolbar
        
        continue_simulation = False
        
        for sim_data in simulations:
            try:
                if step < sim_data['sim'].ta:
                    price = safe_step_func(sim_data['sim'], step)
                    sim_data['path'].append((step, price))
                    continue_simulation = True
            except Exception as e:
                print(f"Critical error in {sim_data['label']}: {str(e)}")
                sim_data['path'].append((step, sim_data['path'][-1][1] if sim_data['path'] else float('nan')))
        
        ax.clear()
        
        # Update timestamp
        now_time = pd.to_datetime(current_case.beg_date).strftime('%Y-%m-%d %H:%M:%S')
        timestamp.set_text(f"Last update: {now_time}")
        
        # Plot with date-based x-axis
        for sim_data in simulations:
            if sim_data['path']:
                days, prices = zip(*sim_data['path'])
                dates = [start_date + pd.Timedelta(days=day) for day in days]
                ax.plot(dates, prices, label=sim_data['label'], color=sim_data['color'])
        
        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.yaxis.set_major_formatter('${x:,.2f}')
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()  # Rotate dates for better visibility
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        canvas.draw()

        if continue_simulation:
            plot_window.after(update_interval, update_plot, step + 1)
        else:
            # Simulation complete - show toolbar
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            print("All simulations completed")
            
            
            measure_greek()
            
            # Add completion marker
            ax.annotate('Simulation Complete', 
                       xy=(0.5, 0.95), 
                       xycoords='axes fraction',
                       ha='center', 
                       bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
            canvas.draw()

    update_plot(0)
    
    # def on_closing():
    #     plt.close(fig)
    #     plot_window.destroy()
    
    # plot_window.protocol("WM_DELETE_WINDOW", on_closing)

    #return plot_window

def measure_greek():
    global greek_call, greek_put
    method = selected_method.get()
    if method == "Mertons JD":
        opt = "JUD"
    elif method == "Heston Vol":
        opt = "HES"
    elif method == "Rough Heston":
        opt = "RHS"
        
    if(differencing == "central"):
        call_high, put_high = alt_copy1.price_option(opt)
        call_mid, put_mid = alt_copy_main.price_option(opt)
        call_low, put_low = alt_copy2.price_option(opt)
        greek_call = (call_high-call_low)/(2*greek_gap)
        greek_put = (put_high-put_low)/(2*greek_gap)
    else:
        call_high, put_high = alt_copy1.price_option(opt)
        call_mid, put_mid = alt_copy_main.price_option(opt)
        greek_call = (call_high-call_mid)/(greek_gap)
        greek_put = (put_high-put_mid)/(greek_gap)
       
    
    if selected_greek.get() == "Rho (ρ)":
        print("gotchya")
        greek_call = greek_call/100
        greek_put = greek_put/100
        greek_res.config(text=f"ρ Call: {greek_call:.2f}, ρ Put: {greek_put:.2f}",fg="green")
    elif selected_greek.get() == "Theta (θ)":
        greek_call = (-greek_call/current_case.ta)*365
        greek_put = (-greek_put/current_case.ta)*365
        greek_res.config(text=f"θ Call: {greek_call:.2f}, θ Put: {greek_put:.2f} ",fg="green")
    elif selected_greek.get() == "Delta (Δ)":
        greek_res.config(text=f"Δ Call: {greek_call:.2f}, Δ Put: {greek_put:.2f} ",fg="green")
    else:
        greek_res.config(text=f"𝓥 : {(greek_call+greek_put)/2}",fg="green")
        
    print(greek_call,greek_put)  
    #if temp == "Vega (𝓥)":
           

                
def compute_delta_h():
    initial_val = current_case.S0
    if (initial_val<0.01):
        h = 0.01 * initial_val
        kind = "forward"
    elif ( initial_val<1):
        h = 0.005 * initial_val
        kind = "central"
    elif (initial_val<100):
        h = 0.003 * initial_val
        kind = "central"
    elif (initial_val<10000):
        h = 0.001 * initial_val
        kind = "central"
    elif (initial_val>=10000):
        h = 0.0005*initial_val
        kind = "central"
        
    return h,kind

def compute_theta_h():
    duration = current_case.ta
    if(duration<30):
        h = 1
        kind = "forward"
        #central diff for over 7 and under 7 forward diff
    elif(duration<182):
        h = 2
        kind = "central"
    else:
        h = 5
        kind = "central"
        
    return h, kind

def compute_rho_h():
    interest = current_case.IR
    if(interest<0.001):
        h = 0.0005
        kind = "forward"
    elif(interest<0.01):
        h = 0.001
        kind = "central"
    elif(interest<0.1):
        h = 0.01
        kind = "central"
    else:
        h = 0.02
        kind = "central"
        
    return h, kind

def compute_vega_h():
    vol_sig = current_case.sig
    if vol_sig > 1.5:
        h = 0.05
        kind = "central"
    elif vol_sig > 0.8:
        h = 0.03
        kind = "central"
    elif vol_sig > 0.3:
        h = 0.02
        kind = "central"
    else:
        h = 0.01
        kind = "forward"
    
    return h,kind
        

# Start the GUI
if __name__ == "__main__":
    create_form()
    root.mainloop()










