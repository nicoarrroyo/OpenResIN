import sys
import time
import threading
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import os
from collections import defaultdict
import nalira_config as c

def table_print(**kwargs):
    """
    The name of the variable and the value is outputted together. 
    An orderly way of outputting variables at the start of a program. 
    
    Parameters
    ----------
    **kwargs : any
        Any number of any type of variable is passed and outputted. 
    
    Returns
    -------
    None.
    
    """
    if not kwargs:
        print("No data to display.")
        return
    
    # Compute max lengths
    max_var_length = max(map(len, kwargs.keys()), default=8)
    max_value_length = max(map(lambda v: len(str(v)), kwargs.values()), default=5)
    
    # Format the header and separator dynamically
    header = f"| {'Variable'.ljust(max_var_length)} | {'Value'.ljust(max_value_length)} |"
    separator = "-" * len(header)
    
    # Print table
    print(separator)
    print(header)
    print(separator)
    for key, value in kwargs.items():
        print(f"| {key.ljust(max_var_length)} | {str(value).ljust(max_value_length)} |")
    print(separator)

def list_folders(folders_path):
    """
    This functions has a bit of a deceiving name because it does more than just 
    list the folders in a directory. It also finds all the folders in the 
    directory that are relevant to sentinel 2 satellite imagery, separates 
    them by year, tile, and month, handles the quantity of folders that need 
    to be process, and WIP

    Parameters
    ----------
    folders_path : TYPE
        DESCRIPTION.

    Returns
    -------
    filtered_folders : TYPE
        DESCRIPTION.

    """
    folders = []
    filtered_folders = []
    possible_years = []
    possible_tiles = []
    possible_months = []
    
    folders = os.listdir(folders_path)
    
    if len(folders) == 0:
        print("found 0 items in searched directory")
        sys.exit(1)
    
    if c.N_IMAGES < -1 or not isinstance(c.N_IMAGES, int):
        print(f"WARNING: N_IMAGES has a bad value: {c.N_IMAGES}")
        print("Check config_NALIRA file to fix")
        sys.exit(1)
    elif c.N_IMAGES == -1:
        n_images = len(folders)
    else:
        n_images = c.N_IMAGES
    
    for folder in folders:
        if len(folder) > 10 and ".SAFE" in folder[5:]:
            filtered_folders.append(folder)
    filtered_folders = filtered_folders[:n_images]
    
    image_info = defaultdict(list)
    for folder in filtered_folders:
        parts = folder.split("_")
        if len(parts) == 7:
            sentinel_name = parts[0]
            datatake_start_sensing_time = parts[2]
            tile_number_field = parts[5]
            year = datatake_start_sensing_time[:4]
            month = datatake_start_sensing_time[4:6]
            image_info[year].append((sentinel_name, tile_number_field, month))
    
    for year in sorted(image_info.keys()):
        print(f"{year}")
        possible_years.append(year)
        for satellite, tile, month in image_info[year]:
            possible_tiles.append(tile)
            possible_months.append(month)
            print(f"satellite {satellite}, tile {tile}, month {month}\n")
    
    if len(set(possible_tiles)) != 1:
        print("WARNING: there is more than one type of tile in the data "
              "folder. This can cause problems with image compositing as the images "
              "are unlikely to overlap perfectly. To fix this, it is "
              "recommended to move all the unintended folders into a storage "
              "folder away from the searched data directory.")
        sys.exit(1)
    
    return filtered_folders

def failure(failure, solution, error):
    print(f"FAILURE: {failure} due to {error}")
    print(f"Trying {solution}")

def confirm_continue_or_exit():
    """
    Asks the user if they want to continue with the program.
    
    If the user enters 'y' or 'yes', the function returns and the script 
    continues. If the user enters 'n' or 'no', the function prints a message 
    and exits the script. It will keep asking until a valid 
    input ('y', 'yes', 'n', 'no') is given.
    """
    while True: # Loop until valid input is received
        response = input("Do you want to continue? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("off we go - continuing program")
            return # Exit the function and let the main script proceed
        
        elif response in ['n', 'no']:
            print("will not continue - exiting program")
            sys.exit() # Stop the script immediately
        else:
            # Ask again if the input was invalid
            print("invalid input. 'y' for yes or 'n' for no.")

def spinner(stop_event, message):
    """
    A simple spinner that runs until stop_event is set.
    The spinner updates the ellipsis by overwriting the same line.
    """
    chase = ["   ", ".  ", ".. ", "...", " ..", "  .", "   ", 
             "  .", " ..", "...", ".. ", ".  "]
    wobble = [" \ ", " | ", " / ", " | "]
    woosh = ["|   ", ")   ", " )  ", "  ) ", "   )", "   |", 
             "   |", "  ( ", " (  ", "(   ", "|   "]
    ellipses = ["   ", ".  ", ".. ", "..."]
    dude = [" :D ", " :) ", " :\ ", " :( ", " :\ ", " :) "]
    
    frames = chase
    frames = wobble
    frames = woosh
    frames = ellipses
    frames = dude
    
    frames = wobble
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write("\r" + message + frame)
        sys.stdout.flush()
        time.sleep(0.2)
        i += 1
    # Clear the spinner message on stop
    sys.stdout.write("\r" + message + "... complete! \n")
    sys.stdout.flush()

def start_spinner(message):
    # Create an event for signaling the spinner to stop
    stop_event = threading.Event()
    
    # Use a thread to run the spinner concurrently
    thread = threading.Thread(target=spinner, args=(stop_event, message))
    thread.start()
    return stop_event, thread

def end_spinner(stop_event, thread):
    # Turn off the spinner once processing is done
    stop_event.set()
    thread.join()

def prompt_roi(image_array, n):
    """
    Opens a Tkinter window displaying the image (as a numpy array).
    Allows the user to select multiple ROIs by click-and-drag.
    The ROI is automatically saved upon mouse release.
    When done, click the final button to close the window and return a 
    list of ROI coordinates.
    
    Parameters
    ----------
    image_array : numpy array
        A numpy array converted from an image. 
    n : int
        An integer representing the number regions of interest (ROIs) that 
        were identified by the user. 
    
    Returns
    -------
    rois : list
        List of floats (upper-left x-coordinate (ulx), 
                        upper-left y-coordinate (uly), 
                        lower-right x-coordinate (lrx), 
                        lower-right y-coordinate (lry)) for each ROI. 
    
    """
    # Convert the numpy array to a PIL image
    image = Image.fromarray(image_array)
    image = image.resize((500, 500))  # Resize the image to fit in the window
    width, height = image.size
    
    rois = []          # List to store confirmed ROI coordinates
    rects = []         # List to store the drawn rectangles
    current_roi = None # To temporarily hold the current ROI coordinates
    current_rect = None
    start_x = start_y = 0
    
    error_counter = 0
    while error_counter < 2:
        try:
            # Create the Tkinter window and canvas
            root = tk.Tk()
            root.title("Select Regions of Interest (ROIs)")
            root.resizable(False, False)
            canvas = tk.Canvas(root, width=width, height=height)
            canvas.pack()
    
            # Display the image on the canvas
            tk_image = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor="nw", image=tk_image)
            break
        except:
            error_counter += 1
            root = tk.Toplevel()
            root.title("CLOSE THIS WINDOW")
            canvas = tk.Canvas(root, width=width, height=height)
            canvas.pack()
            root.destroy()
            print("Please close any windows that were opened")
            root.mainloop()
    if error_counter >= 2:
        print("Broken prompt_roi function")
        return
    
    # Create the lines for following the cursor
    vertical_line = canvas.create_line(0, 0, 0, height, fill="red", dash=(4, 2))
    horizontal_line = canvas.create_line(0, 0, width, 0, fill="red", dash=(4, 2))
    
    # Helper function to update the status bar message.
    def set_status(msg):
        status_label.config(text=msg)
    
    # Event handlers for drawing the ROI rectangle
    def on_button_press(event):
        nonlocal start_x, start_y, current_rect, current_roi
        start_x, start_y = event.x, event.y
        if current_rect is not None:
            canvas.delete(current_rect)  # Delete the previous rectangle if exists
        # Start drawing a rectangle
        current_rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, 
                                               outline="red", width=2)
        current_roi = None  # Reset current ROI
    
    def on_move_press(event):
        nonlocal current_rect
        # Update the rectangle as the mouse is dragged
        canvas.coords(current_rect, start_x, start_y, event.x, event.y)
        # Update the follower lines by resetting their coordinates
        canvas.coords(vertical_line, event.x, 0, event.x, height)
        canvas.coords(horizontal_line, 0, event.y, width, event.y)
    
    def on_button_release(event):
        nonlocal current_roi
        # Finalize the ROI on mouse release
        end_x, end_y = event.x, event.y
        ulx = min(start_x, end_x)
        uly = min(start_y, end_y)
        lrx = max(end_x, start_x)
        lry = max(end_y, start_y)
        current_roi = (ulx, uly, lrx, lry)
        # Auto-save the ROI upon release
        save_roi()
    
    # Event handler for mouse motion (unpressed button) to update the lines
    def on_mouse_motion(event):
        canvas.coords(vertical_line, event.x, 0, event.x, height)
        canvas.coords(horizontal_line, 0, event.y, width, event.y)
    
    # Function to save the current ROI automatically
    def save_roi():
        nonlocal rois, current_roi, rects, current_rect
        if len(rois) < n:
            if current_roi is not None:
                # Ensure the ROI coordinates are within bounds
                current_roi = [max(0, int(roi)) for roi in current_roi]
                current_roi = [min(width, int(roi)) for roi in current_roi]
                
                rois.append(current_roi)
                rects.append(current_rect)  # Keep track of the rectangle reference
                canvas.itemconfig(current_rect, outline="green")
                
                current_roi_converted = np.array(current_roi) * len(image_array) / width
                set_status((f"Saved ROI {current_roi_converted}. {n-len(rois)} left"))
                # Reset current selection variables for the next ROI
                current_roi = None
                current_rect = None
            else:
                set_status("No region of interest selected")
        else:
            set_status(f"Too many selections, expected: {n}. Overwrite a selection.")
    
    # Button callback to finish the ROI selection and close the window
    def finish():
        nonlocal rois
        # If there's a ROI in progress, try to save it.
        if current_roi is not None:
            save_roi()
        if len(rois) < n:
            set_status(f"{n - len(rois)} selection(s) remaining")
        else:
            root.destroy()
    
    def overwrite():
        nonlocal rois, current_roi, current_rect, rects
        # Remove the current drawing if any.
        canvas.delete(current_rect)
        if rois and rects:
            # Remove the last saved ROI and rectangle
            canvas.delete(rects[-1])
            rois.pop()  # Remove the last ROI coordinates
            rects.pop()  # Remove the last rectangle reference
            current_rect = None  # Reset the current rectangle reference
            current_roi = None  # Reset the current ROI coordinates
            set_status("Overwritten ROI")
        else:
            set_status("No regions of interest saved")
    
    def select_all():
        nonlocal current_roi
        current_roi = [0, 0, width, height]
        save_roi()
    
    def hide_cursor(event):
        canvas.config(cursor="none")
    
    def show_cursor(event):
        canvas.config(cursor="")
    # Bind mouse events to the canvas
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    canvas.bind("<Motion>", on_mouse_motion)
    canvas.config(cursor="none")
    canvas.bind("<Enter>", lambda event: canvas.config(cursor="none"))
    canvas.bind("<Leave>", lambda event: canvas.config(cursor=""))
    
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, pady=10)
    
    overwrite_button = tk.Button(button_frame, text="Overwrite", command=overwrite)
    overwrite_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    all_button = tk.Button(button_frame, text="Select Entire Frame", command=select_all)
    all_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    finish_button = tk.Button(button_frame, text="Finish", command=finish)
    finish_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    # Create the status bar below the buttons
    status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X, padx=2, pady=2)
    
    root.mainloop()
    rois_converted = np.array(rois) * len(image_array) / width
    return rois_converted