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

def alert_user(warning, consequence, solution, n_errors=None):
    """
    Alert the user about a possible problem, explain it, and offer a solution.
    
    Provide the user with a warning that something has gone / could go wrong, 
    explain the consequences of the warning, and offer a solution. There is 
    also the option to include a counter for number of errors. Should this not 
    be passed, the number of errors is not included in the output. 

    Parameters
    ----------
    warning : str
        Outline what has gone / could go wrong.
    consequence : str
        Sentence explaining the consequence of the error / warning.
    solution : str
        Suggested solution to the error.
    n_errors : integer, optional
        Number of errors found so far.

    Returns
    -------
    n_errors + 1
        Increments the number of errors found. If n_errors is not given, 
        nothing is returned

    """
    if n_errors:
        print(f"\n | {n_errors}. WARNING: {warning}")
    else:
        print(f"\n | WARNING: {warning}")
    print(f" | CONSEQUENCE: {consequence}")
    print(f" | SOLUTION: {solution}")
    
    if n_errors: return n_errors + 1

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

def prompt_roi(image_array, n, min_vertices=3):
    """
    Opens a Tkinter window displaying the image (as a numpy array).
    Allows the user to select multiple ROIs by clicking out the vertices of
    a polygon (a simple click-and-drag still works fine for a four-cornered
    "rectangle", but any shape with three or more vertices is supported).
    A shape is closed either by clicking back near its first vertex, by
    pressing Enter, or via the "Close Shape" button, and is then
    automatically saved. When done, click "Finish" to close the window and
    return the list of ROI coordinates.
    
    Parameters
    ----------
    image_array : numpy array
        A numpy array converted from an image. 
    n : int
        An integer representing the number regions of interest (ROIs) that 
        were identified by the user. 
    min_vertices : int, optional
        The minimum number of vertices required to close a shape. Defaults 
        to 3 (a triangle), the smallest possible polygon. Pass 2 to restore 
        the old click-drag rectangle-only behaviour (two opposite corners). 
    
    Returns
    -------
    rois : list
        A list with one entry per ROI. Each entry is a flat list of floats
        [x1, y1, x2, y2, ..., xk, yk] representing the (x, y) coordinates of
        every vertex of that ROI's polygon, in the original image_array's
        coordinate space. A plain rectangle is simply a 4-vertex polygon, so
        existing rectangle-only consumers of this data keep working
        unchanged (e.g. taking the centroid of the points still gives the
        rectangle's centre). 
    
    """
    # Convert the numpy array to a PIL image
    image = Image.fromarray(image_array)
    image = image.resize((500, 500))  # Resize the image to fit in the window
    width, height = image.size
    
    CLOSE_RADIUS = 8 # pixels; clicking this close to the first vertex closes the shape
    
    rois = []             # List to store confirmed ROI vertex lists (canvas space)
    shapes = []            # List to store the finalised polygon canvas items
    current_points = []    # Vertices of the shape currently being drawn (canvas space)
    vertex_markers = []    # Canvas item ids for the in-progress vertex dots
    edge_lines = []        # Canvas item ids for the in-progress connecting edges
    preview_line = None    # Canvas item id for the rubber-band line to the cursor
    
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
    
    def points_status():
        n_pts = len(current_points)
        if n_pts == 0:
            return "Click to start a shape."
        return (f"{n_pts} point(s) placed. Click near the first point, "
               "press Enter, or click 'Close Shape' to finish "
               f"(min {min_vertices}).")
    
    # Remove all temporary drawing for the shape currently in progress
    def clear_current_drawing():
        nonlocal preview_line
        for marker in vertex_markers:
            canvas.delete(marker)
        for line in edge_lines:
            canvas.delete(line)
        vertex_markers.clear()
        edge_lines.clear()
        if preview_line is not None:
            canvas.delete(preview_line)
            preview_line = None
    
    def cancel_shape():
        nonlocal current_points
        clear_current_drawing()
        current_points = []
        set_status("Shape cancelled. " + points_status())
    
    # Adds a single vertex (in canvas space) to the shape in progress
    def add_vertex_point(x, y):
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        if len(current_points) > 0:
            last_x, last_y = current_points[-1]
            edge_lines.append(
                canvas.create_line(last_x, last_y, x, y, fill="red", width=2))
        current_points.append((x, y))
        vertex_markers.append(
            canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", outline=""))
        set_status(points_status())
    
    # Event handler for placing a new vertex, or closing the shape if the
    # click lands near the first vertex
    def on_canvas_click(event):
        if len(current_points) >= min_vertices:
            first_x, first_y = current_points[0]
            if ((event.x-first_x)**2 + (event.y-first_y)**2
                    <= CLOSE_RADIUS**2):
                attempt_close()
                return
        add_vertex_point(event.x, event.y)
    
    # Event handler for mouse motion: updates the crosshair lines and the
    # rubber-band preview line from the last placed vertex to the cursor
    def on_mouse_motion(event):
        nonlocal preview_line
        canvas.coords(vertical_line, event.x, 0, event.x, height)
        canvas.coords(horizontal_line, 0, event.y, width, event.y)
        if current_points:
            last_x, last_y = current_points[-1]
            if preview_line is None:
                preview_line = canvas.create_line(
                    last_x, last_y, event.x, event.y,
                    fill="red", dash=(2, 2))
            else:
                canvas.coords(preview_line, last_x, last_y, event.x, event.y)
            # highlight the first vertex when close enough to snap-close
            if len(current_points) >= min_vertices:
                first_x, first_y = current_points[0]
                near_start = ((event.x-first_x)**2 + (event.y-first_y)**2
                             <= CLOSE_RADIUS**2)
                canvas.itemconfig(vertex_markers[0],
                                  fill="yellow" if near_start else "red")
    
    def undo_point():
        if not current_points:
            set_status("No points to undo.")
            return
        current_points.pop()
        canvas.delete(vertex_markers.pop())
        if edge_lines:
            canvas.delete(edge_lines.pop())
        set_status("Removed last point. " + points_status())
    
    # Finalise the shape in progress, provided it has enough vertices and 
    # there is room left for another ROI
    def attempt_close():
        nonlocal current_points, preview_line
        if len(current_points) < min_vertices:
            set_status(f"Need at least {min_vertices} points to close a "
                      f"shape ({len(current_points)} so far).")
            return
        if len(rois) >= n:
            set_status(f"Too many selections, expected: {n}. "
                      "Overwrite a selection first.")
            return
        
        # bound and round the vertices, then flatten to [x1, y1, x2, y2, ...]
        bounded_points = [(max(0, min(width, int(x))),
                          max(0, min(height, int(y))))
                         for x, y in current_points]
        flat_coords = [value for point in bounded_points for value in point]
        
        # draw the finished polygon as a single outlined shape
        shapes.append(canvas.create_polygon(*flat_coords, outline="green",
                                            fill="", width=2))
        rois.append(flat_coords)
        
        clear_current_drawing()
        current_points = []
        
        converted = (np.array(flat_coords)
                    * len(image_array) / width)
        set_status(f"Saved ROI with {len(bounded_points)} vertices "
                  f"{converted}. {n-len(rois)} left")
    
    # Button callback to finish the ROI selection and close the window
    def finish():
        # If there's a shape in progress, try to close and save it.
        if current_points:
            attempt_close()
        if len(rois) < n:
            set_status(f"{n - len(rois)} selection(s) remaining. "
                      + points_status())
        else:
            root.destroy()
    
    def overwrite():
        # Discard any shape currently in progress.
        cancel_shape()
        if rois and shapes:
            # Remove the last saved ROI and its polygon
            canvas.delete(shapes[-1])
            rois.pop()
            shapes.pop()
            set_status("Overwritten ROI")
        else:
            set_status("No regions of interest saved")
    
    def select_all():
        cancel_shape()
        for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
            add_vertex_point(x, y)
        attempt_close()
    
    # Bind mouse events to the canvas
    canvas.bind("<ButtonPress-1>", on_canvas_click)
    canvas.bind("<Motion>", on_mouse_motion)
    canvas.config(cursor="none")
    canvas.bind("<Enter>", lambda event: canvas.config(cursor="none"))
    canvas.bind("<Leave>", lambda event: canvas.config(cursor=""))
    
    # Keyboard shortcuts: Enter closes the shape, Escape cancels it, and
    # Backspace removes the last placed vertex
    root.bind("<Return>", lambda event: attempt_close())
    root.bind("<Escape>", lambda event: cancel_shape())
    root.bind("<BackSpace>", lambda event: undo_point())
    
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, pady=10)
    
    undo_button = tk.Button(button_frame, text="Undo Point", command=undo_point)
    undo_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    close_button = tk.Button(button_frame, text="Close Shape", command=attempt_close)
    close_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    overwrite_button = tk.Button(button_frame, text="Overwrite", command=overwrite)
    overwrite_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    all_button = tk.Button(button_frame, text="Select Entire Frame", command=select_all)
    all_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    finish_button = tk.Button(button_frame, text="Finish", command=finish)
    finish_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
    
    # Create the status bar below the buttons
    status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X, padx=2, pady=2)
    set_status(points_status())
    
    root.mainloop()
    rois_converted = [[float(value) for value in
                      (np.array(roi) * len(image_array) / width)]
                     for roi in rois]
    return rois_converted
