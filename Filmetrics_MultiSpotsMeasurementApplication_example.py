import serial
import time
from threading import Event
import csv
import subprocess
import re


BAUD_RATE = 115200
GRBL_port_path = 'COM13'  #Change this to the desired serial port!

# Mode 1: Glass plate samples
x_init_1 = 16.805                                                                                                     # set x position of well A1, change to fit to your device!
y_init_1 = -7.919                                                                                                     # set y position of well A1, change to fit to your device!
offset_1 = 0.5                                                                                                         # set distance between wells, change to fit to your device!
z_up_1 = "8.419"

# Mode 2: 96-well plate samples
x_init_2 = 199.463
y_init_2 = 37.441
offset_2 = 9.000
z_up_2 = "-11.590"

height_offset = 1.5
h_speed = "2000"
v_speed = "10"

RESULT_CSV = r"C:\Users\KABLab\Desktop\Filmetrics Results-Adedire\sampleretry 1-09.16.2025.csv"

with open(RESULT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Well", "X", "Y", "Thickness (nm)", "GOF"])

def remove_comment(string):
    return string.split(';')[0] if ';' in string else string

def remove_eol_chars(string):
    # removed \n or traling spaces
    return string.strip()

def send_wake_up(ser):
    # Wake up
    # Hit enter a few times to wake the cnc
    ser.write(str.encode("\r\n\r\n"))
    time.sleep(2)  # Wait for cnc to initialize
    ser.flushInput()  # Flush startup text in serial input

def wait_for_movement_completion(ser, cleaned_line):    #wait for cnc to reach destination before sending new movement
    Event().wait(1)
    if cleaned_line != '$X' or '$$':
        idle_counter = 0
        while True:
            # Event().wait(0.01)
            ser.reset_input_buffer()
            ser.write(str.encode('?\n'))
            grbl_out = ser.readline()
            grbl_response = grbl_out.strip().decode('utf-8')
            if 'Idle' in grbl_response:
                idle_counter += 1
            if idle_counter > 10:
                break

def stream_gcode(GRBL_port_path, gcode, home, x, y, z):
    with serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
        send_wake_up(ser)
        cleaned_line = remove_eol_chars(remove_comment(gcode))

        if home:  #if device is being moved home, run this to reset coordinate system
            ser.write(str.encode(f'G92 X{x} Y{y} Z{z} \n')) # Send g-code
            wait_for_movement_completion(ser, cleaned_line)
            ser.readline() # Wait for response with carriage return

        if cleaned_line:   # checks if string is empty
            ser.write(str.encode(gcode + '\n'))
            wait_for_movement_completion(ser, cleaned_line)
            ser.readline()
            position = [x, y, z]

        with open("position.csv", 'w') as csvfile:
            csv.writer(csvfile).writerow(position)

def get_well_coordinates(well_id):
    """Input like A1, B2, return real x, y coordinates"""
    cols = "ABCDEFGHIJKL"
    rows = [str(i) for i in range(1, 13)]

    col = well_id[0].upper()
    row = well_id[1:]

    if col not in cols or row not in rows:
        raise ValueError(f"Invalid coordinate: {well_id}")

    x = x_init + offset * cols.index(col)
    y = y_init + offset * (int(row) - 1)
    return round(x, 3), round(y, 3)

def send_command(cmd, well_id=None, x=None, y=None):
    print(f">>> Sending: {cmd}")
    process.stdin.write(cmd + "\n")
    process.stdin.flush()

    thickness, gof = None, None

    # Wait for C# response
    while True:
        line = process.stdout.readline()
        if not line:
            break
        line = line.strip()
        print(f"[Filmetrics:] {line}")

        if "Polyimide" in line:
            thickness_match = re.findall(r"([-+]?\d*\.?\d+)\s*nm", line, re.IGNORECASE)
            if thickness_match:
                try:
                    thickness = float(thickness_match[-1])
                except:
                    thickness = None

        if "Goodness of fit" in line:
            match = re.search(r"[-+]?\d*\.\d+|\d+", line)
            if match:
                gof = float(match.group())

        line_low = line.lower()
        if "complete" in line_low or "successfully" in line_low:
            break

    if well_id and (thickness is not None and gof >= 0.6):
        with open(RESULT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([well_id, x, y, thickness, gof])

    return thickness, gof


def measure_one_sample(x: float, y: float, z: float, well_id: str, refSta: str, startPoint: bool,
                       x_reference: float, y_reference: float):

    if startPoint:
        # Move to sample
        gcode = f"G01 X{x} Y{y} Z{z} F{h_speed}"
        stream_gcode(GRBL_port_path, gcode, home=False, x=x, y=y, z=z)
        send_command("sample")

        # Move to reference
        gcode = f"G01 X{x_reference} Y{y_reference} F{h_speed}"
        stream_gcode(GRBL_port_path, gcode, home=False, x=x_reference, y=y_reference, z=z)
        send_command(f"reference {refSta}")

        # Move to background
        gcode = f"G01 X128.911 Y-6.485 F{h_speed}"
        stream_gcode(GRBL_port_path, gcode, home=False, x=170.510, y=98.208, z=z)
        send_command("background")

        # Commit baseline
        send_command("commit")

    # Back to the sample
    gcode = f"G01 X{x} Y{y} F{h_speed}"
    stream_gcode(GRBL_port_path, gcode, home=False, x=x, y=y, z=z)
    thickness, gof = send_command("measure", f"{well_id}", x, y)

    # Save result
    send_command(f"save {well_id}")

    return thickness, gof


def shutdown():
    send_command("exit")
    process.stdin.close()
    process.stdout.close()
    process.wait()

if __name__ == "__main__":

    RecipeName = input("Please input the recipe name: ")
    RefSta = RecipeName.split()[-1]

    while True:
        try:
            Mode = int(input("Please choose the mode type (1 or 2): "))
            if Mode == 1 :
                x_init = x_init_1
                y_init = y_init_1
                offset = offset_1
                z_up = z_up_1
                x_Reference = 54.111                                                                                   # set x reference coordinates
                y_Reference = -35.640                                                                                   # set y reference coordinates
                print(f"Valid Mode")
                break
            elif Mode == 2:
                x_init = x_init_2
                y_init = y_init_2
                offset = offset_2
                z_up = z_up_2
                y_Reference = y_init
                print(f"Valid Mode")
                break
            else:
                print("Error: please input 1 or 2")
        except ValueError:
            print("Error: please input an integer (1 or 2): ")

    # start C# console program
    process = subprocess.Popen(
        ["C:/Users/KABLab/Desktop/Filmetrics Framework/Filmetrics/bin/Debug/FilmetricsTool.exe", RecipeName],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='mbcs',
        bufsize=1
    )

    # wells = []
    #
    # while True:
    #     wells_input = input("Input the well position (like A1 B2 C3), Enter: ")
    #     if wells_input.strip() == "":
    #         break
    #     wells_list = wells_input.strip().split()
    #     valid_wells = []
    #     for well in wells_list:
    #         try:
    #             get_well_coordinates(well)  # Verify the input coordinates
    #             valid_wells.append(well.upper())
    #         except ValueError as e:
    #             print(e)
    #     wells.extend(valid_wells)
    #     print(f"Now valid wells: {wells}")

    wells = []
    cols = "ABCDEFG"
    rows = [str(i) for i in range(1, 8)]  # "1" ~ "7"

    while True:
        wells_input = input("Input the well position (like A1 B2 C3) or measure the whole plate (all), Enter: ")
        if wells_input.strip() == "":
            break

        if wells_input.strip().lower() == "all":
            wells = [f"{c}{r}" for r in rows for c in cols]
            print(f"Now measuring full plate: {wells}")
            break

        wells_list = wells_input.strip().split()
        valid_wells = []
        for well in wells_list:
            try:
                get_well_coordinates(well)  # Verify the input coordinates
                valid_wells.append(well.upper())
            except ValueError as e:
                print(e)
        wells.extend(valid_wells)
        print(f"Now valid wells: {wells}")
        break

    # StartPoint = True
    # for well in wells:
    #     x, y = get_well_coordinates(well)
    #     print(f"\n<Moving to well: {well}>")
    #     measure_one_sample(x, y, z_up, well, RefSta, StartPoint, x_Reference, y_Reference)
    #     StartPoint=False
    #     time.sleep(2)

    StartPoint = True

    for well in wells:
        x, y = get_well_coordinates(well)
        print(f"\n<Moving to well: {well}>")

        attempt = 0
        gof = None
        success = False

        while attempt < 2:
            attempt += 1
            print(f"--- Measuring {well}, attempt {attempt} ---")
            thickness, gof = measure_one_sample(x, y, z_up, well, RefSta, StartPoint, x_Reference, y_Reference)

            if gof is not None and gof >= 0.6:
                print(f"✅ Well {well} successful, GOF={gof:.5f}")
                success = True
                StartPoint = False
                break
            else:
                print(f"⚠️ Well {well} failed, GOF={gof} < 0.6, redoing baseline...")
                StartPoint = True
                time.sleep(2)

        if not success:
            print(f"❌ Well {well} failed 2 times, skipping to next well.")
            with open(RESULT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([well, x, y, thickness, gof])
            StartPoint = False
            continue

        time.sleep(2)

    home = True
    gcode = f"G01 X{x_init_1} Y{y_init_1} F2000"  # produce home command
    print("Returning to home position...")
    stream_gcode(GRBL_port_path, gcode, home, x=x, y=y, z=z_up)
    shutdown()

