# Instructions

## Data Collection

- Access the `fri@hakuin` lab machine and nagivate under the `fri` directory.
- At the start of each new terminal window, `cd` into the `robot_learning_ws` directory and run `./intera.sh` and `source devel/setup.bash`.
- Run `roscore`.
- Power on and in another terminal window enable the Sawyer robot.
- Ensure the BariFlex gripper is plugged into power and powered on.
- Run `sudo chmod 666 /dev/ttyACM0` to gain permissions to access the `ttyACM0` port.
- Run `rosrun rosserial_python serial_node.py /dev/ttyACM0`. At this point Sawyer and BariFlex topics should both be being published.
- In another terminal window, ensure all dependencies in the `requirements.txt` file are installed (Note: in the `fri@hakuin` lab machine we stored our virtual environment in `~/bari` so the virtual environment should be all setup by running `source ~/bari/bin/activate`).
- Run the `data_recorder.py` script (in `fri@hakuin` this can be found in our `FRI_bariflex-force` directory).
- After specifying the output path and time interval, data recording should begin!
- Use the `rostopic pub /bariflex_motion std_msgs/UInt16 1` and `rostopic pub /bariflex_motion std_msgs/UInt16 2` commands in another terminal window to open and close the gripper.

## Training

- Ensure all dependencies in the `requirements.txt` file are installed. We assume that training data is stored in the format output by the `data_recorder.py` script in a file named `data.hdf5`.
- Run the `train.py` script.

# Results

- Data collected is in the `FRI_bariflex-force` directory in the `fri@hakuin` lab machine.
- Model trained *without* IQ current observations after 500 epochs:
  - Minimum Training Loss: 1.1958 on epoch 489
  - https://drive.google.com/file/d/1IY-2Xgd2tq40_sjkffUP-vEN1KvgNs9O/view?usp=sharing
- Model trained *with* IQ current observations after 500 epochs:
  - Minimum Training Loss: 1.1453 on epoch 496
  - https://drive.google.com/file/d/1mOGbpq4jAEmcLqoMb0UqoTjASLmSR4pA/view?usp=sharing
