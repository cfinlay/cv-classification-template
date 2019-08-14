"""Python script to automate the execution of shell scripts

Launch jobdispatcher.py on a server and let it run in the background.

It creates a directory in the location it was launched from
with the gpu number specified by the argumnet gpu, e.g. 'gpu1'.

It then creates three subdirectories in that directory:
- queue
- completed
- failed

Then it waits for shell scripts (extension .sh) to appear in the queue
directory. It checks every second and when it sees one or more they are
added to a queue and each job is executed one after the other.

To start running jobs, copy script files (e.g. mnist.sh) to the queue
directory.

First the script is moved to the current directory then executed with
subprocess.run. When complete, it is moved to complete.

Log output is written to logfile.txt.

Edit:
The job dispatcher will now send processes to all 3 GPUs (one will be
blocked off for debugging purposes)
"""

import subprocess
import datetime
import time
import os
import shutil
from collections import deque
import logging
import argparse

# before anything, we're going to check which server we're on
sp = subprocess.Popen(['hostname'], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
out = sp.communicate()
host = out[0].decode('utf-8').split('\n')[0]


# Parse input arguments with argparse
parser = argparse.ArgumentParser('Python script to automate the execution of '
                                 'shell scripts')
parser.add_argument('--blocked-gpu', type=int, nargs='*',
        default=[], metavar='GPU', help='GPU ids to be blocked (default: None)' )
parser.add_argument('--run-dir', type=str, required=True, metavar='DIR',
                    help='Directory where scripts will run')
parser.add_argument('--job-dir', type=str, default=None, metavar='DIR',
        help='Directory name for jobdispatcher queues (default: "./jobdispatcher/")')
parser.add_argument('--wait', type=int, default=60, metavar='WAIT',
                    help='Wait time (seconds) between checks for new items '
                    'in queue (default: 60 seconds)')
args = parser.parse_args()

# Create subdirectories
if args.job_dir:
    base = args.job_dir
else:
    base = os.path.join(".", "jobdispatcher")


# Create general gpu folder with queue, completed, failed subfolders

queue_path = base + "/queue"
execution_path = args.run_dir
completed_jobs_path = base + "/completed"
failed_jobs_path = base + "/failed"
logfile_path = base

for directory in [queue_path, execution_path, completed_jobs_path,
                  failed_jobs_path, logfile_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Log messages go to log file
logging.basicConfig(filename=os.path.join(logfile_path, 'logfile.txt'),
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

logging.info("------------ Job Dispatcher Started ------------")

def get_script_filenames(path=None):
    """Return a list containing the file names of any shell scripts
    in the specified directory (i.e. filenames ending in '.sh').

    If path is None, uses path='.'.  For other details on path, see
    os.listdir() documentation.
    """

    return [fname for fname in os.listdir(path) if fname.endswith('.sh')]

def get_GPUs():
    # Blocked off GPU used for debugging purposes
    if len(args.blocked_gpu) >0:
        blocked_GPU = [str(g) for g in args.blocked_gpu]
    else:
        blocked_GPU=[]

    # Nvidia-smi's process monitoring system with count 1
    sp = subprocess.Popen(['nvidia-smi', 'pmon', '-c', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = list(filter(None, out_str[0].decode("utf-8").split('\n')))

    gpu_info = []
    # We skip the first two lines because it has the column names
    for line in out_list:
        exploded_info = [x for x in line.split(' ') if x != '']
        gpu_info.append(exploded_info)

    available_GPUs = []
    for gpu in gpu_info:
        gpu_id = gpu[0]
        gpu_pid = gpu[1]

        # this is what nvidia-smi pmon -c 1 uses to say that there's no process
        # thankfully, if nvidia-smi pmon -c 1 screws up, which it might
        # this will skip over the column identifiers
        if gpu_pid == '-':
            available_GPUs.append(gpu_id)

    if len(blocked_GPU)>0:
        for g in blocked_GPU:
            if g in available_GPUs:
                available_GPUs.remove(g)

    return available_GPUs

def check_jobs():
    """ Void function that checks the global running jobs list
    and checks if any are completed and to remove them from the list.

    The function also does the necessary logging, and moving of files to the
    completed/failed job subdirectories as done before.
    """

    if running_jobs:
        for job_name in list(running_jobs.keys()):
            process = running_jobs[job_name]
            if process.poll() is not None:

                logging.info('Found completed process... Checking '
                'return code.')

                if process.poll() == 0:
                    logging.info("%s returned %d", job_name,
                    process.poll())
                    completed_processes['completed'].append(job_name)

                else:
                    logging.warning("%s returned %d", job_name,
                    process.poll())
                    completed_processes['failed'].append(job_name)

                # Remove job from running jobs
                del running_jobs[job_name]

        # Move script file to final destination
        for job in completed_processes['failed']:
            subprocess.call(['mv', os.path.join(execution_path, job),
            failed_jobs_path])
            completed_processes['failed'].remove(job)

        for job in completed_processes['completed']:
            subprocess.call(['mv', os.path.join(execution_path, job),
            completed_jobs_path])
            completed_processes['completed'].remove(job)

job_queue = deque()
time_now = datetime.datetime.now()

# This will be used for keeping track of the processes we send out and find out
# which ones have been completed/failed
running_jobs = {}
completed_processes = {'failed': [], 'completed': []}

while True:

    # Let the process kick in
    time.sleep(60)

    files_now_in_queue = get_script_filenames(queue_path)

    if not files_now_in_queue:
        logging.info("Job queue empty. Waiting...")

        while not files_now_in_queue:
            time.sleep(args.wait)
            files_now_in_queue = get_script_filenames(queue_path)

            # Check for completed jobs in the mean time
            check_jobs()

    available_GPUs = get_GPUs()

    # Check if there are any available GPUs
    if not available_GPUs:
        logging.info("No Available GPUs. Waiting...")

        while not available_GPUs:
            time.sleep(args.wait)
            available_GPUs = get_GPUs()

            # Check for completed jobs in the mean time
            check_jobs()

    # Do some checking on how many GPUs to release the process to
    next_GPU = available_GPUs[0]

    logging.info("Next available GPU is #%s", next_GPU)

    # Check for new script files added to queue
    any_new_jobs = sorted(list(set(get_script_filenames(queue_path)) -
                          set(job_queue)))
    if any_new_jobs:
        job_queue.extend(any_new_jobs)
        logging.info("%d new jobs added to queue %s", len(any_new_jobs),
                     any_new_jobs.__repr__())

    # Check if any script files removed from queue
    any_jobs_removed = set(job_queue) - set(files_now_in_queue)
    if any_jobs_removed:
        logging.info("Detected %d jobs removed from queue",
                     len(any_jobs_removed))
        for job in any_jobs_removed:
            job_queue.remove(job)

    current_job = job_queue.popleft()

    shutil.move(os.path.join(queue_path, current_job),
                os.path.join(execution_path, current_job))

    # Execute next script in queue
    logging.info("Running job %s on GPU #%s...", current_job, next_GPU)

    # Make the next available GPU visible to the server
    env_vars = {'CUDA_VISIBLE_DEVICES': next_GPU}

    os.environ.update(env_vars)
    try:
        running_jobs[current_job] = subprocess.Popen([os.path.join(execution_path, current_job)], cwd=execution_path)
    except PermissionError as err:
        logging.warning("PermissionError: {0}".format(err))
        # Will get moved in upper code on next iteration
        completed_processes['failed'].append([current_job])
