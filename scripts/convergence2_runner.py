import os
from pathlib import Path
import subprocess





import io
import selectors
import subprocess
import sys
def capture_subprocess_output(subprocess_args):
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = subprocess.Popen(subprocess_args,
                               bufsize=2,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True)

    # Create callback function for process output
    buf = io.StringIO()
    def handle_output(stream, mask):
        # Because the process' output is line buffered, there's only ever one
        # line to read when this function is called
        line = stream.readline()
        buf.write(line.decode("utf-8"))
        sys.stdout.buffer.write(line)
        pass

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = (return_code == 0)

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)









import re

to_run=["katz", "pagerank", "jaccard", "eigenvector", "hits"]
datasets = ["karate",
            "phonecalls",
            "HepPh",
            "AstroPh",
            "web-Google",
            "soc-Pokec"
           ]

if "-s" in sys.argv:
    datasets=[]
folder = Path(".")
outfile="convergence2_run_results.txt"

for suffix in to_run:
    file_name = Path("convergence2_"+suffix+".ipynb")
    file = folder/file_name
    if not file.is_file():
        folder = Path("./scripts")
        file = folder/file_name
        if not file.is_file():
            print("did not find: "+str(file))
            continue
    command = "jupyter nbconvert --to script "+str(file)
    print(command)
    os.system(command)
print("conversion done!")

for suffix in to_run:
    print("<<< testing: "+suffix +" >>>")
    python_name = folder/Path("convergence2_"+suffix+".py")

    command = "python "+str(python_name)+" datasets "+str(["karate"]).replace(" ", "").replace("'", r"\'")+ " n 1"
    os.system(command)
print("tests done!")

long_result = False

for suffix in to_run:
    if long_result:
        print("\n\n")
    print("<<< "+suffix +" >>>")
    python_name = folder/Path("convergence2_"+suffix+".py")

    command = "python "+str(python_name)+" datasets "+str(datasets).replace(" ", "").replace("'", r"\'")
    print(command)
    _, result = capture_subprocess_output(command)
    #subprocess.run([command], capture_output=True, text=True, shell=True).stdout
    #print(result)
    long_result= result.count('\n') >5
    res = re.search(r"\d\d\d\d_\d\d_\d\d__\d\d_\d\d_\d\d", result)
    if res:
        with open(folder/outfile, "a", encoding="utf-8") as f:
            f.write(suffix+ "\t"+str(res.group())+"\n")
    else:
        print("<<< no output file produced")
#print(repr(lines))