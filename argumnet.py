import argparse
ap= argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="path to input image")
args = vars(ap.parse_args())
print(args)