# from dependencies import *
import dataload
import model
import argparse
parser = argparse.ArgumentParser(description='Calculates volume of a Cylinder')
parser.add_argument('-lr', '--lr', type=float, metavar='', required=True, help='Learning Rate of CNN Model')
parser.add_argument('-s', '--batch_size', type=int, metavar='', required=True, help='Height of Cylinder')
args = parser.parse_args()
print(args)
#Load training set data
'''
--lr : learning rate
○ --batch_size : size of the batch to be used in mini-batch SGD
○ --save_dir : directory to save the model
○ --opt : 1 for Adam, 2 for RMSProp
○ --dataug: 1 for True, 2 for False
○ --test: 1 when you need to test, 0 when you need to train
---------------------------------------------------------
def cylinder_volume(radius, height):
        vol = math.pi * (radius**2) * height
        return vol

if __name__ == '__main__':
        volume = cylinder_volume(args.radius, args.height)
        if args.quiet:
                print(volume)
        elif args.verbose:
                print("Volume of a Cylinder with Radius {} and height {} is ".format(args.radius, args.h$
        else:
                print("Volume of a Cylinder = {}".format(volume))



'''

#Call the CNN model and train under train()
#Print results for the test set under the best hyperparamter setting under test()

def train():
	pass

def test():
	pass