import argparse
import numpy as np

from parser import parse
from utils import write_img





def main():
    parser = argparse.ArgumentParser(description='render a 3D scene to a 2D image')
    parser.add_argument('scene',   metavar='scene',   help='The input scene definition path', type=str)
    parser.add_argument('output',  metavar='output',  help='The output image path',           type=str)
    parser.add_argument('width',   metavar='width',   help='The output image path width',     type=int, nargs='?', default=500)
    parser.add_argument('height',  metavar='height',  help='The output image path height',    type=int, nargs='?', default=500)
    args = parser.parse_args()

    print(args)
    parsed = parse(args.scene)
    print(parsed)

    img = np.zeros([args.height, args.width, 3], dtype=np.float32) # converted to uint8 before saving

    # TODO cast rays

    write_img(img, args.output)


if __name__ == '__main__':
    main()
